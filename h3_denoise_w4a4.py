#!/usr/bin/env python3
"""H3 fase B em W4A4: denoise + decode com os 50 blocos do transformer em int4
nunchaku RESIDENTES na GPU (~10.6GB). Sucede o h3_denoise.py int8+offload.

Estrategia de residencia:
- 300 linears de bloco -> SVDQ2D na GPU (checkpoint do h3_convert.py)
- adaln_proj (13B, o unico peso grande restante) por modo, env H3_ADALN:
    offload (default): int8 na CPU com group offload leaf por bloco
                       (streaming de ~13GB/passo, contra 33GB do int8 puro)
    int4:              excluido do int8 no load (bf16), quantizado int4
                       weight-only na GPU e RESIDENTE (denoise sem streaming)
- resto do transformer (proj_in, embedders, token_refiner, cabecas: ~0.7B) bf16
  na GPU, VAEs na GPU, como antes.

Uso: h3_denoise_w4a4.py <cena> [--steps N] [--saida x.mp4]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/realtime-diffusion/hf-cache"))
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3
from h3_w4a4_loader import load_w4a4_blocks

EMB = os.path.expanduser("~/realtime-diffusion/h3-lab/embeds")
CKPT = os.path.expanduser("~/realtime-diffusion/h3-w4a4/nunchaku/minimax-h3-w4a4")


def log(m):
  print(f"[h3w4a4] {m}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cena", choices=sorted(CENAS_H3))
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--saida", default=None)
  args = ap.parse_args()
  modo_adaln = os.environ.get("H3_ADALN", "offload")

  estado = torch.load(f"{EMB}/{args.cena}.pt", weights_only=False)
  log(f"estado da fase A: {sorted(estado)}; adaln={modo_adaln}")

  from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
  from diffusers.modular_pipelines import SequentialPipelineBlocks
  from torchao.quantization import Int8WeightOnlyConfig

  t0 = time.time()
  base = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
  quer = ["denoise", "decode"]
  sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in quer}
  pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline("MiniMaxAI/MiniMax-H3")

  nao_converter = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
  ]
  if modo_adaln == "int4":
    nao_converter.append("adaln_proj")
  tr = MiniMaxH3Transformer3DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16,
    quantization_config=TorchAoConfig(
      Int8WeightOnlyConfig(version=2), modules_to_not_convert=nao_converter))
  log(f"transformer base em RAM em {time.time()-t0:.0f}s")

  # os 300 linears de bloco viram int4 residente; os originais sao liberados
  load_w4a4_blocks(tr, CKPT, device="cuda")

  # residencia do resto: tudo para a GPU, exceto adaln_proj de cada bloco
  for nome, mod in tr.named_children():
    if nome != "transformer_blocks":
      mod.to("cuda")
  for blk in tr.transformer_blocks:
    for nome, mod in blk.named_children():
      if nome != "adaln_proj":
        mod.to("cuda")
  if modo_adaln == "int4":
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    for blk in tr.transformer_blocks:
      blk.adaln_proj.to("cuda")
      quantize_(blk.adaln_proj, Int4WeightOnlyConfig(group_size=64))
    log("adaln int4 residente na GPU")
  else:
    from diffusers.hooks import apply_group_offloading
    for blk in tr.transformer_blocks:
      apply_group_offloading(blk.adaln_proj, onload_device=torch.device("cuda"),
                             offload_device=torch.device("cpu"),
                             offload_type="leaf_level", use_stream=False)
    log("adaln int8 com offload leaf")
  tr.requires_grad_(False)

  # no caminho int8+offload era o hook do group offload que movia os inputs CPU
  # para a GPU na fronteira do transformer; sem ele, ninguem move. Replica so isso.
  def _mover(mod, args, kwargs):
    a = tuple(t.to("cuda") if torch.is_tensor(t) else t for t in args)
    k = {n: (t.to("cuda") if torch.is_tensor(t) else t) for n, t in kwargs.items()}
    return a, k
  tr.register_forward_pre_hook(_mover, with_kwargs=True)

  pipe.update_components(transformer=tr, transformer_ref=tr)
  restante = [n for n in getattr(pipe, "component_names", [])
              if n not in ("transformer", "transformer_ref")]
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)
  # o video_vae do H3 tem 9.8GB (~5B): com os blocos int4 residentes ele nao
  # cabe mais fixo na GPU; leaf offload paga o streaming so no decode.
  from diffusers.hooks import apply_group_offloading as _offl
  for nome_vae in ("vae", "video_vae", "audio_vae"):
    v = getattr(pipe, nome_vae, None)
    if v is None:
      continue
    if sum(p.numel() for p in v.parameters()) > 1e9:
      _offl(v, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
            offload_type="leaf_level", use_stream=False)
      log(f"{nome_vae} grande: offload leaf")
    else:
      v.to("cuda")
  vram = torch.cuda.memory_allocated() / 1e9
  log(f"fase B W4A4 pronta em {time.time()-t0:.0f}s, VRAM {vram:.1f}GB")

  kwargs = {k: v for k, v in estado.items()}
  kwargs["generator"] = torch.Generator().manual_seed(2047)
  kwargs["num_frames"] = max(124, int(kwargs.get("num_frames") or 0))
  if args.steps:
    kwargs["num_inference_steps"] = args.steps
  kwargs["output"] = ["videos", "audio", "sampling_rate"]

  t = time.time()
  res = pipe(**kwargs)
  dt = time.time() - t
  pico = torch.cuda.max_memory_allocated() / 1e9
  log(f"gerado em {dt:.0f}s (pico VRAM {pico:.1f}GB)")

  saida = args.saida or os.path.expanduser(
    f"~/realtime-diffusion/h3-lab/{args.cena}-w4a4.mp4")
  os.makedirs(os.path.dirname(saida), exist_ok=True)
  from diffusers.utils import export_to_video
  if isinstance(res, dict):
    videos, audio, sr = res.get("videos"), res.get("audio"), res.get("sampling_rate", 32000)
  else:
    videos = getattr(res, "videos", None)
    audio = getattr(res, "audio", None)
    sr = getattr(res, "sampling_rate", 32000)
  video = videos[0] if isinstance(videos, (list, tuple)) else videos
  export_to_video(video, saida, fps=24)
  if audio is not None:
    try:
      import numpy as np
      import soundfile as sf
      arr = audio[0] if isinstance(audio, (list, tuple)) else audio
      if torch.is_tensor(arr):
        arr = arr.float().cpu().numpy()
      arr = np.asarray(arr)
      while arr.ndim > 2:
        arr = arr[0]
      sf.write(saida.replace(".mp4", ".wav"), arr.T.astype("float32"), sr)
      log(f"audio salvo ({sr}Hz)")
    except Exception as e:
      log(f"audio nao salvo: {type(e).__name__}: {e}")
  log(f"saida: {saida}")


if __name__ == "__main__":
  main()
