#!/usr/bin/env python3
"""H3 fase B: pipeline parcial (denoise + decode) com o transformer 33B int8,
consumindo o estado salvo pela fase A. O encoder de 32B nunca entra na RAM aqui.

Uso: h3_denoise.py <cena> [--steps padrao-do-modelo]
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3

EMB = os.path.expanduser("./h3-lab/embeds")


def log(m):
  print(f"[h3den] {m}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cena", choices=sorted(CENAS_H3))
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--saida", default=None)
  args = ap.parse_args()

  estado = torch.load(f"{EMB}/{args.cena}.pt", weights_only=False)
  log(f"estado da fase A: {sorted(estado)}")

  from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
  from diffusers.modular_pipelines import SequentialPipelineBlocks
  from torchao.quantization import Int8WeightOnlyConfig

  t0 = time.time()
  base = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
  quer = ["denoise", "decode"]
  sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in quer}
  pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline("MiniMaxAI/MiniMax-H3")
  log(f"pipeline parcial: {list(sub)}; componentes: {getattr(pipe, 'component_names', '?')}")

  tr = MiniMaxH3Transformer3DModel.from_pretrained(
    "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16,
    quantization_config=TorchAoConfig(
      Int8WeightOnlyConfig(version=2),
      modules_to_not_convert=[
        "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
        "token_refiner", "norm_out", "proj_out", "audio_proj_out",
      ],
    ),
  )
  log(f"transformer int8 em RAM em {time.time()-t0:.0f}s")
  # transformer_ref e o irmao do ref2va (62GB bf16 que NAO cabem): apelida para a
  # mesma instancia int8; o caminho t2va/i2va nao o executa, so fareja config.
  pipe.update_components(transformer=tr, transformer_ref=tr)
  restante = [n for n in getattr(pipe, "component_names", [])
              if n not in ("transformer", "transformer_ref")]
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)
    log(f"restante carregado: {restante}")

  # use_stream=True fixa (pin) dezenas de GB na RAM e o kernel mata o processo
  # sem traceback; sem stream a transferencia e mais lenta mas sobrevive.
  offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                 use_stream=False)
  pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                                        **offload)
  pipe.transformer.requires_grad_(False)
  for nome_vae in ("vae", "video_vae", "audio_vae"):
    v = getattr(pipe, nome_vae, None)
    if v is not None:
      v.to("cuda")
  log(f"fase B pronta em {time.time()-t0:.0f}s")

  # tensores ficam em CPU: os blocos criam internos no proprio device e movem
  # o que precisam; empurrar para cuda aqui causa mismatch no build_packed_sequence
  kwargs = {k: v for k, v in estado.items()}
  kwargs["generator"] = torch.Generator().manual_seed(2047)
  # o H3 gera 5-15s a 24fps: num_frames valido = 17n+5 entre 120 e 360
  kwargs["num_frames"] = max(124, int(kwargs.get("num_frames") or 0))
  if args.steps:
    kwargs["num_inference_steps"] = args.steps
  kwargs["output"] = ["videos", "audio", "sampling_rate"]

  t = time.time()
  res = pipe(**kwargs)
  log(f"gerado em {time.time()-t:.0f}s")
  if os.environ.get("H3_DEBUG") == "1":
    fonte = res if isinstance(res, dict) else getattr(res, "__dict__", {})
    for k, v in dict(fonte).items():
      if torch.is_tensor(v):
        log(f"  saida[{k}]: tensor {tuple(v.shape)} {v.dtype} {v.device}")
      elif isinstance(v, (list, tuple)):
        log(f"  saida[{k}]: {type(v).__name__} len={len(v)}"
            + (f" [0]={tuple(v[0].shape) if torch.is_tensor(v[0]) else type(v[0]).__name__}" if len(v) else ""))
      else:
        log(f"  saida[{k}]: {type(v).__name__} = {str(v)[:80]}")

  saida = args.saida or os.path.expanduser(f"./h3-lab/{args.cena}.mp4")
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
      while arr.ndim > 2:  # (1, 2, N) -> (2, N)
        arr = arr[0]
      sf.write(saida.replace(".mp4", ".wav"), arr.T.astype("float32"), sr)  # (N, 2) estereo
      log(f"audio salvo ({sr}Hz)")
    except Exception as e:
      log(f"audio nao salvo: {type(e).__name__}: {e}")
  log(f"saida: {saida}")


if __name__ == "__main__":
  main()
