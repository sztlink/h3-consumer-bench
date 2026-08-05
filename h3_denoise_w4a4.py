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
    table:             COLAPSO EM TABELA. Os sigmas do run sao deterministicos
                       (schedulers video shift 12 e audio shift 3; sonda real em
                       /tmp/h3-ts.tsv confirmou t=0 compartilhado e 2 niveis por
                       passo); pre-computa as 6 saidas do adaln de cada bloco
                       para todos os sigmas do schedule (+constantes 0 e 1,
                       ~0.9GB na GPU, UMA passada de streaming no setup) e troca
                       adaln_proj por lookup keyed pela linha do temb. O denoise
                       nao toca mais os 13B.
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
  elif modo_adaln == "table":
    # schedulers construidos da CLASSE (o pipe ainda nao carregou os dele neste
    # ponto, e getattr None deixava so as constantes na tabela). Grid default do
    # modular = 50 pontos, 49 avaliacoes; timesteps do forward = 1 - sigmas[:-1].
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
    t0 = time.time()
    n_grid = args.steps or 50
    cands = {0.0, 1.0}
    for shift in (12.0, 3.0):  # video e audio (scheduler_config.json de cada um)
      sc = MiniMaxH3Scheduler(shift=shift)
      sc.set_timesteps(n_grid)
      cands.update(float(v) for v in sc.timesteps.tolist())
    ts_all = torch.tensor(sorted(cands), dtype=torch.float32, device="cuda")
    with torch.no_grad():
      temb_ref = tr.time_embedder(
        tr.time_proj(ts_all).to(tr.time_embedder.linear_1.weight.dtype))

    class AdalnTabela(torch.nn.Module):
      """Substitui adaln_proj por lookup nas saidas pre-computadas. A chave e a
      propria linha do temb (vizinho mais proximo com limiar); ordem das linhas
      da tabela = [t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...], a mesma do modulo
      real, entao o index_select preserva o contrato do bloco."""

      def __init__(self, temb_ref, saidas):
        super().__init__()
        self.temb_ref = temb_ref
        self.saidas = saidas
        self.m = torch.arange(3, device=temb_ref.device)

      def forward(self, temb):
        d = torch.cdist(temb.float(), self.temb_ref)
        prox, idx = d.min(dim=1)
        if bool((prox > 1e-2).any()):
          raise RuntimeError(f"temb fora da tabela (dist {prox.max().item():.4f}), "
                             "schedule inesperado; rode com H3_ADALN=offload")
        linhas = (idx[:, None] * 3 + self.m[None, :]).flatten()
        return tuple(s.index_select(0, linhas) for s in self.saidas)

    # tabela construida DIRETO dos shards bf16 da raiz: matematica exata do
    # modulo real (silu fp32 -> cast bf16 -> linear -> view -> chunk 6), sem o
    # caminho int8 (que alem de vazar buffers na build, degradaria a tabela).
    import json as _json
    from safetensors import safe_open as _so
    snap = os.path.expanduser(
      "~/realtime-diffusion/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
      "fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/transformer")
    idx = _json.load(open(os.path.join(
      snap, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    x_ref = torch.nn.functional.silu(temb_ref).to(torch.bfloat16)
    abertos = {}

    def _tensor(nome):
      sh = idx[nome]
      if sh not in abertos:
        abertos[sh] = _so(os.path.join(snap, sh), framework="pt")
      return abertos[sh].get_tensor(nome)

    with torch.no_grad():
      for i, blk in enumerate(tr.transformer_blocks):
        p = f"transformer_blocks.{i}.adaln_proj.linear"
        W = _tensor(f"{p}.weight").to("cuda")
        b = _tensor(f"{p}.bias").to("cuda")
        y = (x_ref @ W.T + b).view(-1, 6 * 5376)
        saidas = tuple(c.contiguous() for c in y.chunk(6, dim=-1))
        blk.adaln_proj = AdalnTabela(temb_ref, saidas)
        del W, b, y
        if i % 10 == 9:
          torch.cuda.empty_cache()
    abertos.clear()
    torch.cuda.empty_cache()
    gb = sum(s.numel() * 2 for b in tr.transformer_blocks for s in b.adaln_proj.saidas) / 1e9
    log(f"adaln colapsado em tabela: {ts_all.numel()} sigmas (amostra "
        f"{[round(v, 4) for v in ts_all[:4].tolist()]}...) x 3 modalidades, "
        f"{gb:.2f}GB, {time.time()-t0:.0f}s; denoise nao toca mais os 13B")
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
  dump_ts = os.environ.get("H3_DUMP_TS")

  def _mover(mod, args, kwargs):
    a = tuple(t.to("cuda") if torch.is_tensor(t) else t for t in args)
    k = {n: (t.to("cuda") if torch.is_tensor(t) else t) for n, t in kwargs.items()}
    if dump_ts and "timestep" in k:
      with open(dump_ts, "a") as fh:
        v = k["timestep"].detach().float().cpu()
        fh.write(f"{list(v.shape)}\t{v.flatten().tolist()}\n")
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
