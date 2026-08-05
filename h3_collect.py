#!/usr/bin/env python3
"""H3 W4A4 L2: coleta de ativacoes nos 300 alvos (6 linears x 50 blocos) durante
geracao REAL, estratificada por modalidade de token (video, audio, texto).

Hooks de pre-forward calculam absmax channelwise por modalidade NA GPU (nada de
copiar o tensor inteiro) e amostram ate K linhas de token por (alvo, modalidade)
por chamada, reservatorio para a calibracao DeepCompressor. As ativacoes vem do
transformer int8 weight-only (o bf16 nao cabe no host), vies aceito e anotado:
int8 WO preserva as ativacoes em bf16 e a trajetoria fica proxima da nativa.

Uso: h3_collect.py cena1 [cena2 ...]   (default: h3-t1-t2va h3-t5-t2va h3-t4-t2va)
Saida: ~/realtime-diffusion/h3-w4a4/collect_out/{stats,reservoirs}
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3

EMB = os.path.expanduser("./h3-lab/embeds")
OUT = os.path.expanduser("~/realtime-diffusion/h3-w4a4/collect_out")
os.makedirs(OUT, exist_ok=True)
K_RES = 8          # linhas amostradas por (alvo, modalidade) por chamada
RES_MAX = 512      # teto do reservatorio por (alvo, modalidade)
ALVOS_SUFIXO = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "ff.net.0.proj", "ff.net.2")


def log(m):
  print(f"[l2] {m}", flush=True)


def main():
  cenas = sys.argv[1:] or ["h3-t1-t2va", "h3-t5-t2va", "h3-t4-t2va"]

  from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
  from diffusers.modular_pipelines import SequentialPipelineBlocks
  from torchao.quantization import Int8WeightOnlyConfig

  t0 = time.time()
  base = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
  sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in ("denoise", "decode")}
  pipe = SequentialPipelineBlocks.from_blocks_dict(sub).init_pipeline("MiniMaxAI/MiniMax-H3")
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
  pipe.update_components(transformer=tr, transformer_ref=tr)
  restante = [n for n in getattr(pipe, "component_names", [])
              if n not in ("transformer", "transformer_ref")]
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)
  offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                 use_stream=False)
  pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
                                        **offload)
  pipe.transformer.requires_grad_(False)
  for nome_vae in ("vae", "video_vae", "audio_vae"):
    v = getattr(pipe, nome_vae, None)
    if v is not None:
      v.to("cuda")
  log(f"pipeline de coleta pronto em {time.time()-t0:.0f}s")

  alvos = {n: mod for n, mod in pipe.transformer.named_modules()
           if n.startswith("transformer_blocks.") and n.endswith(ALVOS_SUFIXO)}
  log(f"{len(alvos)} alvos com hook")

  # estado da corrida corrente (indices por modalidade, preenchidos por cena)
  corrida = {"mod_idx": None, "gen": torch.Generator().manual_seed(20470)}
  stats = {n: {} for n in alvos}       # n -> mod -> {absmax cuda fp32, count}
  reservas = {n: {} for n in alvos}    # n -> mod -> lista de linhas cpu fp16

  def faz_hook(nome):
    def hook(mod, args):
      x = args[0]
      if not torch.is_tensor(x) or x.ndim != 3:
        return
      xa = x[0].float().abs()          # [S, D] na GPU
      mods = corrida["mod_idx"]
      if mods is None:
        grupos = {"tudo": slice(None)}
      else:
        grupos = mods
      for mkey, idx in grupos.items():
        seg = xa[idx] if not isinstance(idx, slice) else xa
        if seg.numel() == 0:
          continue
        am = seg.amax(dim=0)
        d = stats[nome].setdefault(mkey, {"absmax": torch.zeros_like(am), "count": 0})
        if d["absmax"].shape != am.shape:
          continue
        torch.maximum(d["absmax"], am, out=d["absmax"])
        d["count"] += 1
        # sync GPU->CPU do reservatorio so a cada 2 chamadas, senao os 300 hooks
        # estrangulam o denoise em transferencias minusculas
        if d["count"] % 2:
          continue
        r = reservas[nome].setdefault(mkey, [])
        if len(r) < RES_MAX:
          sel = torch.randint(0, seg.shape[0], (min(K_RES, seg.shape[0]),),
                              generator=corrida["gen"]).to(seg.device)
          r.extend(seg[sel].half().cpu())
    return hook

  handles = [mod.register_forward_pre_hook(faz_hook(n)) for n, mod in alvos.items()]

  for cena in cenas:
    estado = torch.load(f"{EMB}/{cena}.pt", weights_only=False)
    kwargs = {k: v for k, v in estado.items()}
    kwargs["generator"] = torch.Generator().manual_seed(2047)
    kwargs["num_frames"] = max(124, int(kwargs.get("num_frames") or 0))
    kwargs["output"] = ["videos", "video_indices", "audio_indices", "text_indices"]
    t = time.time()
    res = pipe(**kwargs)
    fonte = res if isinstance(res, dict) else getattr(res, "__dict__", {})
    vi, ai, ti = (fonte.get("video_indices"), fonte.get("audio_indices"),
                  fonte.get("text_indices"))
    if vi is not None:
      log(f"{cena}: indices v={vi.numel()} a={ai.numel()} t={ti.numel()} "
          f"(registrados; estratificacao adiada, layout varia por cena)")
    log(f"{cena}: coletada em {time.time()-t:.0f}s")
    os.makedirs(f"{OUT}/stats", exist_ok=True)
    os.makedirs(f"{OUT}/reservoirs", exist_ok=True)
    for n in alvos:
      torch.save({m: {"absmax": d["absmax"].cpu(), "count": d["count"]}
                  for m, d in stats[n].items()}, f"{OUT}/stats/{n}.pt")
      torch.save({m: torch.stack(r) if r else torch.empty(0)
                  for m, r in reservas[n].items()}, f"{OUT}/reservoirs/{n}.pt")
    log(f"{cena}: checkpoint salvo")

  for h in handles:
    h.remove()

  os.makedirs(f"{OUT}/stats", exist_ok=True)
  os.makedirs(f"{OUT}/reservoirs", exist_ok=True)
  for n in alvos:
    torch.save({m: {"absmax": d["absmax"].cpu(), "count": d["count"]}
                for m, d in stats[n].items()}, f"{OUT}/stats/{n}.pt")
    torch.save({m: torch.stack(r) if r else torch.empty(0)
                for m, r in reservas[n].items()}, f"{OUT}/reservoirs/{n}.pt")
  log(f"COLETA-COMPLETA: {len(alvos)} alvos em {OUT}")


if __name__ == "__main__":
  main()
