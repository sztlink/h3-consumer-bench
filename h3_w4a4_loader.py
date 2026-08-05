#!/usr/bin/env python3
"""H3 W4A4 L4b: loader nunchaku para o MiniMaxH3Transformer3DModel (diffusers).

Porta do nunchaku_causal_wan.load_w4a4_blocks. Troca os 6 linears diffusers de
cada bloco (attn.to_q/to_k/to_v/to_out.0, ff.net.0.proj, ff.net.2) por
SVDQW4A4Linear carregado do checkpoint do h3_convert.py. Norms, adaln_proj e os
modulos fora dos blocos ficam como estao (a estrategia de residencia deles e do
runtime, nao do loader).

O H3 opera em sequencia EMPACOTADA 2D (tokens, canais); o forward do nunchaku
exige (B, S, C), entao os modulos entram por um wrapper que promove 2D a 3D.

Import cirurgico do nunchaku (mesmo truque do Wan): o __init__ do pacote puxa
modelos flux que pedem outro diffusers;so nunchaku.models.linear e carregado.
"""
import json
import sys
import types
from pathlib import Path

import torch


def _import_svdq_linear():
  site = Path(torch.__file__).parent.parent
  np = site / "nunchaku"
  for name, sub in [("nunchaku", ""), ("nunchaku.models", "models"),
                    ("nunchaku.ops", "ops")]:
    if name not in sys.modules:
      m = types.ModuleType(name)
      m.__path__ = [str(np / sub)]
      sys.modules[name] = m
  import nunchaku.models.linear as nl
  return nl.SVDQW4A4Linear


SVDQW4A4Linear = _import_svdq_linear()


class SVDQ2D(SVDQW4A4Linear):
  """SVDQW4A4Linear que aceita a sequencia empacotada 2D do H3."""

  def forward(self, x, output=None):
    if x.ndim == 2:
      return super().forward(x.unsqueeze(0)).squeeze(0)
    return super().forward(x, output)


RENAMES = [(".lora_down", ".proj_down"), (".lora_up", ".proj_up"),
           (".smooth_orig", ".smooth_factor_orig"), (".smooth", ".smooth_factor")]


def _rename(key):
  for old, new in RENAMES:
    if key.endswith(old):
      return key[: -len(old)] + new
  return key


def load_w4a4_blocks(tr, ckpt_dir, device="cuda", torch_dtype=torch.bfloat16):
  """Troca os linears quantizados de cada bloco do transformer H3 por SVDQ2D
  carregado do checkpoint convertido. `tr` e o MiniMaxH3Transformer3DModel."""
  from safetensors import safe_open
  sd = {}
  with safe_open(str(Path(ckpt_dir) / "transformer_blocks.safetensors"),
                 framework="pt") as f:
    meta = f.metadata() or {}
    for k in f.keys():
      sd[_rename(k)] = f.get_tensor(k)
  rank = 32
  if "quantization_config" in meta:
    rank = json.loads(meta["quantization_config"]).get("rank", 32)

  slots = sorted({k[len("transformer_blocks.0."):].rsplit(".qweight", 1)[0]
                  for k in sd
                  if k.startswith("transformer_blocks.0.") and k.endswith(".qweight")})
  n_swapped = 0
  for i, blk in enumerate(tr.transformer_blocks):
    prefix = f"transformer_blocks.{i}."
    for local in slots:
      parent = blk
      *path, leaf = local.split(".")
      for p in path:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
      chaves = [k for k in sd if k.startswith(prefix + local + ".")]
      sub = {k[len(prefix) + len(local) + 1:]: sd.pop(k) for k in chaves}
      assert sub, f"sem tensores para {prefix}{local}"
      in_f = sub["smooth_factor"].shape[0]
      out_f = sub["qweight"].shape[0]
      q = SVDQ2D(in_f, out_f, rank=rank, bias="bias" in sub,
                 precision="int4", torch_dtype=torch_dtype, device=device)
      q.load_state_dict({k: v.to(device) for k, v in sub.items()})
      if leaf.isdigit():
        parent[int(leaf)] = q
      else:
        setattr(parent, leaf, q)
      n_swapped += 1
  torch.cuda.empty_cache()
  print(f"[w4a4] {n_swapped} linears trocados em {len(tr.transformer_blocks)} blocos "
        f"(rank {rank})", flush=True)
  return tr
