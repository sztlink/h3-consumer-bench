#!/usr/bin/env python3
"""H3 W4A4 L4a: converte os artefatos L3 (ptq_out/{model,scale,smooth,branch}.pt)
para o checkpoint nunchaku de bloco unico (transformer_blocks.safetensors), na
linhagem do convert_wan.py / convert_krea2.py.

Diferencas estruturais para o H3:
- model.pt e scale.pt do L3 ja estao fatiados por linear DIFFUSERS separado
  (to_q/to_k/to_v com Rq e wscales proprios); smooth e branch vivem por GRUPO
  (chave = reservatorio, to_q para o grupo qkv). Aqui o b do branch e fatiado por
  linhas (a compartilhado, mesma matematica da fatia do GEMM fundido) e o smooth
  do grupo e apontado pelos tres slots via smooth_name_map.
- cada linear vira um slot separado no checkpoint: o runtime diffusers do H3 troca
  to_q/to_k/to_v individualmente (nao ha fuse_projections como no CausalWan).
- unquantized_layers.safetensors NAO e gerado: o runtime le o resto do transformer
  dos shards originais do FL2VA (mesma convencao do L4 do Wan, que nunca leu esse
  arquivo).

Uso (na 4090):
  cd ~/realtime-diffusion/realtime-video && \
  PYTHONPATH=~/realtime-diffusion/src/deepcompressor \
  ../venv-ptq/bin/python h3_convert.py
"""
import json
import os
import time
from pathlib import Path

import safetensors.torch
import torch

from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_transformer_block_state_dict, update_state_dict)

PTQ = Path(os.path.expanduser(os.environ.get(
  "PTQ_DIR", "~/realtime-diffusion/h3-w4a4/ptq_out2")))
OUT = Path(os.path.expanduser(os.environ.get(
  "NUNCHAKU_DIR", "~/realtime-diffusion/h3-w4a4/nunchaku/minimax-h3-w4a4")))
OUT.mkdir(parents=True, exist_ok=True)
N_BLOCKS, RANK, INNER = 50, 32, 7168

LOCAL_MAP = {
  "attn.to_q": "attn.to_q",
  "attn.to_k": "attn.to_k",
  "attn.to_v": "attn.to_v",
  "attn.to_out.0": "attn.to_out.0",
  "ff.net.0.proj": "ff.net.0.proj",
  "ff.net.2": "ff.net.2",
}
SMOOTH_MAP = {
  "attn.to_q": "attn.to_q",
  "attn.to_k": "attn.to_q",
  "attn.to_v": "attn.to_q",
  "attn.to_out.0": "attn.to_out.0",
  "ff.net.0.proj": "ff.net.0.proj",
  "ff.net.2": "ff.net.2",
}
BRANCH_MAP = {k: k for k in LOCAL_MAP}
CONVERT_MAP = {k: "linear" for k in LOCAL_MAP}
QKV_FATIAS = [("attn.to_q", 0, INNER), ("attn.to_k", INNER, 2 * INNER),
              ("attn.to_v", 2 * INNER, 3 * INNER)]


def main():
  t0 = time.time()
  model_sd = torch.load(PTQ / "model.pt", map_location="cpu", weights_only=False)
  scale_sd = torch.load(PTQ / "scale.pt", map_location="cpu", weights_only=False)
  smooth_sd = torch.load(PTQ / "smooth.pt", map_location="cpu", weights_only=False)
  branch_sd = torch.load(PTQ / "branch.pt", map_location="cpu", weights_only=False)

  # smooth em bf16 (mesmo cast do driver do wan) e expansao do branch qkv por fatia
  smooth_sd = {k: v.to(torch.bfloat16) for k, v in smooth_sd.items()}
  branch2 = {}
  for k, v in branch_sd.items():
    if k.endswith("attn.to_q"):
      base = k[: -len("attn.to_q")]
      a, b = v["a.weight"], v["b.weight"]
      for local, ini, fim in QKV_FATIAS:
        branch2[base + local] = {"a.weight": a, "b.weight": b[ini:fim]}
    else:
      branch2[k] = v

  converted = {}
  for blk in range(N_BLOCKS):
    bn = f"transformer_blocks.{blk}"
    pref = bn + "."
    sub_model = {k: v.to("cuda", torch.bfloat16) for k, v in model_sd.items()
                 if k.startswith(pref)}
    sub_scale = {k: v.to("cuda") for k, v in scale_sd.items() if k.startswith(pref)}
    sub_smooth = {k: v.to("cuda") for k, v in smooth_sd.items() if k.startswith(pref)}
    sub_branch = {k: {kk: vv.to("cuda") for kk, vv in v.items()}
                  for k, v in branch2.items() if k.startswith(pref)}
    blk_sd = convert_to_nunchaku_transformer_block_state_dict(
      state_dict=sub_model, scale_dict=sub_scale, smooth_dict=sub_smooth,
      branch_dict=sub_branch, block_name=bn,
      local_name_map=LOCAL_MAP, smooth_name_map=SMOOTH_MAP,
      branch_name_map=BRANCH_MAP, convert_map=CONVERT_MAP)
    update_state_dict(converted, {k: v.cpu() for k, v in blk_sd.items()}, prefix=bn)
    torch.cuda.empty_cache()
    print(f"[l4a] bloco {blk:02d} ok ({time.time()-t0:.0f}s)", flush=True)

  meta = {"quantization_config": json.dumps({"rank": RANK, "precision": "int4"})}
  safetensors.torch.save_file(converted, str(OUT / "transformer_blocks.safetensors"),
                              metadata=meta)
  gb = sum(v.numel() * v.element_size() for v in converted.values()) / 1e9
  print(f"[l4a] L4A-COMPLETO: {len(converted)} tensores, {gb:.2f}GB -> {OUT}", flush=True)


if __name__ == "__main__":
  main()
