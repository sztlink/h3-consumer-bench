#!/usr/bin/env python3
"""H3 W4A4 L3: calibracao SVDQuant dos 50 blocos do MiniMax H3, na linhagem do
ptq_wan.py do krea-realtime-bench (mesma matematica, inline e autocontida).

Diferencas para o H3:
- pesos vem dos SHARDS bf16 do FL2VA sob os nomes ORIGINAIS (blocks.N.attn.qkv_proj
  fundido, mlp.fc1 gate_up 28672, mlp.fc2), consumidos shard a shard pelo index;
  a saida usa os nomes DIFFUSERS (transformer_blocks.N.attn.to_q...) que o runtime ve
- reservatorios do nosso collect_out (512 tokens x in_features por alvo, chave "tudo")
- grid reduzido para a escala 33B (19 pares, eval 2 iters), final 48 iters

Grupos por bloco: qkv (compartilham a entrada, reservatorio do to_q), out, ffn_up,
ffn_down. Saida: model.pt / scale.pt / smooth.pt / branch.pt + ptq_report.json.
"""
import json
import os
import sys
import time
import types
from pathlib import Path

stub = types.ModuleType("deepcompressor.csrc.load")
stub._C = None
sys.modules["deepcompressor.csrc.load"] = stub
sys.path.insert(0, os.path.expanduser("~/realtime-diffusion/src/deepcompressor"))

import torch
from safetensors import safe_open
from deepcompressor.calib.smooth import get_smooth_scale

# ARMADILHA (05/08, run1 = ruido puro): o repo tem DOIS transformers. FL2VA/ e o
# irmao ref2va (apelidado no runtime, nunca executa); o runtime roda o transformer/
# da RAIZ, ja em layout diffusers (to_q separado, max|diff| 2.37 vs a fatia do
# qkv_proj do FL2VA). Calibrar do FL2VA quantiza o modelo errado.
SNAP = os.path.expanduser(
  "~/realtime-diffusion/hf-cache/hub/models--MiniMaxAI--MiniMax-H3/snapshots/"
  "fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/transformer")
IDX_NOME = "diffusion_pytorch_model.safetensors.index.json"
COLLECT = Path(os.path.expanduser("~/realtime-diffusion/h3-w4a4/collect_out"))
OUT = Path(os.path.expanduser("~/realtime-diffusion/h3-w4a4/ptq_out2"))
OUT.mkdir(parents=True, exist_ok=True)
N_BLOCKS, RANK, GROUP, DEV = 50, 32, 64, "cuda"
HID, INNER, FFN = 5376, 7168, 14336

ALPHAS = [i / 10 for i in range(1, 10)]
PAIRS = [(a, 1 - a) for a in ALPHAS] + [(a, 0.0) for a in ALPHAS] + [(0.0, 0.0)]

# (grupo, reservatorio diffusers, [(nome saida diffusers, fonte no shard, fatia)], )
# os shards da raiz JA usam os nomes diffusers: mapeamento direto, sem fatia;
# o grupo qkv segue concatenado na calibracao (mesma entrada, smooth e branch
# compartilhados) e fatiado apenas no salvamento, como antes.
def alvos_do_bloco(b):
  p = f"transformer_blocks.{b}"
  return [
    ("qkv", f"{p}.attn.to_q",
     [(f"{p}.attn.to_q", f"{p}.attn.to_q.weight", None),
      (f"{p}.attn.to_k", f"{p}.attn.to_k.weight", None),
      (f"{p}.attn.to_v", f"{p}.attn.to_v.weight", None)]),
    ("out", f"{p}.attn.to_out.0",
     [(f"{p}.attn.to_out.0", f"{p}.attn.to_out.0.weight", None)]),
    ("ffn_up", f"{p}.ff.net.0.proj",
     [(f"{p}.ff.net.0.proj", f"{p}.ff.net.0.proj.weight", None)]),
    ("ffn_down", f"{p}.ff.net.2",
     [(f"{p}.ff.net.2", f"{p}.ff.net.2.weight", None)]),
  ]


IDX = json.load(open(os.path.join(SNAP, IDX_NOME)))["weight_map"]
_abertos = {}


def tensor_do_shard(nome):
  shard = IDX[nome]
  if shard not in _abertos:
    _abertos[shard] = safe_open(os.path.join(SNAP, shard), framework="pt")
  return _abertos[shard].get_tensor(nome)


def quant_sint4_sim(w, group=GROUP):
  oc, ic = w.shape
  ng = ic // group
  wg = w.view(oc, ng, group)
  scale = wg.abs().amax(dim=-1, keepdim=True).div_(7.0).clamp_min_(1e-8)
  q = wg.div(scale).round_().clamp_(-8, 7)
  return (q * scale).view(oc, ic), scale.view(oc, 1, ng, 1)


def quant_token_sint4_sim(x, group=GROUP):
  n, ic = x.shape
  ng = ic // group
  xg = x.view(n, ng, group)
  scale = xg.abs().amax(dim=-1, keepdim=True).div_(7.0).clamp_min_(1e-8)
  q = xg.div(scale).round_().clamp_(-8, 7)
  return (q * scale).view(n, ic)


def svd_rank(w, rank=RANK):
  U, S, V = torch.svd_lowrank(w, q=rank + 16, niter=4)
  return U[:, :rank] * S[:rank].unsqueeze(0), V[:, :rank].T.contiguous()


def svdquant_branch(w_s, num_iters, tol=0.999):
  b, a = svd_rank(w_s)
  prev = None
  for _ in range(num_iters):
    Rq, _ = quant_sint4_sim(w_s - (b @ a))
    b2, a2 = svd_rank(w_s - Rq)
    err = (w_s - (b2 @ a2) - Rq).pow(2).sum().item()
    if prev is not None and err >= prev * tol:
      break
    b, a, prev = b2, a2, err
  Rq, scale = quant_sint4_sim(w_s - (b @ a))
  return b, a, Rq, scale


def w4a4_sim_out(x, s, b, a, Rq):
  x_s = x / s.unsqueeze(0)
  x_q = quant_token_sint4_sim(x_s)
  return x_q @ Rq.T + (x_s @ a.T) @ b.T


def main():
  t0 = time.time()
  model_sd, scale_sd, smooth_sd, branch_sd = {}, {}, {}, {}
  report = {"config": {"rank": RANK, "group": GROUP, "pairs": len(PAIRS),
                       "eval_iters": 2, "final_iters": 48, "fonte_ativacoes": "int8-loop"},
            "blocks": {}}
  ini = int(os.environ.get("PTQ_INI", "0"))
  for blk in range(ini, N_BLOCKS):
    brep = {}
    for grupo, res_nome, fontes in alvos_do_bloco(blk):
      res = torch.load(COLLECT / "reservoirs" / f"{res_nome}.pt", weights_only=True)
      x = res["tudo"].to(DEV, torch.float32)
      ws, nomes = [], []
      for nome_out, fonte, fatia in fontes:
        w = tensor_do_shard(fonte).to(DEV, torch.float32)
        if fatia is not None:
          w = w[fatia[0]:fatia[1]]
        ws.append(w)
        nomes.append(nome_out)
      w_cat = torch.cat(ws, dim=0)
      ref = x @ w_cat.T
      x_span = x.abs().amax(dim=0)
      w_span = w_cat.abs().amax(dim=0)
      best = (None, float("inf"), None)
      for alpha, beta in PAIRS:
        if alpha == 0.0 and beta == 0.0:
          s = torch.ones_like(x_span)
        else:
          s = get_smooth_scale(alpha=alpha, beta=beta,
                               alpha_base=x_span, beta_base=w_span).clamp_min(1e-5)
        b, a, Rq, _ = svdquant_branch(w_cat * s.unsqueeze(0), num_iters=2)
        err = (w4a4_sim_out(x, s, b, a, Rq) - ref).pow(2).mean().item()
        if err < best[1]:
          best = ((alpha, beta), err, s)
      (alpha, beta), _, s = best
      b, a, Rq, scale = svdquant_branch(w_cat * s.unsqueeze(0), num_iters=48)
      out = w4a4_sim_out(x, s, b, a, Rq)
      rel = ((out - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()
      off = 0
      for nome_out, w_i in zip(nomes, ws):
        oc = w_i.shape[0]
        model_sd[f"{nome_out}.weight"] = Rq[off:off + oc].to(torch.bfloat16).cpu()
        scale_sd[f"{nome_out}.weight.scale.0"] = scale[off:off + oc].to(torch.bfloat16).cpu()
        off += oc
      smooth_sd[res_nome] = s.float().cpu()
      branch_sd[res_nome] = {"a.weight": a.to(torch.bfloat16).cpu(),
                             "b.weight": b.to(torch.bfloat16).cpu()}
      brep[grupo] = {"alpha": alpha, "beta": beta, "rel_out_err": round(rel, 5)}
      del x, ws, w_cat, ref, b, a, Rq
    report["blocks"][blk] = brep
    torch.cuda.empty_cache()
    errs = [v["rel_out_err"] for v in brep.values()]
    print(f"[l3] bloco {blk:02d} rel_err {min(errs):.4f}..{max(errs):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model_sd, OUT / "model.pt")
    torch.save(scale_sd, OUT / "scale.pt")
    torch.save(smooth_sd, OUT / "smooth.pt")
    torch.save(branch_sd, OUT / "branch.pt")
    json.dump(report, open(OUT / "ptq_report.json", "w"), indent=1)
  print(f"[l3] L3-COMPLETO em {(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
  main()
