#!/usr/bin/env python3
"""H3 fase A v2: pipeline PARCIAL (before_encode + text_encoder [+ vae_encoder])
com o encoder int8; salva TODO o estado intermediario para a fase B.

Licao da v1: `output=` escolhe o que devolver, o grafo inteiro roda. O jeito
modular e montar um SequentialPipelineBlocks so com os blocos necessarios.
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h3_cenas import CENAS_H3

OUT = os.path.expanduser("./h3-lab/embeds")
os.makedirs(OUT, exist_ok=True)


def log(m):
  print(f"[h3enc] {m}", flush=True)


from diffusers import ModularPipeline
from diffusers.hooks import apply_group_offloading
from diffusers.modular_pipelines import SequentialPipelineBlocks
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

t0 = time.time()
base = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
quer = ["before_encode", "text_encoder", "vae_encoder"]
sub = {k: v for k, v in base.blocks.sub_blocks.items() if k in quer}
parcial = SequentialPipelineBlocks.from_blocks_dict(sub)
pipe = parcial.init_pipeline("MiniMaxAI/MiniMax-H3")
log(f"pipeline parcial: {list(sub)}")

enc = Qwen3VLForConditionalGeneration.from_pretrained(
  "MiniMaxAI/MiniMax-H3", subfolder="text_encoder", dtype=torch.bfloat16,
  quantization_config=TransformersTorchAoConfig(
    Int8WeightOnlyConfig(version=2),
    modules_to_not_convert=["model.visual", "model.language_model.embed_tokens",
                            "model.language_model.norm", "lm_head"],
  ),
)
log(f"encoder int8 em RAM em {time.time()-t0:.0f}s")
pipe.update_components(text_encoder=enc)
restante = [n for n in pipe.component_names if n != "text_encoder"] if hasattr(pipe, "component_names") else None
try:
  if restante:
    pipe.load_components(names=restante, dtype=torch.bfloat16)
    log(f"componentes restantes carregados: {restante}")
  else:
    pipe.load_components(dtype=torch.bfloat16)
except Exception as e:
  log(f"load restante: {type(e).__name__}: {e}")

offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
               use_stream=True)
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
pipe.text_encoder.requires_grad_(False)
vae = getattr(pipe, "vae", None) or getattr(pipe, "video_vae", None)
if vae is not None:
  vae.to("cuda")
log(f"fase A pronta em {time.time()-t0:.0f}s (componentes: {getattr(pipe, 'component_names', '?')})")

for nome, cena in CENAS_H3.items():
  t = time.time()
  kwargs = dict(prompt=cena["prompt"], num_frames=61, height=480, width=832,
                generator=torch.Generator().manual_seed(2047))
  if cena["image"]:
    from diffusers.utils import load_image
    kwargs["image"] = load_image(cena["image"])
  try:
    estado = pipe(**kwargs)
    if os.environ.get("H3_DEBUG") == "1":
      fonte_dbg = getattr(estado, "intermediates", None) or getattr(estado, "values", None) or {}
      if not fonte_dbg and hasattr(estado, "__dict__"):
        fonte_dbg = {k: v for k, v in estado.__dict__.items() if not k.startswith("_")}
      for k, v in dict(fonte_dbg).items():
        if torch.is_tensor(v):
          log(f"  estado[{k}]: tensor {tuple(v.shape)} {v.dtype}")
        else:
          log(f"  estado[{k}]: {type(v).__name__} = {str(v)[:70]}")
    valores = {}
    fonte = getattr(estado, "intermediates", None) or getattr(estado, "values", None) or {}
    if not fonte and hasattr(estado, "__dict__"):
      fonte = {k: v for k, v in estado.__dict__.items() if not k.startswith("_")}
    for k, v in dict(fonte).items():
      if torch.is_tensor(v):
        valores[k] = v.cpu()
      elif isinstance(v, (int, float, str, list, tuple)):
        valores[k] = v
    torch.save(valores, f"{OUT}/{nome}.pt")
    log(f"{nome}: estado salvo ({sorted(k for k in valores if torch.is_tensor(valores[k]))}) em {time.time()-t:.0f}s")
  except Exception as e:
    import traceback
    traceback.print_exc()
    log(f"{nome} FALHOU: {type(e).__name__}: {e}")

log("ENCODE-COMPLETO")
