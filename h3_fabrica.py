#!/usr/bin/env python3
"""Fabrica noturna H3 W4A4: processo UNICO que paga o setup uma vez (~90s) e gera
uma fila de cenas em sequencia (408s/clipe no modo table), com mux de audio por
cena. Sem isso cada cena pagava recarga inteira do transformer.

Uso: h3_fabrica.py cena1 cena2 ...          # cenas com embed em h3-lab/embeds
     h3_fabrica.py --todas                  # tudo que tiver embed
Env: H3_ADALN (default table), H3_STEPS (default do modelo), H3_DIR_SAIDA.
Saida por cena: <dir>/<cena>-w4a4.mp4 (+ -com-audio.mp4 quando ha wav).
Log de progresso por cena com marcador FABRICA-CENA-OK <cena> (gatilho p/ hooks).
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/realtime-diffusion/hf-cache"))
os.environ.setdefault("H3_ADALN", "table")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_denoise_w4a4 as base

EMB = os.path.expanduser("~/realtime-diffusion/h3-lab/embeds")
DIR_SAIDA = os.path.expanduser(os.environ.get("H3_DIR_SAIDA",
                                              "~/realtime-diffusion/h3-lab"))


def log(m):
  print(f"[fabrica] {m}", flush=True)


def main():
  cenas = sys.argv[1:]
  if cenas == ["--todas"]:
    cenas = sorted(f[:-3] for f in os.listdir(EMB) if f.endswith(".pt"))
  if not cenas:
    sys.exit("nenhuma cena; use h3_fabrica.py <cenas...> ou --todas")
  faltam = [c for c in cenas if not os.path.exists(f"{EMB}/{c}.pt")]
  if faltam:
    sys.exit(f"sem embed da fase A: {faltam}")
  steps = int(os.environ.get("H3_STEPS", "0")) or None
  log(f"{len(cenas)} cenas na fila; adaln={os.environ['H3_ADALN']}; steps={steps}")

  t0 = time.time()
  pipe, tr = base.montar_pipeline(steps_previstos=steps)
  log(f"setup unico em {time.time()-t0:.0f}s")

  for n, cena in enumerate(cenas, 1):
    t = time.time()
    saida = f"{DIR_SAIDA}/{cena}-w4a4.mp4"
    try:
      base.gerar_cena(pipe, cena, saida, steps=steps)
    except Exception as e:
      log(f"FABRICA-CENA-FALHOU {cena}: {type(e).__name__}: {e}")
      continue
    wav = saida.replace(".mp4", ".wav")
    if os.path.exists(wav):
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", saida, "-i", wav,
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                      saida.replace(".mp4", "-com-audio.mp4")], check=False)
    log(f"FABRICA-CENA-OK {cena} ({n}/{len(cenas)}, {time.time()-t:.0f}s)")
  log(f"FABRICA-COMPLETA: {len(cenas)} cenas em {(time.time()-t0)/60:.0f}min")


if __name__ == "__main__":
  main()
