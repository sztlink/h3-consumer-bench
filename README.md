# h3-consumer-bench

MiniMax H3 generating video with stereo audio on one RTX 4090, three days after the weights opened. H3 is a 33B omni transformer whose published paths want either multi GPU serving or a 128GB unified-memory box like the DGX Spark. The machine here has 62GB of system RAM plus 24GB of VRAM, split across a PCIe bus, and nothing fits by the front door. These are the scripts and the numbers for the door that opens.

Method note. Nothing here forks the model or touches the weights. The recipe is a partition of the stock diffusers modular pipeline (PR 14355 branch) plus int8 weight-only quantization (TorchAo) at load time. What gets measured is what MiniMax shipped.

## The wall

| component | bf16 size |
|---|---|
| text encoder (full Qwen3-VL-32B) | 62.1 GB |
| omni transformer (33B) | 61.7 GB |
| VAEs, refiner, tails | a few GB |

Together they want about 124GB. The host has 62GB of RAM. The GPU has 24GB. The two big components must never be resident at the same time, and neither fits the GPU whole.

## The recipe

**Phase A** builds a partial pipeline with only the encode blocks, loads the text encoder alone in int8, encodes every prompt, writes the full intermediate state to disk and exits. The 32B encoder never meets the transformer.

**Phase B** builds a partial pipeline with only the denoise and decode blocks, loads the transformer alone in int8, streams blocks to the GPU one at a time (`block_level`, one block per group, streams OFF) and consumes the saved state.

```bash
python h3_encode.py            # phase A, embeddings to disk
python h3_denoise.py <scene>   # phase B, video plus stereo wav
```

Scenes live in `h3_cenas.py` in the official H3 prompt grammar (alignment instruction, multimodal description on a timeline, soundscape field).

## Numbers (RTX 4090, 62GB RAM host, 2026-08-06)

| stage | time |
|---|---|
| phase A ready (encoder int8 in RAM, leaf offload) | ~104 s |
| encoding one prompt | 2 to 4 s |
| phase B ready (transformer int8, warm cache) | ~83 s |
| one clip, 124 frames, 832x480, 24 fps, with stereo audio | ~950 s |

Around sixteen minutes per 5.2 second clip, soundtrack generated jointly in the same denoising pass as the pixels. For scale, a community recipe runs H3 on a DGX Spark with online FP8, and one public report there clocks a 15 second clip at eighty minutes. Different box, different codec path, read it as a landmark and not a benchmark duel.

## The traps, so you pay less than we did

1. `use_stream=True` on group offload pins tens of GB of host memory and the kernel kills the process with no traceback. Streams off. Slower transfers, but alive.
2. The modular `output=` selector chooses what RETURNS, not what RUNS. Requesting only `prompt_embeds` still executes the whole graph. Partial pipelines are built from blocks (`SequentialPipelineBlocks.from_blocks_dict` then `init_pipeline`), not filtered by outputs.
3. `load_components(names=[...])` loads a subset. Without it, loading everything for a partial graph drags the other giant in.
4. The FL2VA component list includes `transformer_ref`, the 62GB sibling for reference tasks. Alias it to the same int8 instance (`update_components(transformer=tr, transformer_ref=tr)`). The t2va and i2va paths never execute it, and the alias costs zero bytes.
5. Valid `num_frames` is `17n + 5` between 120 and 360. The practical minimum is 124.
6. Tensors restored from phase A stay on CPU. Pushing them to cuda breaks `build_packed_sequence`, which builds its own tensors on the execution device.
7. Audio comes back as a `(1, 2, N)` cuda tensor. Drop the batch dim before the transpose or soundfile writes a 44 byte header and nothing else.
8. `hf download repo --include "A/*" "B/*"` silently ignores `--include` when extra patterns parse as positional filenames. One pattern per flag.

## What comes next

The live path of this transformer is about 19.3B parameters of block linears across six projections per layer. The FFN is a SwiGLU with the gate and up projections shipped fused, and the attention projections arrive split in the diffusers layout. An earlier revision of this README undercounted the FFN and claimed 15.4B and 7.7GB, this paragraph is the correction. Our SVDQuant lineage, the same pipeline behind the Krea 2 Turbo port and the krea-realtime-bench W4A4 work, puts the corrected target near 9.6GB in int4 with a rank 32 branch. The 13B of per-layer AdaLN are precomputable per timestep. The port begins where this baseline ends.

## Lineage

[Krea 2 Turbo W4A4 port](https://github.com/nunchaku-ai/nunchaku/pull/947) then [krea-realtime-bench](https://github.com/sztlink/krea-realtime-bench) then this. Code MIT. The MiniMax H3 weights carry their own community license, read it, four territories are excluded.
