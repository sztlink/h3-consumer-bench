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

## The W4A4 port, done (2026-08-05)

The paragraph above stopped being a plan. The 50 blocks now run as SVDQuant W4A4 through nunchaku kernels, weights and activations in int4, resident on the GPU.

The pipeline, four scripts on top of the baseline recipe.

```bash
python h3_collect.py               # L2, activation stats from REAL generation
python h3_ptq.py                   # L3, SVDQuant calibration, 50 blocks, 24 min
python h3_convert.py               # L4a, nunchaku two-file checkpoint, 27 s
python h3_denoise_w4a4.py <scene>  # L4b, phase B with int4 blocks resident
```

`h3_collect.py` hooks the 300 target linears during real generation and keeps channelwise absmax plus token reservoirs. `h3_ptq.py` is the calibration inline and self contained, smooth grid by simulated W4A4 error, rank 32 branch, sint4 group 64, median relative error 0.14 across 200 groups. `h3_convert.py` slices the shared qkv branch per projection and emits the standard nunchaku block checkpoint, 10.58GB. `h3_w4a4_loader.py` swaps the six diffusers linears per block for `SVDQW4A4Linear` behind a 2D wrapper, because H3 runs a packed 2D sequence and the nunchaku forward wants 3D.

### Numbers (same 4090, same scenes, same seed)

| | int8 offload baseline | W4A4 |
|---|---|---|
| denoise step, 49 steps | ~14 to 15 s | **8.0 s** |
| one clip, 124 frames with stereo audio | ~950 s | **624 s** |
| VRAM resident | streaming, ~1 block at a time | **13.0 GB** |
| VRAM peak | ~20 GB | 18.6 GB |
| phase B setup | ~83 s | 88 s |

What sits where. The int4 blocks and their rank 32 branches live on the GPU whole. The 13B of AdaLN stay int8 on the host behind a leaf offload hook. The video VAE, which weighs 9.8GB and surprises everyone, streams the same way and only pays at decode. Everything else is bf16 resident. Quality holds by eye across the scene bench, trajectories diverge from the bf16 path as any 49 step rollout does under perturbation, coherence and detail do not drop.

This runs on Ampere and Ada. The two NVFP4 quantizations of H3 on the Hub require a Blackwell GPU and quantize weights only, or mark their W4A4 variants experimental with admitted degradation. As far as the public record shows this is the first working weights-and-activations int4 of H3 on the GPUs people already own.

### The new traps

9. **The repo ships two transformers.** `FL2VA/transformer` is the reference sibling in the original fused layout. The runtime executes the root `transformer/`, already in diffusers layout, different weights, max elementwise gap 2.37. Calibrate the root or you calibrate a ghost. Nothing warns you. Shapes match, calibration error looks healthy, the pipeline completes, the output is pure noise. Only the eye catches it.
10. nunchaku wheels are ABI locked to the torch minor. 1.2.1 wants torch 2.8 and dies with a C++ traceback under 2.11. The 1.3.0.dev wheels cover torch 2.11 and drop into the same venv.
11. Without the group offload hook at the model root, nothing moves your CPU state tensors to the GPU at the transformer boundary. One `forward_pre_hook` restores the old contract.
12. With the blocks resident the video VAE no longer fits beside them. Leaf offload it and the cost lands only on decode.

### The AdaLN collapse, done (same day)

The sigma schedule of both schedulers is deterministic, video shift 12 and audio shift 3, timesteps are `1 - sigmas[:-1]`, and the default grid of 50 points drives 49 evaluations. So the six modulation outputs of every block are precomputable for the whole run. `H3_ADALN=table` builds them straight from the bf16 shards in 10 seconds, 98 sigmas times 3 modalities, 0.95GB on the GPU, exact bf16 math where the streamed path ran int8. `adaln_proj` becomes a nearest-row lookup on the temb table that fails loudly on any unexpected schedule.

| | W4A4, AdaLN streamed | W4A4, AdaLN table |
|---|---|---|
| denoise step | 8.0 s | **3.65 s** |
| one clip | 624 s | **408 s** |
| VRAM resident / peak | 13.0 / 18.6 GB | 14.0 / 19.6 GB |

The denoise loop no longer touches host memory at all, and phase B drops the 13B of AdaLN from RAM entirely. Two build traps. The int8 path leaks dequant buffers if you compute the table through the quantized modules, read the shards directly instead. And the partial pipeline has no schedulers loaded at swap time, construct them from the class with the two shifts from the repo configs.

Checkpoint on the Hub. [felipesztutman/MiniMax-H3-W4A4](https://huggingface.co/felipesztutman/MiniMax-H3-W4A4)

Next, the 3090 in the venue.

## Lineage

[Krea 2 Turbo W4A4 port](https://github.com/nunchaku-ai/nunchaku/pull/947) then [krea-realtime-bench](https://github.com/sztlink/krea-realtime-bench) then this. Code MIT. The MiniMax H3 weights carry their own community license, read it, four territories are excluded.
