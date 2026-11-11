# Speed by quant type

Measured on 2026-11-11, for a later README edit.

**Setup:** RTX 3060 12 GB, Gemma 3 4B, everything on the GPU. Prompt is 512 tokens and generation is 128 tokens, each averaged over 3 runs. llama.cpp is CUDA build 11146 (`llama-bench -ngl 99 -p 512 -n 128 -r 3`), run back to back with Ithaca on the same file.

The IQ files were requantized from Q4_K_M with an importance matrix (`llama-quantize --allow-requantize --imatrix`). That's fine for speed, but their output quality is lower than IQ files made from full-precision weights.

| Quant | File (MB) | Ithaca prompt (tok/s) | llama.cpp prompt | Ithaca gen (tok/s) | llama.cpp gen | Ithaca VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M (reference) | 2374 | 2,904 (84%) | 3,458 | 83.0 (98%) | 84.3 | 2467 |
| IQ4_XS | 2158 | 2,796 (72%) | 3,865 | 89.7 (96%) | 93.3 | 2185 |
| IQ4_NL | 2253 | 2,759 (71%) | 3,870 | 89.7 (98%) | 91.8 | 2281 |
| IQ3_M | 1894 | 2,738 (71%) | 3,831 | 89.8 (102%) | 88.3 | 1927 |
| IQ3_S | 1847 | 2,744 (71%) | 3,857 | 87.9 (102%) | 86.4 | 1874 |
| IQ3_XS | 1776 | 2,814 (70%) | 3,997 | 91.8 (103%) | 89.3 | 1804 |
| IQ3_XXS | 1611 | 2,656 (67%) | 3,965 | 91.2 (93%) | 97.6 | 1843 |
| IQ2_M | 1466 | 2,705 (87%) | 3,117 | 89.6 (97%) | 92.7 | 1698 |
| IQ2_S | 1382 | 2,723 (76%) | 3,592 | 92.9 (94%) | 98.9 | 1614 |
| IQ2_XS | 1339 | 2,860 (83%) | 3,440 | 95.2 (95%) | 100.4 | 1707 |
| IQ2_XXS | 1249 | 2,752 (70%) | 3,949 | 98.0 (96%) | 102.2 | 1617 |
| IQ1_M | 1143 | 2,744 (92%) | 2,971 | 87.2 (87%) | 99.8 | 1512 |
| IQ1_S | 1080 | 2,823 (86%) | 3,301 | 92.1 (82%) | 111.7 | 1448 |

The percentages are Ithaca's speed as a share of llama.cpp's.

## How Ithaca runs IQ quants

- **Generation:** weights stay in their GGUF blocks in VRAM. Each GPU thread decodes 32 values through the shared lookup grids, then multiplies them with int8 activations (`src/iq_kernels.cpp`).
- **Prompts:** each matrix is first decoded into the Q8_0 layout, then multiplied on the int8 tensor cores, the same kernel Q4 and Q8 models use.
  - **Exact types:** IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS decode exactly.
  - **Rounded types:** IQ2_XS, IQ2_S and IQ1_M keep a scale per 16 values, so they are rounded to one scale per 32. This affects prompts only, never generation.

## Known gaps

- **Q2_K and Q3_K layers:** low-bit IQ files often use these for a few layers. They still expand to 8 bits on the GPU, which is most of the gap between file size and VRAM above, together with about 0.1–0.3 GB of working buffers.
- **Prompt speed:** prompts trail llama.cpp by 8–33%. Q4_K_M trails by a similar 16%, which points at the shared int8 tensor-core kernel rather than IQ decoding.
- **IQ1 generation:** IQ1_S and IQ1_M generation is 13–18% behind llama.cpp. The extra group sums they need cost more per value than the other types.
