# Speed by quant type

Measured on 2026-11-11, for a later README edit.

**Setup:** RTX 3060 12 GB, Gemma 3 4B, everything on the GPU. Prompt is 512 tokens and generation is 128 tokens, each averaged over 3 runs. llama.cpp is CUDA build 11146 (`llama-bench -ngl 99 -p 512 -n 128 -r 3`), run back to back with Ithaca on the same file.

The IQ and Q2_K/Q3_K files were requantized from Q4_K_M with an importance matrix (`llama-quantize --allow-requantize --imatrix`). That's fine for speed, but their output quality is lower than IQ files made from full-precision weights.

| Quant | File (MB) | Ithaca prompt (tok/s) | llama.cpp prompt | Ithaca gen (tok/s) | llama.cpp gen | Ithaca VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M (reference) | 2374 | 2,911 (84%) | 3,480 | 84.6 (96%) | 88.3 | 2467 |
| IQ4_XS | 2158 | 2,852 (72%) | 3,986 | 93.2 (97%) | 95.9 | 2185 |
| IQ4_NL | 2253 | 2,692 (65%) | 4,166 | 92.2 (101%) | 91.0 | 2281 |
| Q3_K_M | 2001 | 2,883 (81%) | 3,546 | 86.8 (114%) | 76.2 | 2065 |
| Q3_K_S | 1847 | 2,917 (80%) | 3,636 | 94.8 (126%) | 75.3 | 1874 |
| IQ3_M | 1894 | 2,802 (83%) | 3,357 | 89.6 (100%) | 89.6 | 1927 |
| IQ3_S | 1847 | 2,752 (68%) | 4,028 | 92.1 (101%) | 90.9 | 1874 |
| IQ3_XS | 1776 | 2,761 (71%) | 3,877 | 96.5 (105%) | 92.3 | 1804 |
| IQ3_XXS | 1611 | 2,816 (77%) | 3,680 | 91.0 (91%) | 99.6 | 1843 |
| Q2_K | 1648 | 2,746 (91%) | 3,014 | 109.6 (123%) | 88.8 | 1676 |
| IQ2_M | 1466 | 2,718 (83%) | 3,271 | 91.5 (95%) | 95.8 | 1698 |
| IQ2_S | 1382 | 2,776 (82%) | 3,395 | 100.1 (104%) | 96.4 | 1614 |
| IQ2_XS | 1339 | 2,725 (93%) | 2,935 | 98.5 (102%) | 96.8 | 1571 |
| IQ2_XXS | 1249 | 2,798 (76%) | 3,682 | 108.7 (108%) | 100.5 | 1481 |
| IQ1_M | 1143 | 2,866 (94%) | 3,054 | 96.3 (93%) | 103.4 | 1376 |
| IQ1_S | 1080 | 2,846 (79%) | 3,598 | 101.2 (86%) | 117.1 | 1312 |

The percentages are Ithaca's speed as a share of llama.cpp's.

## How Ithaca runs IQ quants (and Q2_K, Q3_K)

- **Generation:** weights stay in their GGUF blocks in VRAM. Each GPU thread decodes 32 values through the shared lookup grids, then multiplies them with int8 activations (`src/iq_kernels.cpp`).
- **Prompts:** each matrix is first decoded into the Q8_0 layout, then multiplied on the int8 tensor cores, the same kernel Q4 and Q8 models use.
  - **Exact types:** IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL and IQ4_XS decode exactly.
  - **Rounded types:** IQ2_XS, IQ2_S, IQ1_M, Q2_K and Q3_K keep a scale (and for Q2_K a min) per 16 values, so they are rounded to one scale per 32. This affects prompts only, never generation.

## Known gaps

- **Prompt speed:** prompts trail llama.cpp by 6–35%. Q4_K_M trails by a similar 16%, which points at the shared int8 tensor-core kernel rather than the decoding.
- **IQ1 and IQ3_XXS generation:** IQ1_S, IQ1_M and IQ3_XXS generation is 7–14% behind llama.cpp. IQ1 needs extra group sums per value, which cost more than the other types.
- **VRAM:** VRAM is about 30–250 MB above file size. That's the working buffers plus the embedding table, which some files store in a format Ithaca expands on the GPU.
