// Quantized kernels (AVX2/FMA/F16C).
#pragma once
#include <cstdint>
#include <immintrin.h>

struct BQ8 { float d; int8_t qs[32]; };  // activation block used for int8 dot products

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return _cvtss_sh(f, 0); }

void dequant_row(int type, const uint8_t* src, float* dst, int64_t n);
void quantize_act(const float* x, BQ8* y, int n);               // n % 32 == 0
void quantize_row_q8_0(const float* x, uint8_t* out, int n);    // ggml Q8_0 layout (34 B / 32)
float dot_q4_0(const uint8_t* w, const BQ8* y, int nb);
float dot_q8_0(const uint8_t* w, const BQ8* y, int nb);
float dot_f32(const float* a, const float* b, int n);
float dot_f16(const float* a, const uint16_t* b, int n);
void axpy_f16(float* y, float a, const uint16_t* x, int n);    // y += a*x
void f32_to_f16(const float* x, uint16_t* y, int n);
