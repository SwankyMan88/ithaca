#include "quant.h"
#include "gguf.h"
#include <cmath>

static inline float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t rd16(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return v; }

static inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

void dequant_row(int type, const uint8_t* s, float* y, int64_t n) {
    switch (type) {
    case GT_F32: memcpy(y, s, n * 4); return;
    case GT_F16: {
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + i * 2))));
        for (; i < n; i++) y[i] = h2f(rd16(s + i * 2));
        return;
    }
    case GT_BF16: for (int64_t i = 0; i < n; i++) y[i] = bf2f(rd16(s + i * 2)); return;
    case GT_Q4_0:
        for (int64_t b = 0; b < n / 32; b++, s += 18, y += 32) {
            float d = h2f(rd16(s));
            for (int j = 0; j < 16; j++) { y[j] = ((s[2 + j] & 0xF) - 8) * d; y[j + 16] = ((s[2 + j] >> 4) - 8) * d; }
        }
        return;
    case GT_Q4_1:
        for (int64_t b = 0; b < n / 32; b++, s += 20, y += 32) {
            float d = h2f(rd16(s)), m = h2f(rd16(s + 2));
            for (int j = 0; j < 16; j++) { y[j] = (s[4 + j] & 0xF) * d + m; y[j + 16] = (s[4 + j] >> 4) * d + m; }
        }
        return;
    case GT_Q5_0:
        for (int64_t b = 0; b < n / 32; b++, s += 22, y += 32) {
            float d = h2f(rd16(s)); uint32_t qh; memcpy(&qh, s + 2, 4); const uint8_t* qs = s + 6;
            for (int j = 0; j < 16; j++) {
                int x0 = (qs[j] & 0xF) | (((qh >> j) << 4) & 0x10);
                int x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
                y[j] = (x0 - 16) * d; y[j + 16] = (x1 - 16) * d;
            }
        }
        return;
    case GT_Q5_1:
        for (int64_t b = 0; b < n / 32; b++, s += 24, y += 32) {
            float d = h2f(rd16(s)), m = h2f(rd16(s + 2)); uint32_t qh; memcpy(&qh, s + 4, 4); const uint8_t* qs = s + 8;
            for (int j = 0; j < 16; j++) {
                int x0 = (qs[j] & 0xF) | (((qh >> j) << 4) & 0x10);
                int x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
                y[j] = x0 * d + m; y[j + 16] = x1 * d + m;
            }
        }
        return;
    case GT_Q8_0:
        for (int64_t b = 0; b < n / 32; b++, s += 34, y += 32) {
            float d = h2f(rd16(s)); const int8_t* q = (const int8_t*)(s + 2);
            for (int j = 0; j < 32; j++) y[j] = q[j] * d;
        }
        return;
    case GT_Q4_K:
        for (int64_t b = 0; b < n / 256; b++, s += 144) {
            float d = h2f(rd16(s)), dmin = h2f(rd16(s + 2));
            const uint8_t* sc = s + 4; const uint8_t* q = s + 16;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t a, m;
                scale_min_k4(is, sc, a, m); float d1 = d * a, m1 = dmin * m;
                scale_min_k4(is + 1, sc, a, m); float d2 = d * a, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
        }
        return;
    case GT_Q5_K:
        for (int64_t b = 0; b < n / 256; b++, s += 176) {
            float d = h2f(rd16(s)), dmin = h2f(rd16(s + 2));
            const uint8_t* sc = s + 4; const uint8_t* qh = s + 16; const uint8_t* ql = s + 48;
            int is = 0; uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t a, m;
                scale_min_k4(is, sc, a, m); float d1 = d * a, m1 = dmin * m;
                scale_min_k4(is + 1, sc, a, m); float d2 = d * a, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
            }
        }
        return;
    case GT_Q6_K:
        for (int64_t b = 0; b < n / 256; b++, s += 210) {
            const uint8_t* ql = s; const uint8_t* qh = s + 128; const int8_t* sc = (const int8_t*)(s + 192);
            float d = h2f(rd16(s + 208));
            for (int nn = 0; nn < 256; nn += 128) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
        }
        return;
    default:
        for (int64_t i = 0; i < n; i++) y[i] = 0;
    }
}

void quantize_act(const float* x, BQ8* y, int n) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int b = 0; b < n / 32; b++, x += 32) {
        __m256 mx = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        float amax = _mm_cvtss_f32(m4);
        float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        y[b].d = d;
        __m256 vid = _mm256_set1_ps(id);
        for (int j = 0; j < 32; j += 8) {
            __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j), vid));
            alignas(32) int32_t t[8]; _mm256_store_si256((__m256i*)t, q);
            for (int k = 0; k < 8; k++) y[b].qs[j + k] = (int8_t)t[k];
        }
    }
}

void quantize_row_q8_0(const float* x, uint8_t* out, int n) {
    for (int b = 0; b < n / 32; b++, x += 32, out += 34) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = std::fmax(amax, std::fabs(x[j]));
        float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        uint16_t dh = f2h(d); memcpy(out, &dh, 2);
        for (int j = 0; j < 32; j++) out[2 + j] = (uint8_t)(int8_t)std::lrintf(x[j] * id);
    }
}

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline __m256 mul_i8(__m256i bx, __m256i by) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i ax = _mm256_sign_epi8(bx, bx), sy = _mm256_sign_epi8(by, bx);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(ax, sy), ones));
}

float dot_q4_0(const uint8_t* w, const BQ8* y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i off = _mm256_set1_epi8(8);
    for (int i = 0; i < nb; i++, w += 18) {
        float d = h2f(rd16(w)) * y[i].d;
        __m128i t = _mm_loadu_si128((const __m128i*)(w + 2));
        __m256i bx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(t, 4), m4), _mm_and_si128(t, m4));
        bx = _mm256_sub_epi8(bx, off);
        __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), mul_i8(bx, by), acc);
    }
    return hsum(acc);
}

float dot_q8_0(const uint8_t* w, const BQ8* y, int nb) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++, w += 34) {
        float d = h2f(rd16(w)) * y[i].d;
        __m256i bx = _mm256_loadu_si256((const __m256i*)(w + 2));
        __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), mul_i8(bx, by), acc);
    }
    return hsum(acc);
}

float dot_f32(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    float s = hsum(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

float dot_f16(const float* a, const uint16_t* b, int n) {
    __m256 s0 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(b + i))), s0);
    float s = hsum(s0);
    for (; i < n; i++) s += a[i] * h2f(b[i]);
    return s;
}

void axpy_f16(float* y, float a, const uint16_t* x, int n) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x + i))), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * h2f(x[i]);
}

void f32_to_f16(const float* x, uint16_t* y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm_storeu_si128((__m128i*)(y + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), 0));
    for (; i < n; i++) y[i] = f2h(x[i]);
}
