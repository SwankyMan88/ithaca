// GPU matmuls straight from IQ-quantized weights (IQ1_S .. IQ4_XS). The weights stay in their GGUF blocks in VRAM
// (1.6 to 4.5 bits per weight); each thread decodes one 32-value sub-block into int8 with the shared lookup grids,
// then multiplies it with the int8 activations (dp4a) for every token of its tile, so a weight is read once per tile.
// The decoding follows ggml's dequantize_row_iq* (ggml-quants.c, MIT, the ggml authors).
#include "common.h"
#include "gguf.h"
#include "iq_tables.h"
#include <cstdarg>

// Lookup tables in one device buffer, at these byte offsets.
enum { T_IQ2XXS = 0, T_IQ2XS = 2048, T_IQ2S = 6144, T_IQ3XXS = 14336, T_IQ3S = 15360, T_IQ1S = 17408, T_KSIGNS = 33792, T_KVAL = 33920, T_END = 33936 };

static const int kIqTypes[9] = {GT_IQ2_XXS, GT_IQ2_XS, GT_IQ2_S, GT_IQ3_XXS, GT_IQ3_S, GT_IQ1_S, GT_IQ1_M, GT_IQ4_NL, GT_IQ4_XS};
static const char* kIqNames[9] = {"iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq1s", "iq1m", "iq4nl", "iq4xs"};
int iq_index(int gtype) { for (int i = 0; i < 9; i++) if (kIqTypes[i] == gtype) return i; return -1; }

std::vector<uint8_t> iq_tables_blob() {
    std::vector<uint8_t> b(T_END);
    memcpy(&b[T_IQ2XXS], iq2xxs_grid, sizeof iq2xxs_grid);
    memcpy(&b[T_IQ2XS], iq2xs_grid, sizeof iq2xs_grid);
    memcpy(&b[T_IQ2S], iq2s_grid, sizeof iq2s_grid);
    memcpy(&b[T_IQ3XXS], iq3xxs_grid, sizeof iq3xxs_grid);
    memcpy(&b[T_IQ3S], iq3s_grid, sizeof iq3s_grid);
    memcpy(&b[T_IQ1S], iq1s_grid, sizeof iq1s_grid);
    memcpy(&b[T_KSIGNS], ksigns_iq2xs, sizeof ksigns_iq2xs);
    memcpy(&b[T_KVAL], kvalues_iq4nl, sizeof kvalues_iq4nl);
    return b;
}

namespace {
struct K {
    std::string s;
    void operator()(const char* fmt, ...) {
        char b[600]; va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a);
        s += b; s += "\n";
    }
};

// %w0..7 <- grid word %t20 with the 4 sign bits in %t21 applied (bytes are nonzero and < 128, so ^0xFF +1 negates).
void signed_word(K& k, int w) {
    k(" and.b32 %%t21,%%t21,15; mul.lo.u32 %%t22,%%t21,0x00204081; and.b32 %%t22,%%t22,0x01010101; mul.lo.u32 %%t23,%%t22,255;");
    k(" xor.b32 %%w%d,%%t20,%%t23; add.u32 %%w%d,%%w%d,%%t22;", w, w, w);
}
// %w<w> <- int8 IQ4 values of the four nibbles in the low halves of the bytes of %t20 (kvalues in %k0..%k3).
void nl_word(K& k, int w) {
    k(" and.b32 %%t21,%%t20,0x07070707; shr.u32 %%t22,%%t21,4; or.b32 %%t21,%%t21,%%t22; and.b32 %%t22,%%t21,0xFF; shr.u32 %%t23,%%t21,8; and.b32 %%t23,%%t23,0xFF00; or.b32 %%t22,%%t22,%%t23;");
    k(" prmt.b32 %%t24,%%k0,%%k1,%%t22; prmt.b32 %%t25,%%k2,%%k3,%%t22;");
    k(" shr.u32 %%t23,%%t20,3; and.b32 %%t23,%%t23,0x01010101; mul.lo.u32 %%t23,%%t23,255;");
    k(" not.b32 %%t26,%%t23; and.b32 %%t24,%%t24,%%t26; and.b32 %%t25,%%t25,%%t23; or.b32 %%w%d,%%t24,%%t25;", w);
}
// u16 at [%rb+off] into a 32-bit register (block bases are only 2-byte aligned)
void ld16(K& k, const char* r, int off) { k(" ld.global.nc.u16 %s,[%%rb+%d];", r, off); }
// u32 from two u16 at [%rb+off] (plus the register offset %qo)
void ld32u(K& k, const char* r, int off) { k(" ld.global.nc.u16 %%t30,[%%qa+%d]; ld.global.nc.u16 %%t31,[%%qa+%d]; shl.b32 %%t31,%%t31,16; or.b32 %s,%%t30,%%t31;", off, off + 2, r); }

// Decodes sub-block %ib (0..7) of the block at %rb: %w0..7 (32 int8 values in order), scales %s0 (values 0..15)
// and %s1 (16..31). IQ1: %dg0..3 = delta sign of each 8-group; the values are 8 x grid and the scales d/8.
void decode(K& k, int t) {
    k(" ld.global.nc.b16 %%h0,[%%rb]; cvt.f32.f16 %%fd,%%h0;");
    switch (kIqTypes[t]) {
    case GT_IQ2_XXS:
        k(" shl.b32 %%t1,%%ib,3; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;");
        ld32u(k, "%t2", 2); ld32u(k, "%t3", 6);
        k(" shr.u32 %%t4,%%t3,28; cvt.rn.f32.u32 %%f1,%%t4; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3E800000; mov.f32 %%s1,%%s0;");
        for (int l = 0; l < 4; l++) {
            k(" shr.u32 %%t5,%%t2,%d; and.b32 %%t5,%%t5,255; mul.wide.u32 %%qb,%%t5,8; add.u64 %%qb,%%qb,%%tb; ld.global.nc.v2.u32 {%%t10,%%t11},[%%qb+%d];", 8 * l, T_IQ2XXS);
            k(" shr.u32 %%t6,%%t3,%d; and.b32 %%t6,%%t6,127; cvt.u64.u32 %%qb,%%t6; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u8 %%t7,[%%qb+%d];", 7 * l, T_KSIGNS);
            k(" mov.b32 %%t20,%%t10; mov.b32 %%t21,%%t7;"); signed_word(k, 2 * l);
            k(" mov.b32 %%t20,%%t11; shr.u32 %%t21,%%t7,4;"); signed_word(k, 2 * l + 1);
        }
        break;
    case GT_IQ2_XS:
        k(" cvt.u64.u32 %%qa,%%ib; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t4,[%%qa+66];");
        k(" and.b32 %%t5,%%t4,15; cvt.rn.f32.u32 %%f1,%%t5; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3E800000;");
        k(" shr.u32 %%t5,%%t4,4; cvt.rn.f32.u32 %%f1,%%t5; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s1,%%f1,0f3E800000;");
        k(" shl.b32 %%t1,%%ib,3; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;");
        for (int l = 0; l < 4; l++) {
            k(" ld.global.nc.u16 %%t2,[%%qa+%d]; and.b32 %%t5,%%t2,511; mul.wide.u32 %%qb,%%t5,8; add.u64 %%qb,%%qb,%%tb; ld.global.nc.v2.u32 {%%t10,%%t11},[%%qb+%d];", 2 + 2 * l, T_IQ2XS);
            k(" shr.u32 %%t6,%%t2,9; cvt.u64.u32 %%qb,%%t6; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u8 %%t7,[%%qb+%d];", T_KSIGNS);
            k(" mov.b32 %%t20,%%t10; mov.b32 %%t21,%%t7;"); signed_word(k, 2 * l);
            k(" mov.b32 %%t20,%%t11; shr.u32 %%t21,%%t7,4;"); signed_word(k, 2 * l + 1);
        }
        break;
    case GT_IQ2_S:
        k(" cvt.u64.u32 %%qa,%%ib; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t4,[%%qa+74]; ld.global.nc.u8 %%t3,[%%qa+66];");
        k(" and.b32 %%t5,%%t4,15; cvt.rn.f32.u32 %%f1,%%t5; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3E800000;");
        k(" shr.u32 %%t5,%%t4,4; cvt.rn.f32.u32 %%f1,%%t5; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s1,%%f1,0f3E800000;");
        k(" shl.b32 %%t1,%%ib,2; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;");
        for (int l = 0; l < 4; l++) {
            k(" ld.global.nc.u8 %%t2,[%%qa+%d]; shl.b32 %%t5,%%t3,%d; and.b32 %%t5,%%t5,0x300; or.b32 %%t5,%%t5,%%t2;", 2 + l, 8 - 2 * l);
            k(" mul.wide.u32 %%qb,%%t5,8; add.u64 %%qb,%%qb,%%tb; ld.global.nc.v2.u32 {%%t10,%%t11},[%%qb+%d]; ld.global.nc.u8 %%t7,[%%qa+%d];", T_IQ2S, 34 + l);
            k(" mov.b32 %%t20,%%t10; mov.b32 %%t21,%%t7;"); signed_word(k, 2 * l);
            k(" mov.b32 %%t20,%%t11; shr.u32 %%t21,%%t7,4;"); signed_word(k, 2 * l + 1);
        }
        break;
    case GT_IQ3_XXS:
        k(" shl.b32 %%t1,%%ib,2; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;"); ld32u(k, "%t3", 66);
        k(" shr.u32 %%t4,%%t3,28; cvt.rn.f32.u32 %%f1,%%t4; add.f32 %%f1,%%f1,0f3F000000; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3F000000; mov.f32 %%s1,%%s0;");
        k(" shl.b32 %%t1,%%ib,3; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;");
        for (int l = 0; l < 4; l++) {
            k(" shr.u32 %%t6,%%t3,%d; and.b32 %%t6,%%t6,127; cvt.u64.u32 %%qb,%%t6; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u8 %%t7,[%%qb+%d];", 7 * l, T_KSIGNS);
            k(" ld.global.nc.u8 %%t5,[%%qa+%d]; mul.wide.u32 %%qb,%%t5,4; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u32 %%t10,[%%qb+%d];", 2 + 2 * l, T_IQ3XXS);
            k(" ld.global.nc.u8 %%t5,[%%qa+%d]; mul.wide.u32 %%qb,%%t5,4; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u32 %%t11,[%%qb+%d];", 3 + 2 * l, T_IQ3XXS);
            k(" mov.b32 %%t20,%%t10; mov.b32 %%t21,%%t7;"); signed_word(k, 2 * l);
            k(" mov.b32 %%t20,%%t11; shr.u32 %%t21,%%t7,4;"); signed_word(k, 2 * l + 1);
        }
        break;
    case GT_IQ3_S:
        k(" shr.u32 %%t1,%%ib,1; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t4,[%%qa+106];");
        k(" and.b32 %%t1,%%ib,1; shl.b32 %%t1,%%t1,2; shr.u32 %%t4,%%t4,%%t1; and.b32 %%t4,%%t4,15; shl.b32 %%t4,%%t4,1; add.u32 %%t4,%%t4,1; cvt.rn.f32.u32 %%f1,%%t4; mul.f32 %%s0,%%f1,%%fd; mov.f32 %%s1,%%s0;");
        k(" cvt.u64.u32 %%qa,%%ib; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t3,[%%qa+66];");
        k(" shl.b32 %%t1,%%ib,3; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; shl.b32 %%t1,%%ib,2; cvt.u64.u32 %%qc,%%t1; add.u64 %%qc,%%qc,%%rb;");
        for (int l = 0; l < 4; l++) {
            k(" ld.global.nc.u8 %%t7,[%%qc+%d];", 74 + l);
            k(" ld.global.nc.u8 %%t5,[%%qa+%d]; shl.b32 %%t6,%%t3,%d; and.b32 %%t6,%%t6,256; or.b32 %%t5,%%t5,%%t6; mul.wide.u32 %%qb,%%t5,4; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u32 %%t10,[%%qb+%d];", 2 + 2 * l, 8 - 2 * l, T_IQ3S);
            k(" ld.global.nc.u8 %%t5,[%%qa+%d]; shl.b32 %%t6,%%t3,%d; and.b32 %%t6,%%t6,256; or.b32 %%t5,%%t5,%%t6; mul.wide.u32 %%qb,%%t5,4; add.u64 %%qb,%%qb,%%tb; ld.global.nc.u32 %%t11,[%%qb+%d];", 3 + 2 * l, 7 - 2 * l, T_IQ3S);
            k(" mov.b32 %%t20,%%t10; mov.b32 %%t21,%%t7;"); signed_word(k, 2 * l);
            k(" mov.b32 %%t20,%%t11; shr.u32 %%t21,%%t7,4;"); signed_word(k, 2 * l + 1);
        }
        break;
    case GT_IQ1_S:
        k(" shl.b32 %%t1,%%ib,1; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u16 %%t3,[%%qa+34];");
        k(" shr.u32 %%t4,%%t3,12; and.b32 %%t4,%%t4,7; shl.b32 %%t4,%%t4,1; add.u32 %%t4,%%t4,1; cvt.rn.f32.u32 %%f1,%%t4; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3E000000; mov.f32 %%s1,%%s0;");
        k(" and.b32 %%t4,%%t3,0x8000; setp.ne.u32 %%p5,%%t4,0; selp.s32 %%dg0,-1,1,%%p5; mov.b32 %%dg1,%%dg0; mov.b32 %%dg2,%%dg0; mov.b32 %%dg3,%%dg0;");
        k(" shl.b32 %%t1,%%ib,2; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb;");
        for (int l = 0; l < 4; l++) {
            k(" ld.global.nc.u8 %%t5,[%%qa+%d]; shr.u32 %%t6,%%t3,%d; and.b32 %%t6,%%t6,7; shl.b32 %%t6,%%t6,8; or.b32 %%t5,%%t5,%%t6;", 2 + l, 3 * l);
            k(" mul.wide.u32 %%qb,%%t5,8; add.u64 %%qb,%%qb,%%tb; ld.global.nc.v2.u32 {%%w%d,%%w%d},[%%qb+%d];", 2 * l, 2 * l + 1, T_IQ1S);
        }
        break;
    case GT_IQ1_M: {
        k(" ld.global.nc.u16 %%t10,[%%rb+48]; ld.global.nc.u16 %%t11,[%%rb+50]; ld.global.nc.u16 %%t12,[%%rb+52]; ld.global.nc.u16 %%t13,[%%rb+54];");
        k(" shr.u32 %%t4,%%t10,12; shr.u32 %%t5,%%t11,8; and.b32 %%t5,%%t5,0xf0; or.b32 %%t4,%%t4,%%t5; shr.u32 %%t5,%%t12,4; and.b32 %%t5,%%t5,0xf00; or.b32 %%t4,%%t4,%%t5; and.b32 %%t5,%%t13,0xf000; or.b32 %%t4,%%t4,%%t5;");
        k(" cvt.u16.u32 %%h1,%%t4; cvt.f32.f16 %%fd,%%h1;");
        k(" shr.u32 %%t1,%%ib,1; shl.b32 %%t1,%%t1,1; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u16 %%t3,[%%qa+48];");
        k(" and.b32 %%t1,%%ib,1; mul.lo.u32 %%t1,%%t1,6; shr.u32 %%t3,%%t3,%%t1;");
        k(" and.b32 %%t4,%%t3,7; shl.b32 %%t4,%%t4,1; add.u32 %%t4,%%t4,1; cvt.rn.f32.u32 %%f1,%%t4; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s0,%%f1,0f3E000000;");
        k(" shr.u32 %%t4,%%t3,3; and.b32 %%t4,%%t4,7; shl.b32 %%t4,%%t4,1; add.u32 %%t4,%%t4,1; cvt.rn.f32.u32 %%f1,%%t4; mul.f32 %%f1,%%f1,%%fd; mul.f32 %%s1,%%f1,0f3E000000;");
        k(" shl.b32 %%t1,%%ib,1; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t8,[%%qa+32]; ld.global.nc.u8 %%t9,[%%qa+33];");
        k(" shl.b32 %%t1,%%ib,2; cvt.u64.u32 %%qc,%%t1; add.u64 %%qc,%%qc,%%rb;");
        for (int l = 0; l < 4; l++) {
            const char* qh = l < 2 ? "%t8" : "%t9";
            k(" ld.global.nc.u8 %%t5,[%%qc+%d]; shl.b32 %%t6,%s,%d; and.b32 %%t6,%%t6,0x700; or.b32 %%t5,%%t5,%%t6;", l, qh, l % 2 ? 4 : 8);
            k(" mul.wide.u32 %%qb,%%t5,8; add.u64 %%qb,%%qb,%%tb; ld.global.nc.v2.u32 {%%w%d,%%w%d},[%%qb+%d];", 2 * l, 2 * l + 1, T_IQ1S);
            k(" and.b32 %%t6,%s,%d; setp.ne.u32 %%p5,%%t6,0; selp.s32 %%dg%d,-1,1,%%p5;", qh, l % 2 ? 0x80 : 0x08, l);
        }
        break;
    }
    case GT_IQ4_NL:
        k(" mov.f32 %%s0,%%fd; mov.f32 %%s1,%%fd;");
        for (int i = 0; i < 4; i++) {
            k(" ld.global.nc.u16 %%t30,[%%rb+%d]; ld.global.nc.u16 %%t31,[%%rb+%d]; shl.b32 %%t31,%%t31,16; or.b32 %%t%d,%%t30,%%t31;", 2 + 4 * i, 4 + 4 * i, 10 + i);
        }
        for (int i = 0; i < 4; i++) {
            k(" and.b32 %%t20,%%t%d,0x0F0F0F0F;", 10 + i); nl_word(k, i);
            k(" shr.u32 %%t20,%%t%d,4; and.b32 %%t20,%%t20,0x0F0F0F0F;", 10 + i); nl_word(k, 4 + i);
        }
        break;
    case GT_IQ4_XS:
        k(" ld.global.nc.u16 %%t3,[%%rb+2]; shr.u32 %%t1,%%ib,1; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.u8 %%t4,[%%qa+4];");
        k(" and.b32 %%t1,%%ib,1; shl.b32 %%t1,%%t1,2; shr.u32 %%t4,%%t4,%%t1; and.b32 %%t4,%%t4,15;");
        k(" shl.b32 %%t1,%%ib,1; shr.u32 %%t5,%%t3,%%t1; and.b32 %%t5,%%t5,3; shl.b32 %%t5,%%t5,4; or.b32 %%t4,%%t4,%%t5; sub.s32 %%t4,%%t4,32;");
        k(" cvt.rn.f32.s32 %%f1,%%t4; mul.f32 %%s0,%%f1,%%fd; mov.f32 %%s1,%%s0;");
        k(" shl.b32 %%t1,%%ib,4; cvt.u64.u32 %%qa,%%t1; add.u64 %%qa,%%qa,%%rb; ld.global.nc.v2.u32 {%%t10,%%t11},[%%qa+8]; ld.global.nc.v2.u32 {%%t12,%%t13},[%%qa+16];");
        for (int i = 0; i < 4; i++) {
            k(" and.b32 %%t20,%%t%d,0x0F0F0F0F;", 10 + i); nl_word(k, i);
            k(" shr.u32 %%t20,%%t%d,4; and.b32 %%t20,%%t20,0x0F0F0F0F;", 10 + i); nl_word(k, 4 + i);
        }
        break;
    }
}

// Y[tok][row] = sum over k of W[row][k] * X[tok][k], X given as int8 blocks of 32 (XQ) with scales (XD).
// Each warp owns R rows; each CTA (8 warps) covers TT tokens.
std::string gen_iqmv(int t, int R, int TT) {
    K k;
    const int it = kIqTypes[t];
    const bool sb = it != GT_IQ4_NL, iq1 = it == GT_IQ1_S || it == GT_IQ1_M;
    const int blk = it == GT_IQ2_XXS ? 66 : it == GT_IQ2_XS ? 74 : it == GT_IQ2_S ? 82 : it == GT_IQ3_XXS ? 98 : it == GT_IQ3_S ? 110 :
                    it == GT_IQ1_S ? 50 : it == GT_IQ1_M ? 56 : it == GT_IQ4_NL ? 18 : 136;
    k(".visible .entry iqmv_%s_%d_%d(.param .u64 pW, .param .u64 pT, .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS, .param .u64 pY, .param .u32 pC, .param .u32 pR, .param .u32 pN)",
      kIqNames[t], R, TT);
    k("{");
    k(" .reg .pred %%p<8>; .reg .pred %%pr<%d>; .reg .pred %%pt<%d>; .reg .b32 %%t<40>; .reg .b32 %%w<8>; .reg .b32 %%x<8>; .reg .b32 %%k<4>; .reg .b32 %%dg<4>; .reg .b32 %%gs<4>;", R, TT);
    k(" .reg .b32 %%ib, %%nb, %%row, %%tok0, %%lane, %%C, %%NR, %%NT, %%bi; .reg .b64 %%rb, %%qa, %%qb, %%qc, %%tb, %%wb, %%xq, %%xd, %%yp, %%rowb<%d>, %%rbytes, %%qo; .reg .b16 %%h<2>;", R);
    k(" .reg .f32 %%fd, %%s0, %%s1, %%f<8>, %%xs; .reg .f32 %%a<%d>;", R * TT);
    k(" ld.param.u64 %%wb,[pW]; ld.param.u64 %%tb,[pT]; ld.param.u64 %%xq,[pXQ]; ld.param.u64 %%xd,[pXD]; ld.param.u64 %%yp,[pY];");
    k(" ld.param.u32 %%C,[pC]; ld.param.u32 %%NR,[pR]; ld.param.u32 %%NT,[pN];");
    k(" mov.u32 %%t1,%%tid.x; and.b32 %%lane,%%t1,31; shr.u32 %%t1,%%t1,5; mov.u32 %%t2,%%ctaid.x; mul.lo.u32 %%t2,%%t2,%d; mad.lo.u32 %%row,%%t1,%d,%%t2;", 8 * R, R);
    k(" mov.u32 %%tok0,%%ctaid.y; mul.lo.u32 %%tok0,%%tok0,%d; shr.u32 %%nb,%%C,5;", TT);
    k(" setp.ge.u32 %%p1,%%row,%%NR; @%%p1 bra DONE;");
    // row stride in bytes
    if (sb) k(" shr.u32 %%t3,%%C,8; mul.wide.u32 %%rbytes,%%t3,%d;", blk);
    else k(" mul.wide.u32 %%rbytes,%%nb,%d;", blk);
    for (int j = 0; j < R; j++) {
        k(" add.u32 %%t3,%%row,%d; setp.lt.u32 %%pr%d,%%t3,%%NR; cvt.u64.u32 %%qa,%%t3; mul.lo.u64 %%rowb%d,%%qa,%%rbytes; add.u64 %%rowb%d,%%rowb%d,%%wb;", j, j, j, j, j);
    }
    for (int i = 0; i < TT; i++) k(" add.u32 %%t3,%%tok0,%d; setp.lt.u32 %%pt%d,%%t3,%%NT;", i, i);
    for (int i = 0; i < R * TT; i++) k(" mov.f32 %%a%d,0f00000000;", i);
    if (it == GT_IQ4_NL || it == GT_IQ4_XS) k(" ld.global.nc.v4.u32 {%%k0,%%k1,%%k2,%%k3},[%%tb+%d];", T_KVAL);
    k(" mov.u32 %%bi,%%lane; mov.u32 %%t32,0x01010101;");
    k("LOOP:");
    k(" setp.ge.u32 %%p2,%%bi,%%nb; @%%p2 bra END;");
    if (sb) k(" and.b32 %%ib,%%bi,7; shr.u32 %%t3,%%bi,3; mul.wide.u32 %%qa,%%t3,%d;", blk);
    else k(" mov.u32 %%ib,0; mul.wide.u32 %%qa,%%bi,%d;", blk);
    k(" mov.u64 %%qo,%%qa;");   // block offset within a row
    for (int j = 0; j < R; j++) {
        k(" @!%%pr%d bra SKIPR%d;", j, j);
        k(" add.u64 %%rb,%%rowb%d,%%qo;", j);
        decode(k, t);
        for (int i = 0; i < TT; i++) {
            k(" @!%%pt%d bra SKIPT%d_%d;", i, j, i);
            // activations of token tok0+i, block bi
            k(" add.u32 %%t3,%%tok0,%d; mul.lo.u32 %%t4,%%t3,%%nb; add.u32 %%t4,%%t4,%%bi;", i);
            k(" mul.wide.u32 %%qa,%%t4,32; add.u64 %%qa,%%qa,%%xq; ld.global.nc.v4.u32 {%%x0,%%x1,%%x2,%%x3},[%%qa]; ld.global.nc.v4.u32 {%%x4,%%x5,%%x6,%%x7},[%%qa+16];");
            k(" mul.wide.u32 %%qa,%%t4,4; add.u64 %%qa,%%qa,%%xd; ld.global.nc.f32 %%xs,[%%qa];");
            k(" mov.u32 %%t5,0; mov.u32 %%t6,0;");
            for (int q = 0; q < 4; q++) k(" dp4a.s32.s32 %%t5,%%w%d,%%x%d,%%t5;", q, q);
            for (int q = 4; q < 8; q++) k(" dp4a.s32.s32 %%t6,%%w%d,%%x%d,%%t6;", q, q);
            if (iq1) {   // value = 8 * grid + delta sign, per group of 8
                for (int g = 0; g < 4; g++) k(" mov.u32 %%gs%d,0; dp4a.s32.s32 %%gs%d,%%x%d,%%t32,%%gs%d; dp4a.s32.s32 %%gs%d,%%x%d,%%t32,%%gs%d;", g, g, 2 * g, g, g, 2 * g + 1, g);
                k(" shl.b32 %%t5,%%t5,3; mad.lo.s32 %%t5,%%gs0,%%dg0,%%t5; mad.lo.s32 %%t5,%%gs1,%%dg1,%%t5;");
                k(" shl.b32 %%t6,%%t6,3; mad.lo.s32 %%t6,%%gs2,%%dg2,%%t6; mad.lo.s32 %%t6,%%gs3,%%dg3,%%t6;");
            }
            k(" cvt.rn.f32.s32 %%f2,%%t5; cvt.rn.f32.s32 %%f3,%%t6; mul.f32 %%f2,%%f2,%%s0; fma.rn.f32 %%f2,%%f3,%%s1,%%f2; fma.rn.f32 %%a%d,%%f2,%%xs,%%a%d;", j * TT + i, j * TT + i);
            k("SKIPT%d_%d:", j, i);
        }
        k("SKIPR%d:", j);
    }
    k(" add.u32 %%bi,%%bi,32; bra LOOP;");
    k("END:");
    for (int i = 0; i < R * TT; i++)
        for (int o = 16; o >= 1; o /= 2) k(" shfl.sync.bfly.b32 %%f4,%%a%d,%d,31,0xffffffff; add.f32 %%a%d,%%a%d,%%f4;", i, o, i, i);
    k(" setp.ne.u32 %%p3,%%lane,0; @%%p3 bra DONE;");
    for (int j = 0; j < R; j++)
        for (int i = 0; i < TT; i++) {
            k(" and.pred %%p4,%%pr%d,%%pt%d; add.u32 %%t3,%%tok0,%d; mul.lo.u32 %%t3,%%t3,%%NR; add.u32 %%t3,%%t3,%%row; add.u32 %%t3,%%t3,%d;", j, i, i, j);
            k(" mul.wide.u32 %%qa,%%t3,4; add.u64 %%qa,%%qa,%%yp; @%%p4 st.global.f32 [%%qa],%%a%d;", j * TT + i);
        }
    k("DONE:");
    k(" ret;");
    k("}");
    return k.s;
}
}

// iqq8_<type>: the whole matrix in the Q8_0 GPU layout (f16 scale per 32 values at pS, int8 values at pQ), so prompts
// can use the int8 tensor-core matmul. Exact where a type has one scale per 32 values; IQ2_XS, IQ2_S and IQ1_M (a scale
// per 16) are requantized to one scale per 32, which only rounds the values as much as the activations already are.
std::string gen_iqq8(int t) {
    K k;
    const int it = kIqTypes[t];
    const bool sb = it != GT_IQ4_NL, iq1 = it == GT_IQ1_S || it == GT_IQ1_M, half = it == GT_IQ2_XS || it == GT_IQ2_S || it == GT_IQ1_M;
    const int blk = it == GT_IQ2_XXS ? 66 : it == GT_IQ2_XS ? 74 : it == GT_IQ2_S ? 82 : it == GT_IQ3_XXS ? 98 : it == GT_IQ3_S ? 110 :
                    it == GT_IQ1_S ? 50 : it == GT_IQ1_M ? 56 : it == GT_IQ4_NL ? 18 : 136;
    k(".visible .entry iqq8_%s(.param .u64 pW, .param .u64 pT, .param .u64 pS, .param .u64 pQ, .param .u32 pC, .param .u32 pR)", kIqNames[t]);
    k("{");
    k(" .reg .pred %%p<8>; .reg .b32 %%t<40>; .reg .b32 %%w<8>; .reg .b32 %%k<4>; .reg .b32 %%dg<4>; .reg .b32 %%ib, %%nb, %%bi, %%row, %%C, %%NR, %%gi;");
    k(" .reg .b64 %%rb, %%qa, %%qb, %%qc, %%tb, %%wb, %%sp, %%qp, %%rbytes; .reg .b16 %%h<4>; .reg .f32 %%fd, %%s0, %%s1, %%f<8>, %%amax, %%inv; .reg .f32 %%v<32>;");
    k(" ld.param.u64 %%wb,[pW]; ld.param.u64 %%tb,[pT]; ld.param.u64 %%sp,[pS]; ld.param.u64 %%qp,[pQ]; ld.param.u32 %%C,[pC]; ld.param.u32 %%NR,[pR];");
    k(" shr.u32 %%nb,%%C,5; mov.u32 %%t1,%%ctaid.x; mov.u32 %%t2,%%tid.x; mad.lo.u32 %%gi,%%t1,256,%%t2;");
    k(" div.u32 %%row,%%gi,%%nb; rem.u32 %%bi,%%gi,%%nb; setp.ge.u32 %%p1,%%row,%%NR; @%%p1 bra DONE;");
    if (sb) k(" shr.u32 %%t3,%%C,8; mul.wide.u32 %%rbytes,%%t3,%d; and.b32 %%ib,%%bi,7; shr.u32 %%t3,%%bi,3; mul.wide.u32 %%qc,%%t3,%d;", blk, blk);
    else k(" mul.wide.u32 %%rbytes,%%nb,%d; mov.u32 %%ib,0; mul.wide.u32 %%qc,%%bi,%d;", blk, blk);
    k(" cvt.u64.u32 %%qa,%%row; mul.lo.u64 %%rb,%%qa,%%rbytes; add.u64 %%rb,%%rb,%%wb; add.u64 %%rb,%%rb,%%qc;");
    if (it == GT_IQ4_NL || it == GT_IQ4_XS) k(" ld.global.nc.v4.u32 {%%k0,%%k1,%%k2,%%k3},[%%tb+%d];", T_KVAL);
    decode(k, t);
    if (iq1)   // int8 value 8 * grid + delta sign, per byte: in the biased (+128) domain no byte borrows or carries
        for (int w = 0; w < 8; w++) {
            k(" shl.b32 %%w%d,%%w%d,3; and.b32 %%w%d,%%w%d,0xF8F8F8F8; xor.b32 %%w%d,%%w%d,0x80808080;", w, w, w, w, w, w);
            k(" mul.lo.s32 %%t5,%%dg%d,0x01010101; add.u32 %%w%d,%%w%d,%%t5; xor.b32 %%w%d,%%w%d,0x80808080;", w / 2, w, w, w, w);
        }
    if (!half) {
        k(" cvt.rn.f16.f32 %%h0,%%s0;");
    } else {   // values as floats, then one scale for all 32
        k(" mov.f32 %%amax,0f00000000;");
        for (int e = 0; e < 32; e++) {
            k(" bfe.s32 %%t5,%%w%d,%d,8; cvt.rn.f32.s32 %%v%d,%%t5; mul.f32 %%v%d,%%v%d,%%s%d; abs.f32 %%f1,%%v%d; max.f32 %%amax,%%amax,%%f1;", e / 4, (e % 4) * 8, e, e, e, e < 16 ? 0 : 1, e);
        }
        k(" mul.f32 %%f2,%%amax,0f3C010204; setp.gt.f32 %%p2,%%amax,0f00000000; rcp.approx.f32 %%inv,%%f2; selp.f32 %%inv,%%inv,0f00000000,%%p2; cvt.rn.f16.f32 %%h0,%%f2;");
        for (int w = 0; w < 8; w++) {
            for (int b = 0; b < 4; b++) k(" mul.f32 %%f3,%%v%d,%%inv; cvt.rni.s32.f32 %%t%d,%%f3;", 4 * w + b, 10 + b);
            k(" bfi.b32 %%w%d,%%t11,%%t10,8,8; bfi.b32 %%w%d,%%t12,%%w%d,16,8; bfi.b32 %%w%d,%%t13,%%w%d,24,8;", w, w, w, w, w);
        }
    }
    k(" mul.wide.u32 %%qa,%%gi,2; add.u64 %%qa,%%qa,%%sp; st.global.b16 [%%qa],%%h0;");
    k(" mul.wide.u32 %%qa,%%gi,32; add.u64 %%qa,%%qa,%%qp; st.global.v4.b32 [%%qa],{%%w0,%%w1,%%w2,%%w3}; st.global.v4.b32 [%%qa+16],{%%w4,%%w5,%%w6,%%w7};");
    k("DONE:");
    k(" ret;");
    k("}");
    return k.s;
}

// All IQ matmul kernels: iqmv_<type>_<rows per warp>_<tokens per CTA>.
std::string iq_ptx() {
    std::string s;
    for (int t = 0; t < 9; t++) s += gen_iqmv(t, 4, 1) + gen_iqmv(t, 1, 1) + gen_iqmv(t, 2, 8) + gen_iqq8(t);
    return s;
}
const char* iq_name(int i) { return kIqNames[i]; }
