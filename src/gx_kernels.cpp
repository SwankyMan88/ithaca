// PTX for the media engines (image / video / audio): fp32 activations, fp16 weights.
// Everything is generated as text here and JIT-compiled by the NVIDIA driver at startup.
#include <cstdio>
#include <string>

std::string gx_ptx();
std::string gx_ptx_tc();
// int8 prompt matmul tiles {warps along tokens, warps along weight rows, 16-token and 8-row mma fragments per warp}:
// [0] 128 x 128 for big matrices, [1] 128 tokens x 64 rows when the big tile would leave SMs idle. gx.cpp sizes grids from these.
extern const int kMmqTiles[2][4] = {{2, 4, 4, 4}, {2, 4, 4, 2}};
static const int kMmqRegs = 0;   // register cap for the int8 matmuls (0 = none)

namespace {
std::string gen_gemm_tc(const char* name, int mode);
std::string gen_lin_fast(const char* name, bool q8, bool half);
struct Out {
    std::string s;
    char b[512];
    template <class... A> void operator()(const char* fmt, A... a) { snprintf(b, sizeof b, fmt, a...); s += b; s += "\n"; }
    void raw(const char* t) { s += t; }
};

// ---------------------------------------------------------------------------------------------
// Tiled GEMM: C[M,N] = alpha * A[M,K] * B + bias + beta * C, batched over grid.z.
//   mode 0 "lin" : A f32 [M][lda],  B f16 [N][ldb]   (activations x weights^T)
//   mode 1 "nt"  : A f32,           B f32 [N][ldb]   (Q * K^T)
//   mode 2 "nn"  : A f32,           B f32 [K][ldb]   (P * V)
//   mode 3 "conv": A f16 [M][K] (conv weight, M = out channels), B = input image gathered on the fly
//                  (implicit im2col, NCHW), N = output pixels. Bias is per row (channel).
// 64x64 output tile per CTA, K step 8, 256 threads, 4x4 outputs per thread.
std::string gen_gemm(const char* name, int mode) {
    Out o;
    const bool af16 = mode == 3, bf16 = mode == 0;
    const int esA = af16 ? 2 : 4, esB = bf16 ? 2 : 4;
    o(".visible .entry %s(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pBias,", name);
    o(" .param .u32 pM, .param .u32 pN, .param .u32 pK, .param .u32 pLda, .param .u32 pLdb, .param .u32 pLdc,");
    o(" .param .u32 pSA, .param .u32 pSB, .param .u32 pSC, .param .f32 pAlpha, .param .f32 pBeta, .param .u32 pFlags,");
    o(" .param .u32 pH, .param .u32 pW, .param .u32 pKH, .param .u32 pKW, .param .u32 pS, .param .u32 pP, .param .u32 pWo, .param .u32 pPW)");
    o("{");
    o(" .reg .pred %%p<8>; .reg .pred %%vA, %%vB0, %%vB1, %%vk, %%pbias, %%pbm, %%pbeta;");
    o(" .reg .b32 %%r<64>; .reg .b64 %%rd<40>; .reg .f32 %%c<16>; .reg .f32 %%a<4>; .reg .f32 %%b<4>; .reg .f32 %%f<12>; .reg .b16 %%h<4>;");
    o(" .shared .align 16 .f32 sA[512]; .shared .align 16 .f32 sB[512];");
    o(" ld.param.u64 %%rd1,[pA]; ld.param.u64 %%rd2,[pB]; ld.param.u64 %%rd3,[pC]; ld.param.u64 %%rd4,[pBias];");
    o(" ld.param.u32 %%r1,[pM]; ld.param.u32 %%r2,[pN]; ld.param.u32 %%r3,[pK]; ld.param.u32 %%r4,[pLda]; ld.param.u32 %%r5,[pLdb]; ld.param.u32 %%r6,[pLdc];");
    o(" ld.param.u32 %%r7,[pSA]; ld.param.u32 %%r8,[pSB]; ld.param.u32 %%r9,[pSC]; ld.param.f32 %%f1,[pAlpha]; ld.param.f32 %%f2,[pBeta]; ld.param.u32 %%r10,[pFlags];");
    o(" ld.param.u32 %%r40,[pH]; ld.param.u32 %%r41,[pW]; ld.param.u32 %%r42,[pKH]; ld.param.u32 %%r43,[pKW]; ld.param.u32 %%r44,[pS]; ld.param.u32 %%r45,[pP]; ld.param.u32 %%r46,[pWo]; ld.param.u32 %%r57,[pPW];");
    o(" mul.lo.u32 %%r47,%%r42,%%r43;");                                                    // KH*KW
    o(" and.b32 %%r11,%%r10,1; setp.ne.u32 %%pbias,%%r11,0; and.b32 %%r11,%%r10,2; setp.ne.u32 %%pbm,%%r11,0; setp.ne.f32 %%pbeta,%%f2,0f00000000;");
    o(" mov.u32 %%r12,%%tid.x; and.b32 %%r13,%%r12,15; shr.u32 %%r14,%%r12,4;");              // tid, tx, ty
    o(" mov.u32 %%r15,%%ctaid.x; shl.b32 %%r15,%%r15,6; mov.u32 %%r16,%%ctaid.y; shl.b32 %%r16,%%r16,6; mov.u32 %%r17,%%ctaid.z;"); // bn, bm, z
    o(" mul.wide.u32 %%rd5,%%r17,%%r7; mul.lo.u64 %%rd5,%%rd5,%d; add.u64 %%rd1,%%rd1,%%rd5;", esA);
    o(" mul.wide.u32 %%rd5,%%r17,%%r8; mul.lo.u64 %%rd5,%%rd5,%d; add.u64 %%rd2,%%rd2,%%rd5;", esB);
    o(" mul.wide.u32 %%rd5,%%r17,%%r9; shl.b64 %%rd5,%%rd5,2; add.u64 %%rd3,%%rd3,%%rd5;");
    // A load mapping: row ar = tid>>2, k offset ak = (tid&3)*2 ; row pointer fixed for the whole loop
    o(" shr.u32 %%r18,%%r12,2; and.b32 %%r19,%%r12,3; shl.b32 %%r19,%%r19,1;");
    o(" add.u32 %%r20,%%r16,%%r18; setp.lt.u32 %%vA,%%r20,%%r1;");                            // m, valid
    o(" mul.wide.u32 %%rd6,%%r20,%%r4; mul.lo.u64 %%rd6,%%rd6,%d; add.u64 %%rd6,%%rd1,%%rd6;", esA); // A row ptr
    o(" mov.u64 %%rd7,sA; mov.u64 %%rd8,sB;");
    o(" mad.lo.u32 %%r21,%%r19,64,%%r18; mul.wide.u32 %%rd9,%%r21,4; add.u64 %%rd9,%%rd7,%%rd9;"); // sA store addr (e=0); e=1 is +256B
    if (mode == 0 || mode == 1) {
        o(" add.u32 %%r22,%%r15,%%r18; setp.lt.u32 %%vB0,%%r22,%%r2;");                      // n, valid
        o(" mul.wide.u32 %%rd10,%%r22,%%r5; mul.lo.u64 %%rd10,%%rd10,%d; add.u64 %%rd10,%%rd2,%%rd10;", esB);
        o(" mul.wide.u32 %%rd11,%%r21,4; add.u64 %%rd11,%%rd8,%%rd11;");                      // sB store (same layout as A)
    } else {
        o(" shr.u32 %%r23,%%r12,5; and.b32 %%r24,%%r12,31; shl.b32 %%r24,%%r24,1;");            // bk2, bc
        o(" add.u32 %%r22,%%r15,%%r24; setp.lt.u32 %%vB0,%%r22,%%r2; add.u32 %%r25,%%r22,1; setp.lt.u32 %%vB1,%%r25,%%r2;");
        o(" mad.lo.u32 %%r26,%%r23,64,%%r24; mul.wide.u32 %%rd11,%%r26,4; add.u64 %%rd11,%%rd8,%%rd11;");
        if (mode == 3) {  // output pixel coordinates of the two columns this thread gathers
            o(" div.u32 %%r27,%%r22,%%r46; mul.lo.u32 %%r28,%%r27,%%r46; sub.u32 %%r28,%%r22,%%r28;");  // oy0, ox0
            o(" div.u32 %%r29,%%r25,%%r46; mul.lo.u32 %%r30,%%r29,%%r46; sub.u32 %%r30,%%r25,%%r30;");  // oy1, ox1
            o(" mul.lo.u32 %%r27,%%r27,%%r44; sub.u32 %%r27,%%r27,%%r45; mul.lo.u32 %%r28,%%r28,%%r44; sub.u32 %%r28,%%r28,%%r57;"); // iy0 base, ix0 base
            o(" mul.lo.u32 %%r29,%%r29,%%r44; sub.u32 %%r29,%%r29,%%r45; mul.lo.u32 %%r30,%%r30,%%r44; sub.u32 %%r30,%%r30,%%r57;");
        }
    }
    o(" shl.b32 %%r31,%%r14,4; cvt.u64.u32 %%rd12,%%r31; add.u64 %%rd12,%%rd7,%%rd12;");     // sA read base (ty*4 floats)
    o(" shl.b32 %%r31,%%r13,4; cvt.u64.u32 %%rd13,%%r31; add.u64 %%rd13,%%rd8,%%rd13;");     // sB read base (tx*4 floats)
    for (int i = 0; i < 16; i++) o(" mov.f32 %%c%d,0f00000000;", i);
    o(" mov.u32 %%r32,0;");
    o("LK:");
    o(" setp.ge.u32 %%p1,%%r32,%%r3; @%%p1 bra EK;");
    // ---- load A (2 elements)
    for (int e = 0; e < 2; e++) {
        o(" add.u32 %%r33,%%r32,%%r19; add.u32 %%r33,%%r33,%d; setp.lt.u32 %%vk,%%r33,%%r3; and.pred %%vk,%%vk,%%vA;", e);
        o(" mov.f32 %%f3,0f00000000; mul.wide.u32 %%rd14,%%r33,%d; add.u64 %%rd14,%%rd6,%%rd14;", esA);
        if (af16) o(" @%%vk ld.global.nc.b16 %%h1,[%%rd14]; @%%vk cvt.f32.f16 %%f3,%%h1;");
        else o(" @%%vk ld.global.f32 %%f3,[%%rd14];");
        o(" st.shared.f32 [%%rd9+%d],%%f3;", e * 256);
    }
    // ---- load B (2 elements)
    if (mode == 0 || mode == 1) {
        for (int e = 0; e < 2; e++) {
            o(" add.u32 %%r33,%%r32,%%r19; add.u32 %%r33,%%r33,%d; setp.lt.u32 %%vk,%%r33,%%r3; and.pred %%vk,%%vk,%%vB0;", e);
            o(" mov.f32 %%f3,0f00000000; mul.wide.u32 %%rd14,%%r33,%d; add.u64 %%rd14,%%rd10,%%rd14;", esB);
            if (bf16) o(" @%%vk ld.global.nc.b16 %%h1,[%%rd14]; @%%vk cvt.f32.f16 %%f3,%%h1;");
            else o(" @%%vk ld.global.f32 %%f3,[%%rd14];");
            o(" st.shared.f32 [%%rd11+%d],%%f3;", e * 256);
        }
    } else if (mode == 2) {
        o(" add.u32 %%r33,%%r32,%%r23; setp.lt.u32 %%vk,%%r33,%%r3;");
        o(" mul.lo.u32 %%r34,%%r33,%%r5; add.u32 %%r34,%%r34,%%r22; mul.wide.u32 %%rd14,%%r34,4; add.u64 %%rd14,%%rd2,%%rd14;");
        o(" mov.f32 %%f3,0f00000000; mov.f32 %%f4,0f00000000;");
        o(" and.pred %%p2,%%vk,%%vB0; @%%p2 ld.global.f32 %%f3,[%%rd14]; and.pred %%p3,%%vk,%%vB1; @%%p3 ld.global.f32 %%f4,[%%rd14+4];");
        o(" st.shared.f32 [%%rd11],%%f3; st.shared.f32 [%%rd11+4],%%f4;");
    } else {
        o(" add.u32 %%r33,%%r32,%%r23; setp.lt.u32 %%vk,%%r33,%%r3;");
        o(" div.u32 %%r34,%%r33,%%r47; mul.lo.u32 %%r35,%%r34,%%r47; sub.u32 %%r35,%%r33,%%r35;");   // ci, r
        o(" div.u32 %%r36,%%r35,%%r43; mul.lo.u32 %%r37,%%r36,%%r43; sub.u32 %%r37,%%r35,%%r37;");   // ky, kx
        o(" mul.lo.u32 %%r34,%%r34,%%r40;");                                                         // ci*H
        for (int e = 0; e < 2; e++) {
            int ry = e ? 29 : 27, rx = e ? 30 : 28;
            o(" add.u32 %%r38,%%r%d,%%r36; add.u32 %%r39,%%r%d,%%r37;", ry, rx);                     // iy, ix (wrap when negative)
            o(" setp.lt.u32 %%p2,%%r38,%%r40; setp.lt.u32 %%p3,%%r39,%%r41; and.pred %%p2,%%p2,%%p3; and.pred %%p2,%%p2,%%vk; and.pred %%p2,%%p2,%%vB%d;", e);
            o(" add.u32 %%r48,%%r34,%%r38; mad.lo.u32 %%r48,%%r48,%%r41,%%r39; mul.wide.u32 %%rd14,%%r48,4; add.u64 %%rd14,%%rd2,%%rd14;");
            o(" mov.f32 %%f3,0f00000000; @%%p2 ld.global.f32 %%f3,[%%rd14]; st.shared.f32 [%%rd11+%d],%%f3;", e * 4);
        }
    }
    o(" bar.sync 0;");
    for (int kk = 0; kk < 8; kk++) {
        o(" ld.shared.v4.f32 {%%a0,%%a1,%%a2,%%a3},[%%rd12+%d]; ld.shared.v4.f32 {%%b0,%%b1,%%b2,%%b3},[%%rd13+%d];", kk * 256, kk * 256);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) o(" fma.rn.f32 %%c%d,%%a%d,%%b%d,%%c%d;", i * 4 + j, i, j, i * 4 + j);
    }
    o(" bar.sync 0;");
    o(" add.u32 %%r32,%%r32,8; bra LK;");
    o("EK:");
    for (int i = 0; i < 4; i++) {
        o(" shl.b32 %%r50,%%r14,2; add.u32 %%r50,%%r50,%d; add.u32 %%r50,%%r50,%%r16; setp.lt.u32 %%p4,%%r50,%%r1;", i);  // m
        for (int j = 0; j < 4; j++) {
            o(" shl.b32 %%r51,%%r13,2; add.u32 %%r51,%%r51,%d; add.u32 %%r51,%%r51,%%r15; setp.lt.u32 %%p5,%%r51,%%r2; and.pred %%p5,%%p5,%%p4;", j);
            o(" @!%%p5 bra S%d_%d;", i, j);
            o(" mul.f32 %%f5,%%c%d,%%f1;", i * 4 + j);
            o(" @%%pbias selp.u32 %%r52,%%r50,%%r51,%%pbm; @%%pbias mul.wide.u32 %%rd15,%%r52,4; @%%pbias add.u64 %%rd15,%%rd4,%%rd15; @%%pbias ld.global.f32 %%f6,[%%rd15]; @%%pbias add.f32 %%f5,%%f5,%%f6;");
            o(" mad.lo.u32 %%r53,%%r50,%%r6,%%r51; mul.wide.u32 %%rd16,%%r53,4; add.u64 %%rd16,%%rd3,%%rd16;");
            o(" @%%pbeta ld.global.f32 %%f7,[%%rd16]; @%%pbeta fma.rn.f32 %%f5,%%f7,%%f2,%%f5;");
            o(" st.global.f32 [%%rd16],%%f5;");
            o("S%d_%d:", i, j);
        }
    }
    o(" ret;\n}");
    return o.s;
}

// ---------------------------------------------------------------------------------------------
// Tensor-core version of gen_gemm (same parameters and modes), using TF32 mma.sync (sm_80+).
// 64x64 CTA tile, K step 16, 128 threads = 4 warps, each warp a 32x32 sub-tile (2x4 m16n8k8 tiles).
// Shared tiles are [row][k] with a padded stride of 20 floats to avoid bank conflicts.
std::string gen_gemm_tc(const char* name, int mode) {
    Out o;
    const bool af16 = mode == 3, bf16 = mode == 0;
    const int esA = af16 ? 2 : 4, esB = bf16 ? 2 : 4;
    o(".visible .entry %s(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pBias,", name);
    o(" .param .u32 pM, .param .u32 pN, .param .u32 pK, .param .u32 pLda, .param .u32 pLdb, .param .u32 pLdc,");
    o(" .param .u32 pSA, .param .u32 pSB, .param .u32 pSC, .param .f32 pAlpha, .param .f32 pBeta, .param .u32 pFlags,");
    o(" .param .u32 pH, .param .u32 pW, .param .u32 pKH, .param .u32 pKW, .param .u32 pS, .param .u32 pP, .param .u32 pWo, .param .u32 pPW)");
    o("{");
    o(" .reg .pred %%p<8>; .reg .pred %%pbias, %%pbm, %%pbeta, %%vn;");
    o(" .reg .b32 %%r<80>; .reg .b64 %%rd<40>; .reg .f32 %%c<32>; .reg .f32 %%f<12>; .reg .b32 %%a<8>; .reg .b32 %%b<8>; .reg .b16 %%h<4>;");
    o(" .shared .align 16 .f32 tA[1280]; .shared .align 16 .f32 tB[1280];");
    o(" ld.param.u64 %%rd1,[pA]; ld.param.u64 %%rd2,[pB]; ld.param.u64 %%rd3,[pC]; ld.param.u64 %%rd4,[pBias];");
    o(" ld.param.u32 %%r1,[pM]; ld.param.u32 %%r2,[pN]; ld.param.u32 %%r3,[pK]; ld.param.u32 %%r4,[pLda]; ld.param.u32 %%r5,[pLdb]; ld.param.u32 %%r6,[pLdc];");
    o(" ld.param.u32 %%r7,[pSA]; ld.param.u32 %%r8,[pSB]; ld.param.u32 %%r9,[pSC]; ld.param.f32 %%f1,[pAlpha]; ld.param.f32 %%f2,[pBeta]; ld.param.u32 %%r10,[pFlags];");
    o(" ld.param.u32 %%r40,[pH]; ld.param.u32 %%r41,[pW]; ld.param.u32 %%r42,[pKH]; ld.param.u32 %%r43,[pKW]; ld.param.u32 %%r44,[pS]; ld.param.u32 %%r45,[pP]; ld.param.u32 %%r46,[pWo]; ld.param.u32 %%r57,[pPW];");
    o(" mul.lo.u32 %%r47,%%r42,%%r43;");
    o(" and.b32 %%r11,%%r10,1; setp.ne.u32 %%pbias,%%r11,0; and.b32 %%r11,%%r10,2; setp.ne.u32 %%pbm,%%r11,0; setp.ne.f32 %%pbeta,%%f2,0f00000000;");
    o(" mov.u32 %%r12,%%tid.x; and.b32 %%r13,%%r12,31; shr.u32 %%r14,%%r12,5;");                 // tid, lane, warp
    o(" shr.u32 %%r60,%%r13,2; and.b32 %%r61,%%r13,3;");                                          // g, t4
    o(" shr.u32 %%r62,%%r14,1; shl.b32 %%r62,%%r62,5; and.b32 %%r63,%%r14,1; shl.b32 %%r63,%%r63,5;"); // wm, wn
    o(" mov.u32 %%r15,%%ctaid.x; shl.b32 %%r15,%%r15,6; mov.u32 %%r16,%%ctaid.y; shl.b32 %%r16,%%r16,6; mov.u32 %%r17,%%ctaid.z;");
    o(" mul.wide.u32 %%rd5,%%r17,%%r7; mul.lo.u64 %%rd5,%%rd5,%d; add.u64 %%rd1,%%rd1,%%rd5;", esA);
    o(" mul.wide.u32 %%rd5,%%r17,%%r8; mul.lo.u64 %%rd5,%%rd5,%d; add.u64 %%rd2,%%rd2,%%rd5;", esB);
    o(" mul.wide.u32 %%rd5,%%r17,%%r9; shl.b64 %%rd5,%%rd5,2; add.u64 %%rd3,%%rd3,%%rd5;");
    o(" mov.u64 %%rd7,tA; mov.u64 %%rd8,tB;");
    // row-major loader mapping (A, and B in lin/nt): kk = tid&15 fixed, row = (tid>>4) + 8e
    o(" and.b32 %%r18,%%r12,15; shr.u32 %%r19,%%r12,4;");
    o(" mad.lo.u32 %%r20,%%r19,20,%%r18; mul.wide.u32 %%rd9,%%r20,4; add.u64 %%rd9,%%rd7,%%rd9;");  // tA store base
    if (mode == 0 || mode == 1) {
        o(" mul.wide.u32 %%rd11,%%r20,4; add.u64 %%rd11,%%rd8,%%rd11;");                          // tB store base (same mapping)
    } else {
        // column loader mapping (nn / conv): n = tid&63 fixed, k = (tid>>6) + 2e
        o(" and.b32 %%r21,%%r12,63; shr.u32 %%r22,%%r12,6;");
        o(" mad.lo.u32 %%r23,%%r21,20,%%r22; mul.wide.u32 %%rd11,%%r23,4; add.u64 %%rd11,%%rd8,%%rd11;");
        o(" add.u32 %%r24,%%r15,%%r21; setp.lt.u32 %%vn,%%r24,%%r2;");                              // n, valid
        if (mode == 3) {
            o(" div.u32 %%r27,%%r24,%%r46; mul.lo.u32 %%r28,%%r27,%%r46; sub.u32 %%r28,%%r24,%%r28;");
            o(" mul.lo.u32 %%r27,%%r27,%%r44; sub.u32 %%r27,%%r27,%%r45; mul.lo.u32 %%r28,%%r28,%%r44; sub.u32 %%r28,%%r28,%%r57;"); // iy0, ix0
        }
    }
    // fragment read bases
    o(" add.u32 %%r25,%%r62,%%r60; mad.lo.u32 %%r25,%%r25,20,%%r61; mul.wide.u32 %%rd12,%%r25,4; add.u64 %%rd12,%%rd7,%%rd12;");
    o(" add.u32 %%r26,%%r63,%%r60; mad.lo.u32 %%r26,%%r26,20,%%r61; mul.wide.u32 %%rd13,%%r26,4; add.u64 %%rd13,%%rd8,%%rd13;");
    for (int i = 0; i < 32; i++) o(" mov.f32 %%c%d,0f00000000;", i);
    o(" mov.u32 %%r32,0;");
    o("LK:");
    o(" setp.ge.u32 %%p1,%%r32,%%r3; @%%p1 bra EK;");
    o(" add.u32 %%r33,%%r32,%%r18; setp.lt.u32 %%p6,%%r33,%%r3;");   // k for the row-major loaders, valid
    for (int e = 0; e < 8; e++) {   // A
        o(" add.u32 %%r34,%%r19,%d; add.u32 %%r34,%%r34,%%r16; setp.lt.u32 %%p2,%%r34,%%r1; and.pred %%p2,%%p2,%%p6;", e * 8);
        o(" mad.lo.u32 %%r35,%%r34,%%r4,%%r33; mul.wide.u32 %%rd14,%%r35,%d; add.u64 %%rd14,%%rd1,%%rd14; mov.f32 %%f3,0f00000000;", esA);
        if (af16) o(" @%%p2 ld.global.nc.b16 %%h1,[%%rd14]; @%%p2 cvt.f32.f16 %%f3,%%h1;");
        else o(" @%%p2 ld.global.f32 %%f3,[%%rd14];");
        o(" st.shared.f32 [%%rd9+%d],%%f3;", e * 640);
    }
    if (mode == 0 || mode == 1) {
        for (int e = 0; e < 8; e++) {
            o(" add.u32 %%r34,%%r19,%d; add.u32 %%r34,%%r34,%%r15; setp.lt.u32 %%p2,%%r34,%%r2; and.pred %%p2,%%p2,%%p6;", e * 8);
            o(" mad.lo.u32 %%r35,%%r34,%%r5,%%r33; mul.wide.u32 %%rd14,%%r35,%d; add.u64 %%rd14,%%rd2,%%rd14; mov.f32 %%f3,0f00000000;", esB);
            if (bf16) o(" @%%p2 ld.global.nc.b16 %%h1,[%%rd14]; @%%p2 cvt.f32.f16 %%f3,%%h1;");
            else o(" @%%p2 ld.global.f32 %%f3,[%%rd14];");
            o(" st.shared.f32 [%%rd11+%d],%%f3;", e * 640);
        }
    } else {
        for (int e = 0; e < 8; e++) {
            o(" add.u32 %%r36,%%r32,%%r22; add.u32 %%r36,%%r36,%d; setp.lt.u32 %%p3,%%r36,%%r3; and.pred %%p3,%%p3,%%vn; mov.f32 %%f3,0f00000000;", e * 2);
            if (mode == 2) {
                o(" mad.lo.u32 %%r37,%%r36,%%r5,%%r24; mul.wide.u32 %%rd14,%%r37,4; add.u64 %%rd14,%%rd2,%%rd14; @%%p3 ld.global.f32 %%f3,[%%rd14];");
            } else {
                o(" div.u32 %%r37,%%r36,%%r47; mul.lo.u32 %%r38,%%r37,%%r47; sub.u32 %%r38,%%r36,%%r38;");   // ci, r
                o(" div.u32 %%r39,%%r38,%%r43; mul.lo.u32 %%r48,%%r39,%%r43; sub.u32 %%r48,%%r38,%%r48;");   // ky, kx
                o(" add.u32 %%r49,%%r27,%%r39; add.u32 %%r50,%%r28,%%r48;");
                o(" setp.lt.u32 %%p4,%%r49,%%r40; setp.lt.u32 %%p5,%%r50,%%r41; and.pred %%p3,%%p3,%%p4; and.pred %%p3,%%p3,%%p5;");
                o(" mad.lo.u32 %%r51,%%r37,%%r40,%%r49; mad.lo.u32 %%r51,%%r51,%%r41,%%r50; mul.wide.u32 %%rd14,%%r51,4; add.u64 %%rd14,%%rd2,%%rd14; @%%p3 ld.global.f32 %%f3,[%%rd14];");
            }
            o(" st.shared.f32 [%%rd11+%d],%%f3;", e * 8);
        }
    }
    o(" bar.sync 0;");
    for (int ks = 0; ks < 16; ks += 8) {
        for (int mt = 0; mt < 2; mt++) {
            int base = mt * 1280 + ks * 4;
            o(" ld.shared.f32 %%f4,[%%rd12+%d]; ld.shared.f32 %%f5,[%%rd12+%d]; ld.shared.f32 %%f6,[%%rd12+%d]; ld.shared.f32 %%f7,[%%rd12+%d];",
              base, base + 640, base + 16, base + 656);
            o(" cvt.rna.tf32.f32 %%a%d,%%f4; cvt.rna.tf32.f32 %%a%d,%%f5; cvt.rna.tf32.f32 %%a%d,%%f6; cvt.rna.tf32.f32 %%a%d,%%f7;",
              mt * 4, mt * 4 + 1, mt * 4 + 2, mt * 4 + 3);
        }
        for (int nt = 0; nt < 4; nt++) {
            int base = nt * 640 + ks * 4;
            o(" ld.shared.f32 %%f8,[%%rd13+%d]; ld.shared.f32 %%f9,[%%rd13+%d]; cvt.rna.tf32.f32 %%b%d,%%f8; cvt.rna.tf32.f32 %%b%d,%%f9;",
              base, base + 16, nt * 2, nt * 2 + 1);
        }
        for (int mt = 0; mt < 2; mt++)
            for (int nt = 0; nt < 4; nt++) {
                int c = (mt * 4 + nt) * 4;
                o(" mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%%c%d,%%c%d,%%c%d,%%c%d},{%%a%d,%%a%d,%%a%d,%%a%d},{%%b%d,%%b%d},{%%c%d,%%c%d,%%c%d,%%c%d};",
                  c, c + 1, c + 2, c + 3, mt * 4, mt * 4 + 1, mt * 4 + 2, mt * 4 + 3, nt * 2, nt * 2 + 1, c, c + 1, c + 2, c + 3);
            }
    }
    o(" bar.sync 0;");
    o(" add.u32 %%r32,%%r32,16; bra LK;");
    o("EK:");
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 4; nt++)
            for (int h = 0; h < 2; h++)
                for (int j = 0; j < 2; j++) {
                    int ci = (mt * 4 + nt) * 4 + h * 2 + j;
                    o(" add.u32 %%r52,%%r16,%%r62; add.u32 %%r52,%%r52,%%r60; add.u32 %%r52,%%r52,%d;", mt * 16 + h * 8);
                    o(" add.u32 %%r53,%%r15,%%r63; shl.b32 %%r54,%%r61,1; add.u32 %%r53,%%r53,%%r54; add.u32 %%r53,%%r53,%d;", nt * 8 + j);
                    o(" setp.lt.u32 %%p4,%%r52,%%r1; setp.lt.u32 %%p5,%%r53,%%r2; and.pred %%p5,%%p5,%%p4; @!%%p5 bra S%d;", ci);
                    o(" mul.f32 %%f5,%%c%d,%%f1;", ci);
                    o(" @%%pbias selp.u32 %%r55,%%r52,%%r53,%%pbm; @%%pbias mul.wide.u32 %%rd15,%%r55,4; @%%pbias add.u64 %%rd15,%%rd4,%%rd15; @%%pbias ld.global.f32 %%f6,[%%rd15]; @%%pbias add.f32 %%f5,%%f5,%%f6;");
                    o(" mad.lo.u32 %%r56,%%r52,%%r6,%%r53; mul.wide.u32 %%rd16,%%r56,4; add.u64 %%rd16,%%rd3,%%rd16;");
                    o(" @%%pbeta ld.global.f32 %%f7,[%%rd16]; @%%pbeta fma.rn.f32 %%f5,%%f7,%%f2,%%f5; st.global.f32 [%%rd16],%%f5;");
                    o("S%d:", ci);
                }
    o(" ret;\n}");
    return o.s;
}

// Fast tensor-core linear for K % 32 == 0: y[M,N] = x[M,K] W^T (+bias, +beta*y), W fp16 [N][K].
// 64x64x32 tiles, 128-bit loads, next tile prefetched into registers while the current one computes.
std::string gen_lin_fast(const char* name, bool q8 = false, bool half = false) {
    Out o;
    const bool quant = q8;
    o(".visible .entry %s(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pBias,", name);
    o(" .param .u32 pM, .param .u32 pN, .param .u32 pK, .param .u32 pLda, .param .u32 pLdb, .param .u32 pLdc,");
    o(" .param .u32 pSA, .param .u32 pSB, .param .u32 pSC, .param .f32 pAlpha, .param .f32 pBeta, .param .u32 pFlags,");
    o(" .param .u32 pH, .param .u32 pW, .param .u32 pKH, .param .u32 pKW, .param .u32 pS, .param .u32 pP, .param .u32 pWo, .param .u32 pPW%s)", quant ? ", .param .u64 pScale" : "");
    o("{");
    o(" .reg .pred %%p<8>; .reg .pred %%pbias, %%pbm, %%pbeta, %%va<4>, %%vb<2>;");
    o(" .reg .b32 %%r<80>; .reg .b64 %%rd<40>; .reg .f32 %%c<32>; .reg .f32 %%f<12>; .reg .b32 %%a<8>; .reg .b32 %%b<8>;");
    o(" .reg .f32 %%x<16>; .reg .b32 %%y<8>; .reg .f32 %%z<16>; .reg .b16 %%h<16>;");
    o(" .shared .align 16 .f32 fA[2304]; .shared .align 16 .f32 fB[2304];");
    o(" ld.param.u64 %%rd1,[pA]; ld.param.u64 %%rd2,[pB]; ld.param.u64 %%rd3,[pC]; ld.param.u64 %%rd4,[pBias];");
    o(" ld.param.u32 %%r1,[pM]; ld.param.u32 %%r2,[pN]; ld.param.u32 %%r3,[pK]; ld.param.u32 %%r4,[pLda]; ld.param.u32 %%r5,[pLdb]; ld.param.u32 %%r6,[pLdc];");
    o(" ld.param.f32 %%f1,[pAlpha]; ld.param.f32 %%f2,[pBeta]; ld.param.u32 %%r10,[pFlags];");
    if (quant) o(" ld.param.u64 %%rd18,[pScale];");
    o(" and.b32 %%r11,%%r10,1; setp.ne.u32 %%pbias,%%r11,0; and.b32 %%r11,%%r10,2; setp.ne.u32 %%pbm,%%r11,0; setp.ne.f32 %%pbeta,%%f2,0f00000000;");
    o(" mov.u32 %%r12,%%tid.x; and.b32 %%r13,%%r12,31; shr.u32 %%r14,%%r12,5; shr.u32 %%r60,%%r13,2; and.b32 %%r61,%%r13,3;");
    o(" shr.u32 %%r62,%%r14,1; shl.b32 %%r62,%%r62,5; and.b32 %%r63,%%r14,1; shl.b32 %%r63,%%r63,5;");
    o(" mov.u32 %%r15,%%ctaid.x; shl.b32 %%r15,%%r15,6; mov.u32 %%r16,%%ctaid.y; shl.b32 %%r16,%%r16,6;");
    o(" mov.u64 %%rd7,fA; mov.u64 %%rd8,fB;");
    // A slots: row = (tid>>3) + 16e, k = (tid&7)*4 ; B slots: n = (tid>>2) + 32e, k = (tid&3)*8
    o(" shr.u32 %%r18,%%r12,3; and.b32 %%r19,%%r12,7; shl.b32 %%r19,%%r19,2;");
    o(" shr.u32 %%r20,%%r12,2; and.b32 %%r21,%%r12,3; shl.b32 %%r21,%%r21,3;");
    for (int e = 0; e < 4; e++) {   // A row pointers (k offset added per tile)
        o(" add.u32 %%r22,%%r18,%d; add.u32 %%r22,%%r22,%%r16; setp.lt.u32 %%va%d,%%r22,%%r1; min.u32 %%r22,%%r22,%%r1;", e * 16, e);
        o(" mad.lo.u32 %%r23,%%r22,%%r4,%%r19; mul.wide.u32 %%rd%d,%%r23,4; add.u64 %%rd%d,%%rd1,%%rd%d;", 20 + e, 20 + e, 20 + e);
    }
    for (int e = 0; e < 2; e++) {
        o(" add.u32 %%r22,%%r20,%d; add.u32 %%r22,%%r22,%%r15; setp.lt.u32 %%vb%d,%%r22,%%r2; min.u32 %%r22,%%r22,%%r2;", e * 32, e);
        o(" mad.lo.u32 %%r23,%%r22,%%r5,%%r21; mul.wide.u32 %%rd%d,%%r23,%d; add.u64 %%rd%d,%%rd2,%%rd%d;", 24 + e, q8 ? 1 : 2, 24 + e, 24 + e);
    }
    // shared store addresses
    o(" mad.lo.u32 %%r24,%%r18,36,%%r19; mul.wide.u32 %%rd9,%%r24,4; add.u64 %%rd9,%%rd7,%%rd9;");
    o(" mad.lo.u32 %%r25,%%r20,36,%%r21; mul.wide.u32 %%rd11,%%r25,4; add.u64 %%rd11,%%rd8,%%rd11;");
    // fragment read bases
    o(" add.u32 %%r26,%%r62,%%r60; mad.lo.u32 %%r26,%%r26,36,%%r61; mul.wide.u32 %%rd12,%%r26,4; add.u64 %%rd12,%%rd7,%%rd12;");
    o(" add.u32 %%r27,%%r63,%%r60; mad.lo.u32 %%r27,%%r27,36,%%r61; mul.wide.u32 %%rd13,%%r27,4; add.u64 %%rd13,%%rd8,%%rd13;");
    for (int i = 0; i < 32; i++) o(" mov.f32 %%c%d,0f00000000;", i);
    auto gload = [&](const char* koff_reg) {   // global -> registers for the tile at byte offsets (k*4 for A, k*2 for B)
        for (int e = 0; e < 4; e++) {
            o(" mul.wide.u32 %%rd30,%s,4; add.u64 %%rd30,%%rd%d,%%rd30;", koff_reg, 20 + e);
            o(" mov.f32 %%x%d,0f00000000; mov.f32 %%x%d,0f00000000; mov.f32 %%x%d,0f00000000; mov.f32 %%x%d,0f00000000;", e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
            o(" @%%va%d ld.global.v4.f32 {%%x%d,%%x%d,%%x%d,%%x%d},[%%rd30];", e, e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
        }
        for (int e = 0; e < 2; e++) {
            o(" mul.wide.u32 %%rd31,%s,%d; add.u64 %%rd31,%%rd%d,%%rd31;", koff_reg, q8 ? 1 : 2, 24 + e);
            o(" mov.b32 %%y%d,0; mov.b32 %%y%d,0; mov.b32 %%y%d,0; mov.b32 %%y%d,0;", e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
            if (quant) o(" @%%vb%d ld.global.nc.v2.b32 {%%y%d,%%y%d},[%%rd31];", e, e * 4, e * 4 + 1);   // 8 int8 weights
            else o(" @%%vb%d ld.global.nc.v4.b32 {%%y%d,%%y%d,%%y%d,%%y%d},[%%rd31];", e, e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
        }
    };
    auto sstore = [&](const char* koff) {
        for (int e = 0; e < 4; e++) o(" st.shared.v4.f32 [%%rd9+%d],{%%x%d,%%x%d,%%x%d,%%x%d};", e * 16 * 36 * 4, e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
        for (int e = 0; e < 2; e++) {
            if (q8) {
                // scale for (row n, group k/32): S[n*(K/32) + koff/32]
                o(" add.u32 %%r71,%%r20,%d; add.u32 %%r71,%%r71,%%r15; min.u32 %%r71,%%r71,%%r2; shr.u32 %%r72,%%r3,5; shr.u32 %%r73,%s,5;", e * 32, koff);
                o(" mad.lo.u32 %%r71,%%r71,%%r72,%%r73; mul.wide.u32 %%rd19,%%r71,4; add.u64 %%rd19,%%rd18,%%rd19; mov.f32 %%f10,0f00000000; @%%vb%d ld.global.nc.f32 %%f10,[%%rd19];", e);
                for (int q = 0; q < 8; q++)
                    o(" bfe.s32 %%r70,%%y%d,%d,8; cvt.rn.f32.s32 %%z%d,%%r70; mul.f32 %%z%d,%%z%d,%%f10;", e * 4 + q / 4, (q % 4) * 8, q, q, q);
            } else
                for (int q = 0; q < 4; q++) {
                    o(" mov.b32 {%%h%d,%%h%d},%%y%d; cvt.f32.f16 %%z%d,%%h%d; cvt.f32.f16 %%z%d,%%h%d;", q * 2, q * 2 + 1, e * 4 + q, q * 2, q * 2, q * 2 + 1, q * 2 + 1);
                }
            o(" st.shared.v4.f32 [%%rd11+%d],{%%z0,%%z1,%%z2,%%z3}; st.shared.v4.f32 [%%rd11+%d],{%%z4,%%z5,%%z6,%%z7};", e * 32 * 36 * 4, e * 32 * 36 * 4 + 16);
        }
    };
    // fp16 fragments read column pairs (2*t4): one extra t4 offset on top of the tf32 bases
    o(" mul.wide.u32 %%rd34,%%r61,4; add.u64 %%rd35,%%rd13,%%rd34; add.u64 %%rd34,%%rd12,%%rd34;");
    o(" mov.u32 %%r32,0;");
    gload("%r32");
    sstore("%r32");
    o(" bar.sync 0;");
    o("LK:");
    o(" add.u32 %%r33,%%r32,32; setp.lt.u32 %%p1,%%r33,%%r3;");
    o(" @!%%p1 bra NOPF;");
    gload("%r33");
    o("NOPF:");
    if (half) for (int ks = 0; ks < 32; ks += 16) {
        // m16n8k16 fp16 tensor cores (fp32 accumulate): A pairs (row, k..k+1), B pairs (n, k..k+1)
        for (int mt = 0; mt < 2; mt++) {
            int base = mt * 2304 + ks * 4;
            o(" ld.shared.v2.f32 {%%f4,%%f5},[%%rd34+%d]; ld.shared.v2.f32 {%%f6,%%f7},[%%rd34+%d];", base, base + 1152);
            o(" cvt.rn.f16x2.f32 %%a%d,%%f5,%%f4; cvt.rn.f16x2.f32 %%a%d,%%f7,%%f6;", mt * 4, mt * 4 + 1);
            o(" ld.shared.v2.f32 {%%f4,%%f5},[%%rd34+%d]; ld.shared.v2.f32 {%%f6,%%f7},[%%rd34+%d];", base + 32, base + 1152 + 32);
            o(" cvt.rn.f16x2.f32 %%a%d,%%f5,%%f4; cvt.rn.f16x2.f32 %%a%d,%%f7,%%f6;", mt * 4 + 2, mt * 4 + 3);
        }
        for (int nt = 0; nt < 4; nt++) {
            int base = nt * 1152 + ks * 4;
            o(" ld.shared.v2.f32 {%%f8,%%f9},[%%rd35+%d]; cvt.rn.f16x2.f32 %%b%d,%%f9,%%f8;", base, nt * 2);
            o(" ld.shared.v2.f32 {%%f8,%%f9},[%%rd35+%d]; cvt.rn.f16x2.f32 %%b%d,%%f9,%%f8;", base + 32, nt * 2 + 1);
        }
        for (int mt = 0; mt < 2; mt++)
            for (int nt = 0; nt < 4; nt++) {
                int c = (mt * 4 + nt) * 4;
                o(" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%%c%d,%%c%d,%%c%d,%%c%d},{%%a%d,%%a%d,%%a%d,%%a%d},{%%b%d,%%b%d},{%%c%d,%%c%d,%%c%d,%%c%d};",
                  c, c + 1, c + 2, c + 3, mt * 4, mt * 4 + 1, mt * 4 + 2, mt * 4 + 3, nt * 2, nt * 2 + 1, c, c + 1, c + 2, c + 3);
            }
    }
    else for (int ks = 0; ks < 32; ks += 8) {
        for (int mt = 0; mt < 2; mt++) {
            int base = mt * 2304 + ks * 4;
            o(" ld.shared.f32 %%f4,[%%rd12+%d]; ld.shared.f32 %%f5,[%%rd12+%d]; ld.shared.f32 %%f6,[%%rd12+%d]; ld.shared.f32 %%f7,[%%rd12+%d];",
              base, base + 1152, base + 16, base + 1168);
            o(" cvt.rna.tf32.f32 %%a%d,%%f4; cvt.rna.tf32.f32 %%a%d,%%f5; cvt.rna.tf32.f32 %%a%d,%%f6; cvt.rna.tf32.f32 %%a%d,%%f7;", mt * 4, mt * 4 + 1, mt * 4 + 2, mt * 4 + 3);
        }
        for (int nt = 0; nt < 4; nt++) {
            int base = nt * 1152 + ks * 4;
            o(" ld.shared.f32 %%f8,[%%rd13+%d]; ld.shared.f32 %%f9,[%%rd13+%d]; cvt.rna.tf32.f32 %%b%d,%%f8; cvt.rna.tf32.f32 %%b%d,%%f9;", base, base + 16, nt * 2, nt * 2 + 1);
        }
        for (int mt = 0; mt < 2; mt++)
            for (int nt = 0; nt < 4; nt++) {
                int c = (mt * 4 + nt) * 4;
                o(" mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%%c%d,%%c%d,%%c%d,%%c%d},{%%a%d,%%a%d,%%a%d,%%a%d},{%%b%d,%%b%d},{%%c%d,%%c%d,%%c%d,%%c%d};",
                  c, c + 1, c + 2, c + 3, mt * 4, mt * 4 + 1, mt * 4 + 2, mt * 4 + 3, nt * 2, nt * 2 + 1, c, c + 1, c + 2, c + 3);
            }
    }
    o(" bar.sync 0;");
    o(" @!%%p1 bra EK;");
    sstore("%r33");
    o(" bar.sync 0;");
    o(" mov.u32 %%r32,%%r33; bra LK;");
    o("EK:");
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 4; nt++)
            for (int h = 0; h < 2; h++)
                for (int j = 0; j < 2; j++) {
                    int ci = (mt * 4 + nt) * 4 + h * 2 + j;
                    o(" add.u32 %%r52,%%r16,%%r62; add.u32 %%r52,%%r52,%%r60; add.u32 %%r52,%%r52,%d;", mt * 16 + h * 8);
                    o(" add.u32 %%r53,%%r15,%%r63; shl.b32 %%r54,%%r61,1; add.u32 %%r53,%%r53,%%r54; add.u32 %%r53,%%r53,%d;", nt * 8 + j);
                    o(" setp.lt.u32 %%p4,%%r52,%%r1; setp.lt.u32 %%p5,%%r53,%%r2; and.pred %%p5,%%p5,%%p4; @!%%p5 bra S%d;", ci);
                    o(" mul.f32 %%f5,%%c%d,%%f1;", ci);
                    o(" @%%pbias selp.u32 %%r55,%%r52,%%r53,%%pbm; @%%pbias mul.wide.u32 %%rd15,%%r55,4; @%%pbias add.u64 %%rd15,%%rd4,%%rd15; @%%pbias ld.global.f32 %%f6,[%%rd15]; @%%pbias add.f32 %%f5,%%f5,%%f6;");
                    o(" mad.lo.u32 %%r56,%%r52,%%r6,%%r53; mul.wide.u32 %%rd16,%%r56,4; add.u64 %%rd16,%%rd3,%%rd16;");
                    o(" @%%pbeta ld.global.f32 %%f7,[%%rd16]; @%%pbeta fma.rn.f32 %%f5,%%f7,%%f2,%%f5; st.global.f32 [%%rd16],%%f5;");
                    o("S%d:", ci);
                }
    o(" ret;\n}");
    return o.s;
}

// ---------------------------------------------------------------------------------------------
// Flash attention (fp16 tensor cores, fp32 softmax state), any head dim D % 8 == 0 up to 128, no mask.
// q [T][ld], k/v [S][ld] with head h at column h*D; out [T][ld]; ld % 4 == 0. grid = (ceil(T/128), heads), 256 threads.
// Each of the 8 warps owns 16 query rows; keys stream through shared memory 64 at a time (K and V row-major,
// fp16, head dim padded with zeros to a multiple of 16) with an online softmax, so the T x S score
// matrix never exists in memory.
std::string gen_flash(const char* name, int D) {
    Out o;
    const int KC = (D + 15) / 16, DP = KC * 16, DT = D / 8, NT = 256, G4 = D / 4;
    o(".visible .entry %s(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pO, .param .u32 pT, .param .u32 pS, .param .u32 pLd, .param .f32 pScale)", name);
    o("{");
    o(" .reg .pred %%p<8>; .reg .b32 %%r<64>; .reg .b64 %%rd<32>; .reg .f32 %%f<32>; .reg .b16 %%h<8>;");
    o(" .reg .b32 %%q<%d>; .reg .f32 %%s<32>; .reg .f32 %%o<%d>; .reg .b32 %%a<4>; .reg .b32 %%b<2>; .reg .f32 %%m<2>; .reg .f32 %%l<2>;", KC * 4, DT * 4);
    o(" .shared .align 16 .b16 sK[%d]; .shared .align 16 .b16 sV[%d];", 64 * DP, 64 * DP);
    o(" ld.param.u64 %%rd1,[pQ]; ld.param.u64 %%rd2,[pK]; ld.param.u64 %%rd3,[pV]; ld.param.u64 %%rd4,[pO];");
    o(" ld.param.u32 %%r1,[pT]; ld.param.u32 %%r2,[pS]; ld.param.u32 %%r3,[pLd]; ld.param.f32 %%f20,[pScale];");
    o(" mov.u32 %%r4,%%tid.x; and.b32 %%r5,%%r4,31; shr.u32 %%r6,%%r4,5; shr.u32 %%r7,%%r5,2; and.b32 %%r8,%%r5,3;");   // tid lane warp g t
    o(" mov.u32 %%r9,%%ctaid.x; shl.b32 %%r9,%%r9,7; shl.b32 %%r10,%%r6,4; add.u32 %%r9,%%r9,%%r10;");                  // q0 = block*128 + warp*16
    o(" mov.u32 %%r11,%%ctaid.y; mul.lo.u32 %%r11,%%r11,%d;", D);                                                       // head column offset
    o(" add.u32 %%r12,%%r9,%%r7; add.u32 %%r13,%%r12,8; setp.lt.u32 %%p1,%%r12,%%r1; setp.lt.u32 %%p2,%%r13,%%r1;");   // rows g, g+8
    o(" mov.u64 %%rd5,sK; mov.u64 %%rd6,sV;");
    // Q fragments (pre-scaled by softmax scale * log2(e)) as fp16 pairs; padding dims are zero
    o(" mad.lo.u32 %%r14,%%r12,%%r3,%%r11; mul.wide.u32 %%rd7,%%r14,4; add.u64 %%rd7,%%rd1,%%rd7;");
    o(" mad.lo.u32 %%r15,%%r13,%%r3,%%r11; mul.wide.u32 %%rd8,%%r15,4; add.u64 %%rd8,%%rd1,%%rd8;");
    o(" shl.b32 %%r16,%%r8,3; cvt.u64.u32 %%rd9,%%r16;");                                                   // 2t floats = 8t bytes
    for (int c = 0; c < KC; c++) {
        for (int part = 0; part < 4; part++) {
            const char* row = (part & 1) ? "%rd8" : "%rd7";
            const char* pr = (part & 1) ? "%p2" : "%p1";
            int col0 = c * 16 + (part >= 2 ? 8 : 0), coff = col0 * 4;
            if (col0 < D) o(" add.u64 %%rd10,%s,%%rd9; mov.f32 %%f1,0f00000000; mov.f32 %%f2,0f00000000; @%s ld.global.v2.f32 {%%f1,%%f2},[%%rd10+%d];", row, pr, coff);
            else o(" mov.f32 %%f1,0f00000000; mov.f32 %%f2,0f00000000;");
            o(" mul.f32 %%f1,%%f1,%%f20; mul.f32 %%f2,%%f2,%%f20; cvt.rn.f16x2.f32 %%q%d,%%f2,%%f1;", c * 4 + part);
        }
    }
    for (int i = 0; i < DT * 4; i++) o(" mov.f32 %%o%d,0f00000000;", i);
    o(" mov.f32 %%m0,0fF149F2CA; mov.f32 %%m1,0fF149F2CA; mov.f32 %%l0,0f00000000; mov.f32 %%l1,0f00000000;");
    if (DP > D) {   // zero the padded K columns once; staging never writes them
        o(" mov.u32 %%r21,%%r4;");
        o("PZ:");
        o(" setp.ge.u32 %%p4,%%r21,%d; @%%p4 bra PZE;", 64 * (DP - D));
        o(" div.u32 %%r22,%%r21,%d; rem.u32 %%r23,%%r21,%d; mad.lo.u32 %%r26,%%r22,%d,%d; add.u32 %%r26,%%r26,%%r23;", DP - D, DP - D, DP, D);
        o(" mul.wide.u32 %%rd14,%%r26,2; add.u64 %%rd14,%%rd5,%%rd14; mov.b16 %%h3,0; st.shared.b16 [%%rd14],%%h3; add.u32 %%r21,%%r21,%d; bra PZ;", NT);
        o("PZE:");
    }
    // per-thread PV fragment base: rows 2t (+1, +8, +9), column g
    o(" shl.b32 %%r40,%%r8,1; mul.lo.u32 %%r40,%%r40,%d; add.u32 %%r40,%%r40,%%r7; mul.wide.u32 %%rd20,%%r40,2; add.u64 %%rd20,%%rd6,%%rd20;", DP);
    o(" mov.u32 %%r20,0;");   // key block start
    o("KB:");
    o(" setp.ge.u32 %%p3,%%r20,%%r2; @%%p3 bra END;");
    o(" bar.sync 0;");
    // stage K and V (64 keys x D) as fp16, 4 columns per thread step
    o(" mov.u32 %%r21,%%r4;");
    o("LD:");
    o(" setp.ge.u32 %%p4,%%r21,%d; @%%p4 bra LDE;", 64 * G4);
    o(" div.u32 %%r22,%%r21,%d; rem.u32 %%r23,%%r21,%d; shl.b32 %%r23,%%r23,2; add.u32 %%r24,%%r20,%%r22; setp.lt.u32 %%p5,%%r24,%%r2;", G4, G4);   // key, col
    o(" mad.lo.u32 %%r25,%%r24,%%r3,%%r11; add.u32 %%r25,%%r25,%%r23; mul.wide.u32 %%rd11,%%r25,4;");
    o(" add.u64 %%rd12,%%rd2,%%rd11; add.u64 %%rd13,%%rd3,%%rd11;");
    o(" mov.f32 %%f3,0f00000000; mov.f32 %%f4,0f00000000; mov.f32 %%f5,0f00000000; mov.f32 %%f6,0f00000000;");
    o(" mov.f32 %%f7,0f00000000; mov.f32 %%f8,0f00000000; mov.f32 %%f9,0f00000000; mov.f32 %%f10,0f00000000;");
    o(" @%%p5 ld.global.nc.v4.f32 {%%f3,%%f4,%%f5,%%f6},[%%rd12]; @%%p5 ld.global.nc.v4.f32 {%%f7,%%f8,%%f9,%%f10},[%%rd13];");
    o(" cvt.rn.f16x2.f32 %%a0,%%f4,%%f3; cvt.rn.f16x2.f32 %%a1,%%f6,%%f5; cvt.rn.f16x2.f32 %%a2,%%f8,%%f7; cvt.rn.f16x2.f32 %%a3,%%f10,%%f9;");
    o(" mad.lo.u32 %%r26,%%r22,%d,%%r23; mul.wide.u32 %%rd14,%%r26,2;", DP);
    o(" add.u64 %%rd15,%%rd5,%%rd14; st.shared.v2.b32 [%%rd15],{%%a0,%%a1}; add.u64 %%rd16,%%rd6,%%rd14; st.shared.v2.b32 [%%rd16],{%%a2,%%a3};");
    o(" add.u32 %%r21,%%r21,%d; bra LD;", NT);
    o("LDE:");
    o(" bar.sync 0;");
    // S = Q K^T for 64 keys: 8 tiles of 16x8
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 4; i++) o(" mov.f32 %%s%d,0f00000000;", j * 4 + i);
        o(" add.u32 %%r27,%%r7,%d; mul.lo.u32 %%r27,%%r27,%d; shl.b32 %%r28,%%r8,1; add.u32 %%r27,%%r27,%%r28; mul.wide.u32 %%rd17,%%r27,2; add.u64 %%rd17,%%rd5,%%rd17;", j * 8, DP);
        for (int c = 0; c < KC; c++) {
            o(" ld.shared.b32 %%b0,[%%rd17+%d]; ld.shared.b32 %%b1,[%%rd17+%d];", c * 32, c * 32 + 16);
            o(" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%%s%d,%%s%d,%%s%d,%%s%d},{%%q%d,%%q%d,%%q%d,%%q%d},{%%b0,%%b1},{%%s%d,%%s%d,%%s%d,%%s%d};",
              j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3, c * 4, c * 4 + 1, c * 4 + 2, c * 4 + 3, j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3);
        }
        o(" add.u32 %%r29,%%r20,%d; shl.b32 %%r30,%%r8,1; add.u32 %%r29,%%r29,%%r30; setp.ge.u32 %%p6,%%r29,%%r2; add.u32 %%r31,%%r29,1; setp.ge.u32 %%p7,%%r31,%%r2;", j * 8);
        o(" @%%p6 mov.f32 %%s%d,0fF149F2CA; @%%p6 mov.f32 %%s%d,0fF149F2CA; @%%p7 mov.f32 %%s%d,0fF149F2CA; @%%p7 mov.f32 %%s%d,0fF149F2CA;", j * 4, j * 4 + 2, j * 4 + 1, j * 4 + 3);
    }
    // online softmax (base 2; q was pre-scaled by log2 e)
    o(" mov.f32 %%f11,%%m0; mov.f32 %%f12,%%m1;");
    for (int j = 0; j < 8; j++) o(" max.f32 %%f11,%%f11,%%s%d; max.f32 %%f11,%%f11,%%s%d; max.f32 %%f12,%%f12,%%s%d; max.f32 %%f12,%%f12,%%s%d;", j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3);
    for (int x : {1, 2}) o(" shfl.sync.bfly.b32 %%f13,%%f11,%d,31,0xffffffff; max.f32 %%f11,%%f11,%%f13; shfl.sync.bfly.b32 %%f13,%%f12,%d,31,0xffffffff; max.f32 %%f12,%%f12,%%f13;", x, x);
    o(" sub.f32 %%f14,%%m0,%%f11; ex2.approx.f32 %%f14,%%f14; sub.f32 %%f15,%%m1,%%f12; ex2.approx.f32 %%f15,%%f15; mov.f32 %%m0,%%f11; mov.f32 %%m1,%%f12;");
    o(" mov.f32 %%f16,0f00000000; mov.f32 %%f17,0f00000000;");
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 4; i++) {
            const char* mm = i < 2 ? "%m0" : "%m1";
            int acc = i < 2 ? 16 : 17;
            o(" sub.f32 %%s%d,%%s%d,%s; ex2.approx.f32 %%s%d,%%s%d; add.f32 %%f%d,%%f%d,%%s%d;", j * 4 + i, j * 4 + i, mm, j * 4 + i, j * 4 + i, acc, acc, j * 4 + i);
        }
    for (int x : {1, 2}) o(" shfl.sync.bfly.b32 %%f13,%%f16,%d,31,0xffffffff; add.f32 %%f16,%%f16,%%f13; shfl.sync.bfly.b32 %%f13,%%f17,%d,31,0xffffffff; add.f32 %%f17,%%f17,%%f13;", x, x);
    o(" fma.rn.f32 %%l0,%%l0,%%f14,%%f16; fma.rn.f32 %%l1,%%l1,%%f15,%%f17;");
    for (int dt = 0; dt < DT; dt++) o(" mul.f32 %%o%d,%%o%d,%%f14; mul.f32 %%o%d,%%o%d,%%f14; mul.f32 %%o%d,%%o%d,%%f15; mul.f32 %%o%d,%%o%d,%%f15;",
                                      dt * 4, dt * 4, dt * 4 + 1, dt * 4 + 1, dt * 4 + 2, dt * 4 + 2, dt * 4 + 3, dt * 4 + 3);
    // O += P V: P from the S accumulators (fp16); V fragments gathered from the row-major tile
    for (int kc = 0; kc < 4; kc++) {
        int j0 = kc * 2, j1 = kc * 2 + 1;
        o(" cvt.rn.f16x2.f32 %%a0,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a1,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a2,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a3,%%s%d,%%s%d;",
          j0 * 4 + 1, j0 * 4, j0 * 4 + 3, j0 * 4 + 2, j1 * 4 + 1, j1 * 4, j1 * 4 + 3, j1 * 4 + 2);
        for (int dt = 0; dt < DT; dt++) {
            int off = (kc * 16 * DP + dt * 8) * 2;
            o(" ld.shared.b16 %%h0,[%%rd20+%d]; ld.shared.b16 %%h1,[%%rd20+%d]; ld.shared.b16 %%h2,[%%rd20+%d]; ld.shared.b16 %%h3,[%%rd20+%d];",
              off, off + DP * 2, off + 8 * DP * 2, off + 9 * DP * 2);
            o(" mov.b32 %%b0,{%%h0,%%h1}; mov.b32 %%b1,{%%h2,%%h3};");
            o(" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%%o%d,%%o%d,%%o%d,%%o%d},{%%a0,%%a1,%%a2,%%a3},{%%b0,%%b1},{%%o%d,%%o%d,%%o%d,%%o%d};",
              dt * 4, dt * 4 + 1, dt * 4 + 2, dt * 4 + 3, dt * 4, dt * 4 + 1, dt * 4 + 2, dt * 4 + 3);
        }
    }
    o(" add.u32 %%r20,%%r20,64; bra KB;");
    o("END:");
    o(" rcp.approx.f32 %%f18,%%l0; rcp.approx.f32 %%f19,%%l1;");
    o(" mad.lo.u32 %%r34,%%r12,%%r3,%%r11; mul.wide.u32 %%rd18,%%r34,4; add.u64 %%rd18,%%rd4,%%rd18; add.u64 %%rd18,%%rd18,%%rd9;");
    o(" mad.lo.u32 %%r35,%%r13,%%r3,%%r11; mul.wide.u32 %%rd19,%%r35,4; add.u64 %%rd19,%%rd4,%%rd19; add.u64 %%rd19,%%rd19,%%rd9;");
    for (int dt = 0; dt < DT; dt++) {
        o(" mul.f32 %%f21,%%o%d,%%f18; mul.f32 %%f22,%%o%d,%%f18; @%%p1 st.global.v2.f32 [%%rd18+%d],{%%f21,%%f22};", dt * 4, dt * 4 + 1, dt * 32);
        o(" mul.f32 %%f23,%%o%d,%%f19; mul.f32 %%f24,%%o%d,%%f19; @%%p2 st.global.v2.f32 [%%rd19+%d],{%%f23,%%f24};", dt * 4 + 2, dt * 4 + 3, dt * 32);
    }
    o(" ret;\n}");
    return o.s;
}

// ---------------------------------------------------------------------------------------------
// Causal flash attention for chat-model prompts, straight from the fp16 KV cache.
// q [T][ldq] f32 (head h at column h*D), K/V cache [pos][kvd] f16 (kv head = h / grp), out [T][ldq].
// Query row t sits at position P0 + t and sees keys [lo, hi]: hi = P0 + t (or P0 + NC - 1 for a non-causal
// image block), lo = hi - WIN + 1 on sliding-window layers. Output dims [DV0, DV0 + DV): heads wider than 128
// run in two launches so the accumulators fit in registers. grid = (ceil(T/128), heads), 256 threads.
// q8: the cache holds int8 values plus an f32 scale per (token, kv head) in separate arrays.
std::string gen_flash_kv(const char* name, int D, int DV, bool q8 = false) {
    Out o;
    // int8 cache: 4 values + the (token, kv head) scale -> 4 fp16 in %%a{lo},%%a{hi}
    auto deq4 = [&](const char* base_bytes, const char* scales, int lo, int hi) {
        o(" add.u64 %%rd28,%s,%%rd27; mov.b32 %%e0,0; @%%p5 ld.global.nc.b32 %%e0,[%%rd28];", base_bytes);
        o(" mad.lo.u32 %%r71,%%r24,%%r72,%%r70; mul.wide.u32 %%rd29,%%r71,4; add.u64 %%rd29,%s,%%rd29; mov.f32 %%g4,0f00000000; @%%p5 ld.global.nc.f32 %%g4,[%%rd29];", scales);
        for (int i = 0; i < 4; i++) o(" bfe.s32 %%e%d,%%e0,%d,8; cvt.rn.f32.s32 %%g%d,%%e%d; mul.f32 %%g%d,%%g%d,%%g4;", i + 1, i * 8, i, i + 1, i, i);
        o(" cvt.rn.f16x2.f32 %%a%d,%%g1,%%g0; cvt.rn.f16x2.f32 %%a%d,%%g3,%%g2;", lo, hi);
    };

    const int KC = (D + 15) / 16, DP = KC * 16, DT = DV / 8, NT = 256, G4 = D / 4, GV4 = DV / 4;
    o(".visible .entry %s(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pO, .param .u32 pT, .param .u32 pP0, .param .u32 pLdq,"
      " .param .u32 pKvd, .param .u32 pGrp, .param .f32 pScale, .param .u32 pWin, .param .u32 pNC, .param .u32 pDV0%s)", name,
      q8 ? ", .param .u64 pKs, .param .u64 pVs, .param .u32 pNKV" : "");
    o("{");
    o(" .reg .b32 %%e<8>; .reg .f32 %%g<6>;");
    o(" .reg .pred %%p<12>; .reg .b32 %%r<80>; .reg .b64 %%rd<32>; .reg .f32 %%f<32>; .reg .b16 %%h<8>;");
    o(" .reg .b32 %%q<%d>; .reg .f32 %%s<32>; .reg .f32 %%o<%d>; .reg .b32 %%a<4>; .reg .b32 %%b<2>; .reg .f32 %%m<2>; .reg .f32 %%l<2>;", KC * 4, DT * 4);
    o(" .shared .align 16 .b16 sK[%d]; .shared .align 16 .b16 sV[%d];", 64 * DP, 64 * DV);
    o(" ld.param.u64 %%rd1,[pQ]; ld.param.u64 %%rd2,[pK]; ld.param.u64 %%rd3,[pV]; ld.param.u64 %%rd4,[pO];");
    o(" ld.param.u32 %%r1,[pT]; ld.param.u32 %%r2,[pP0]; ld.param.u32 %%r3,[pLdq]; ld.param.u32 %%r50,[pKvd]; ld.param.u32 %%r51,[pGrp];");
    o(" ld.param.f32 %%f20,[pScale]; ld.param.u32 %%r52,[pWin]; ld.param.u32 %%r53,[pNC]; ld.param.u32 %%r54,[pDV0];");
    o(" mov.u32 %%r4,%%tid.x; and.b32 %%r5,%%r4,31; shr.u32 %%r6,%%r4,5; shr.u32 %%r7,%%r5,2; and.b32 %%r8,%%r5,3;");   // tid lane warp g t
    o(" mov.u32 %%r55,%%ctaid.x; shl.b32 %%r55,%%r55,7; shl.b32 %%r10,%%r6,4; add.u32 %%r9,%%r55,%%r10;");              // block q0, warp q0
    o(" mov.u32 %%r11,%%ctaid.y; div.u32 %%r56,%%r11,%%r51; mov.u32 %%r70,%%r56; mul.lo.u32 %%r56,%%r56,%d; mul.lo.u32 %%r11,%%r11,%d;", D, D);   // kv col, q col
    if (q8) o(" ld.param.u64 %%rd25,[pKs]; ld.param.u64 %%rd26,[pVs]; ld.param.u32 %%r72,[pNKV];");
    o(" add.u32 %%r12,%%r9,%%r7; add.u32 %%r13,%%r12,8; setp.lt.u32 %%p1,%%r12,%%r1; setp.lt.u32 %%p2,%%r13,%%r1;");
    // key limits for rows g and g+8: hi (inclusive) and lo
    o(" setp.ne.u32 %%p8,%%r53,0; add.u32 %%r57,%%r2,%%r53; sub.u32 %%r57,%%r57,1;");                                 // NC block end
    o(" add.u32 %%r58,%%r2,%%r12; add.u32 %%r59,%%r2,%%r13; @%%p8 mov.u32 %%r58,%%r57; @%%p8 mov.u32 %%r59,%%r57;");  // hi_g, hi_g8
    o(" mov.u32 %%r60,0; mov.u32 %%r61,0; selp.u32 %%r69,0,%%r52,%%p8; setp.ne.u32 %%p9,%%r69,0;");
    o(" add.u32 %%r62,%%r58,1; setp.gt.u32 %%p10,%%r62,%%r69; and.pred %%p10,%%p10,%%p9; @%%p10 sub.u32 %%r60,%%r62,%%r69;");
    o(" add.u32 %%r63,%%r59,1; setp.gt.u32 %%p11,%%r63,%%r69; and.pred %%p11,%%p11,%%p9; @%%p11 sub.u32 %%r61,%%r63,%%r69;");
    // block key range [kb0, kend): from the window start of the block's first row to the last row's limit
    o(" add.u32 %%r64,%%r55,127; sub.u32 %%r65,%%r1,1; min.u32 %%r64,%%r64,%%r65; add.u32 %%r66,%%r2,%%r64; add.u32 %%r66,%%r66,1; @%%p8 add.u32 %%r66,%%r2,%%r53;");   // kend
    o(" mov.u32 %%r20,0; add.u32 %%r67,%%r2,%%r55; add.u32 %%r67,%%r67,1; setp.gt.u32 %%p10,%%r67,%%r69; and.pred %%p10,%%p10,%%p9; @%%p10 sub.u32 %%r20,%%r67,%%r69; and.b32 %%r20,%%r20,-64;");
    o(" mov.u64 %%rd5,sK; mov.u64 %%rd6,sV;");
    // Q fragments, pre-scaled by softmax scale * log2(e)
    o(" mad.lo.u32 %%r14,%%r12,%%r3,%%r11; mul.wide.u32 %%rd7,%%r14,4; add.u64 %%rd7,%%rd1,%%rd7;");
    o(" mad.lo.u32 %%r15,%%r13,%%r3,%%r11; mul.wide.u32 %%rd8,%%r15,4; add.u64 %%rd8,%%rd1,%%rd8;");
    o(" shl.b32 %%r16,%%r8,3; cvt.u64.u32 %%rd9,%%r16;");
    for (int c = 0; c < KC; c++)
        for (int part = 0; part < 4; part++) {
            const char* row = (part & 1) ? "%rd8" : "%rd7";
            const char* pr = (part & 1) ? "%p2" : "%p1";
            int col0 = c * 16 + (part >= 2 ? 8 : 0);
            if (col0 < D) o(" add.u64 %%rd10,%s,%%rd9; mov.f32 %%f1,0f00000000; mov.f32 %%f2,0f00000000; @%s ld.global.v2.f32 {%%f1,%%f2},[%%rd10+%d];", row, pr, col0 * 4);
            else o(" mov.f32 %%f1,0f00000000; mov.f32 %%f2,0f00000000;");
            o(" mul.f32 %%f1,%%f1,%%f20; mul.f32 %%f2,%%f2,%%f20; cvt.rn.f16x2.f32 %%q%d,%%f2,%%f1;", c * 4 + part);
        }
    for (int i = 0; i < DT * 4; i++) o(" mov.f32 %%o%d,0f00000000;", i);
    o(" mov.f32 %%m0,0fF149F2CA; mov.f32 %%m1,0fF149F2CA; mov.f32 %%l0,0f00000000; mov.f32 %%l1,0f00000000;");
    if (DP > D) {
        o(" mov.u32 %%r21,%%r4;");
        o("PZ:");
        o(" setp.ge.u32 %%p4,%%r21,%d; @%%p4 bra PZE;", 64 * (DP - D));
        o(" div.u32 %%r22,%%r21,%d; rem.u32 %%r23,%%r21,%d; mad.lo.u32 %%r26,%%r22,%d,%d; add.u32 %%r26,%%r26,%%r23;", DP - D, DP - D, DP, D);
        o(" mul.wide.u32 %%rd14,%%r26,2; add.u64 %%rd14,%%rd5,%%rd14; mov.b16 %%h3,0; st.shared.b16 [%%rd14],%%h3; add.u32 %%r21,%%r21,%d; bra PZ;", NT);
        o("PZE:");
    }
    o(" shl.b32 %%r40,%%r8,1; mul.lo.u32 %%r40,%%r40,%d; add.u32 %%r40,%%r40,%%r7; mul.wide.u32 %%rd20,%%r40,2; add.u64 %%rd20,%%rd6,%%rd20;", DV);
    o("KB:");
    o(" setp.ge.u32 %%p3,%%r20,%%r66; @%%p3 bra END;");
    o(" bar.sync 0;");
    // stage K (all D dims) and V (dims DV0..DV0+DV) as fp16, 4 values per step
    o(" mov.u32 %%r21,%%r4;");
    o("LD:");
    o(" setp.ge.u32 %%p4,%%r21,%d; @%%p4 bra LDE;", 64 * G4);
    o(" div.u32 %%r22,%%r21,%d; rem.u32 %%r23,%%r21,%d; shl.b32 %%r23,%%r23,2; add.u32 %%r24,%%r20,%%r22; setp.lt.u32 %%p5,%%r24,%%r66;", G4, G4);
    o(" mad.lo.u32 %%r25,%%r24,%%r50,%%r56; add.u32 %%r25,%%r25,%%r23; mul.wide.u32 %%rd11,%%r25,2; add.u64 %%rd12,%%rd2,%%rd11;");
    if (q8) { o(" cvt.u64.u32 %%rd27,%%r25;"); deq4("%rd2", "%rd25", 0, 1); }
    else o(" mov.b32 %%a0,0; mov.b32 %%a1,0; @%%p5 ld.global.nc.v2.b32 {%%a0,%%a1},[%%rd12];");
    o(" mad.lo.u32 %%r26,%%r22,%d,%%r23; mul.wide.u32 %%rd14,%%r26,2; add.u64 %%rd15,%%rd5,%%rd14; st.shared.v2.b32 [%%rd15],{%%a0,%%a1};", DP);
    o(" add.u32 %%r21,%%r21,%d; bra LD;", NT);
    o("LDE:");
    o(" mov.u32 %%r21,%%r4;");
    o("LV:");
    o(" setp.ge.u32 %%p4,%%r21,%d; @%%p4 bra LVE;", 64 * GV4);
    o(" div.u32 %%r22,%%r21,%d; rem.u32 %%r23,%%r21,%d; shl.b32 %%r23,%%r23,2; add.u32 %%r24,%%r20,%%r22; setp.lt.u32 %%p5,%%r24,%%r66;", GV4, GV4);
    o(" mad.lo.u32 %%r25,%%r24,%%r50,%%r56; add.u32 %%r25,%%r25,%%r54; add.u32 %%r25,%%r25,%%r23; mul.wide.u32 %%rd11,%%r25,2; add.u64 %%rd13,%%rd3,%%rd11;");
    if (q8) { o(" cvt.u64.u32 %%rd27,%%r25;"); deq4("%rd3", "%rd26", 2, 3); }
    else o(" mov.b32 %%a2,0; mov.b32 %%a3,0; @%%p5 ld.global.nc.v2.b32 {%%a2,%%a3},[%%rd13];");
    o(" mad.lo.u32 %%r26,%%r22,%d,%%r23; mul.wide.u32 %%rd14,%%r26,2; add.u64 %%rd16,%%rd6,%%rd14; st.shared.v2.b32 [%%rd16],{%%a2,%%a3};", DV);
    o(" add.u32 %%r21,%%r21,%d; bra LV;", NT);
    o("LVE:");
    o(" bar.sync 0;");
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 4; i++) o(" mov.f32 %%s%d,0f00000000;", j * 4 + i);
        o(" add.u32 %%r27,%%r7,%d; mul.lo.u32 %%r27,%%r27,%d; shl.b32 %%r28,%%r8,1; add.u32 %%r27,%%r27,%%r28; mul.wide.u32 %%rd17,%%r27,2; add.u64 %%rd17,%%rd5,%%rd17;", j * 8, DP);
        for (int c = 0; c < KC; c++) {
            o(" ld.shared.b32 %%b0,[%%rd17+%d]; ld.shared.b32 %%b1,[%%rd17+%d];", c * 32, c * 32 + 16);
            o(" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%%s%d,%%s%d,%%s%d,%%s%d},{%%q%d,%%q%d,%%q%d,%%q%d},{%%b0,%%b1},{%%s%d,%%s%d,%%s%d,%%s%d};",
              j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3, c * 4, c * 4 + 1, c * 4 + 2, c * 4 + 3, j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3);
        }
        // mask: key k0 = kb + 8j + 2t (and k0 + 1) against each row's [lo, hi]
        o(" add.u32 %%r29,%%r20,%d; shl.b32 %%r30,%%r8,1; add.u32 %%r29,%%r29,%%r30; add.u32 %%r31,%%r29,1;", j * 8);
        o(" setp.gt.u32 %%p6,%%r29,%%r58; setp.lt.or.u32 %%p6,%%r29,%%r60,%%p6; @%%p6 mov.f32 %%s%d,0fF149F2CA;", j * 4);
        o(" setp.gt.u32 %%p6,%%r31,%%r58; setp.lt.or.u32 %%p6,%%r31,%%r60,%%p6; @%%p6 mov.f32 %%s%d,0fF149F2CA;", j * 4 + 1);
        o(" setp.gt.u32 %%p6,%%r29,%%r59; setp.lt.or.u32 %%p6,%%r29,%%r61,%%p6; @%%p6 mov.f32 %%s%d,0fF149F2CA;", j * 4 + 2);
        o(" setp.gt.u32 %%p6,%%r31,%%r59; setp.lt.or.u32 %%p6,%%r31,%%r61,%%p6; @%%p6 mov.f32 %%s%d,0fF149F2CA;", j * 4 + 3);
    }
    o(" mov.f32 %%f11,%%m0; mov.f32 %%f12,%%m1;");
    for (int j = 0; j < 8; j++) o(" max.f32 %%f11,%%f11,%%s%d; max.f32 %%f11,%%f11,%%s%d; max.f32 %%f12,%%f12,%%s%d; max.f32 %%f12,%%f12,%%s%d;", j * 4, j * 4 + 1, j * 4 + 2, j * 4 + 3);
    for (int x : {1, 2}) o(" shfl.sync.bfly.b32 %%f13,%%f11,%d,31,0xffffffff; max.f32 %%f11,%%f11,%%f13; shfl.sync.bfly.b32 %%f13,%%f12,%d,31,0xffffffff; max.f32 %%f12,%%f12,%%f13;", x, x);
    o(" sub.f32 %%f14,%%m0,%%f11; ex2.approx.f32 %%f14,%%f14; sub.f32 %%f15,%%m1,%%f12; ex2.approx.f32 %%f15,%%f15; mov.f32 %%m0,%%f11; mov.f32 %%m1,%%f12;");
    o(" mov.f32 %%f16,0f00000000; mov.f32 %%f17,0f00000000;");
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 4; i++) {
            const char* mm = i < 2 ? "%m0" : "%m1";
            int acc = i < 2 ? 16 : 17;
            o(" sub.f32 %%s%d,%%s%d,%s; ex2.approx.f32 %%s%d,%%s%d; add.f32 %%f%d,%%f%d,%%s%d;", j * 4 + i, j * 4 + i, mm, j * 4 + i, j * 4 + i, acc, acc, j * 4 + i);
        }
    for (int x : {1, 2}) o(" shfl.sync.bfly.b32 %%f13,%%f16,%d,31,0xffffffff; add.f32 %%f16,%%f16,%%f13; shfl.sync.bfly.b32 %%f13,%%f17,%d,31,0xffffffff; add.f32 %%f17,%%f17,%%f13;", x, x);
    o(" fma.rn.f32 %%l0,%%l0,%%f14,%%f16; fma.rn.f32 %%l1,%%l1,%%f15,%%f17;");
    for (int dt = 0; dt < DT; dt++) o(" mul.f32 %%o%d,%%o%d,%%f14; mul.f32 %%o%d,%%o%d,%%f14; mul.f32 %%o%d,%%o%d,%%f15; mul.f32 %%o%d,%%o%d,%%f15;",
                                      dt * 4, dt * 4, dt * 4 + 1, dt * 4 + 1, dt * 4 + 2, dt * 4 + 2, dt * 4 + 3, dt * 4 + 3);
    for (int kc = 0; kc < 4; kc++) {
        int j0 = kc * 2, j1 = kc * 2 + 1;
        o(" cvt.rn.f16x2.f32 %%a0,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a1,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a2,%%s%d,%%s%d; cvt.rn.f16x2.f32 %%a3,%%s%d,%%s%d;",
          j0 * 4 + 1, j0 * 4, j0 * 4 + 3, j0 * 4 + 2, j1 * 4 + 1, j1 * 4, j1 * 4 + 3, j1 * 4 + 2);
        for (int dt = 0; dt < DT; dt++) {
            int off = (kc * 16 * DV + dt * 8) * 2;
            o(" ld.shared.b16 %%h0,[%%rd20+%d]; ld.shared.b16 %%h1,[%%rd20+%d]; ld.shared.b16 %%h2,[%%rd20+%d]; ld.shared.b16 %%h3,[%%rd20+%d];",
              off, off + DV * 2, off + 8 * DV * 2, off + 9 * DV * 2);
            o(" mov.b32 %%b0,{%%h0,%%h1}; mov.b32 %%b1,{%%h2,%%h3};");
            o(" mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%%o%d,%%o%d,%%o%d,%%o%d},{%%a0,%%a1,%%a2,%%a3},{%%b0,%%b1},{%%o%d,%%o%d,%%o%d,%%o%d};",
              dt * 4, dt * 4 + 1, dt * 4 + 2, dt * 4 + 3, dt * 4, dt * 4 + 1, dt * 4 + 2, dt * 4 + 3);
        }
    }
    o(" add.u32 %%r20,%%r20,64; bra KB;");
    o("END:");
    o(" rcp.approx.f32 %%f18,%%l0; rcp.approx.f32 %%f19,%%l1;");
    o(" add.u32 %%r68,%%r11,%%r54;");   // output column = head col + DV0
    o(" mad.lo.u32 %%r34,%%r12,%%r3,%%r68; mul.wide.u32 %%rd18,%%r34,4; add.u64 %%rd18,%%rd4,%%rd18; add.u64 %%rd18,%%rd18,%%rd9;");
    o(" mad.lo.u32 %%r35,%%r13,%%r3,%%r68; mul.wide.u32 %%rd19,%%r35,4; add.u64 %%rd19,%%rd4,%%rd19; add.u64 %%rd19,%%rd19,%%rd9;");
    for (int dt = 0; dt < DT; dt++) {
        o(" mul.f32 %%f21,%%o%d,%%f18; mul.f32 %%f22,%%o%d,%%f18; @%%p1 st.global.v2.f32 [%%rd18+%d],{%%f21,%%f22};", dt * 4, dt * 4 + 1, dt * 32);
        o(" mul.f32 %%f23,%%o%d,%%f19; mul.f32 %%f24,%%o%d,%%f19; @%%p2 st.global.v2.f32 [%%rd19+%d],{%%f23,%%f24};", dt * 4 + 2, dt * 4 + 3, dt * 32);
    }
    o(" ret;\n}");
    return o.s;
}

// Elementwise kernel skeleton: i = global thread index (%r5), exits when i >= pN (%r1).
std::string ew(const char* name, const char* params, const char* loads, const char* body) {
    std::string s = ".visible .entry ";
    s += name; s += "("; s += params; s += ", .param .u32 pN)\n{\n";
    s += " .reg .pred %p<6>; .reg .b32 %r<32>; .reg .f32 %f<24>; .reg .b64 %rd<16>; .reg .b16 %h<4>;\n";
    s += " ld.param.u32 %r1,[pN];\n";
    s += loads;
    s += " mov.u32 %r2,%ctaid.x; mov.u32 %r3,%ntid.x; mov.u32 %r4,%tid.x; mad.lo.u32 %r5,%r2,%r3,%r4;\n";
    s += " mov.u32 %r6,%ctaid.y; mov.u32 %r7,%nctaid.x; mul.lo.u32 %r6,%r6,%r7; mad.lo.u32 %r5,%r6,%r3,%r5;\n";  // 2D grid for big n
    s += " setp.ge.u32 %p1,%r5,%r1; @%p1 bra DONE;\n";
    s += body;
    s += "DONE:\n ret;\n}\n";
    return s;
}

// x in %f2 -> gelu(x) (erf form, Abramowitz-Stegun 7.1.26, |err| < 1.5e-7) in %f2. Uses %f10-%f19.
const char* kGelu =
    " mul.f32 %f10,%f2,0f3F3504F3; abs.f32 %f11,%f10;\n"                       // z = x/sqrt2, |z|
    " fma.rn.f32 %f12,%f11,0f3EA7BA05,0f3F800000; rcp.approx.f32 %f12,%f12;\n"  // t = 1/(1+p|z|)
    " fma.rn.f32 %f13,%f12,0f3F87DC22,0fBFBA00E3;\n"                           // a5 t + a4
    " fma.rn.f32 %f13,%f13,%f12,0f3FB5F0E3;\n"                                 // *t + a3
    " fma.rn.f32 %f13,%f13,%f12,0fBE91A98E;\n"                                 // *t + a2
    " fma.rn.f32 %f13,%f13,%f12,0f3E827906;\n"                                 // *t + a1
    " mul.f32 %f13,%f13,%f12;\n"
    " mul.f32 %f14,%f11,%f11; mul.f32 %f14,%f14,0fBFB8AA3B; ex2.approx.f32 %f14,%f14;\n"  // e^{-z^2}
    " mul.f32 %f13,%f13,%f14; sub.f32 %f13,0f3F800000,%f13;\n"                // erf(|z|)
    " setp.lt.f32 %p4,%f10,0f00000000; @%p4 neg.f32 %f13,%f13;\n"
    " add.f32 %f13,%f13,0f3F800000; mul.f32 %f13,%f13,0f3F000000; mul.f32 %f2,%f2,%f13;\n";

// Block-wide reduction of R across 256 threads (8 warps); result broadcast. Needs %rlane etc.
std::string bred(const std::string& R) {
    std::string s;
    char b[200];
    for (int o = 16; o >= 1; o /= 2) { snprintf(b, sizeof b, " shfl.sync.bfly.b32 %%ft,%s,%d,31,0xffffffff; add.f32 %s,%s,%%ft;\n", R.c_str(), o, R.c_str(), R.c_str()); s += b; }
    s += " @%plane0 st.shared.f32 [%rdw]," + R + ";\n bar.sync 0;\n ld.shared.f32 " + R + ",[%rdr];\n";
    for (int k = 1; k < 8; k++) { snprintf(b, sizeof b, " ld.shared.f32 %%ft,[%%rdr+%d]; add.f32 %s,%s,%%ft;\n", k * 4, R.c_str(), R.c_str()); s += b; }
    s += " bar.sync 0;\n";
    return s;
}
std::string bmax(const std::string& R) {
    std::string s = bred(R);
    for (size_t p; (p = s.find("add.f32")) != std::string::npos;) s.replace(p, 7, "max.f32");
    return s;
}
const char* kRed =
    " .reg .pred %plane0; .reg .f32 %ft; .reg .b64 %rdw,%rdr; .reg .b32 %rlane,%rwarp;\n"
    " .shared .align 4 .f32 red[8];\n"
    " mov.u32 %rlane,%tid.x; shr.u32 %rwarp,%rlane,5; and.b32 %rlane,%rlane,31; setp.eq.u32 %plane0,%rlane,0;\n"
    " mov.u64 %rdr,red; mul.wide.u32 %rdw,%rwarp,4; add.u64 %rdw,%rdr,%rdw;\n";
}  // namespace

// ---------------------------------------------------------------------------------------------
// Int8 prompt matmuls for the chat engine (the llama.cpp "mmq" idea): activations are quantized to int8 per
// 32 values, multiplied against the 4/8-bit weights with int8 tensor cores, and each 32-wide step is scaled
// back to f32. 4-bit weights go in unsigned (0..15); the offset or min is applied with the row sums of x.
// t_qx: X f32 [n*32] -> Q int8 [n*32], D float2 [n] = (scale, scale * sum of q). 8 threads per 32 values.
std::string gen_qx(const char* name) {
    Out o;
    o(".visible .entry %s(.param .u64 pX, .param .u64 pQ, .param .u64 pD, .param .u32 pN)", name);
    o("{");
    o(" .reg .pred %%p<2>; .reg .b32 %%r<24>; .reg .b64 %%rd<8>; .reg .f32 %%f<16>;");
    o(" ld.param.u64 %%rd1,[pX]; ld.param.u64 %%rd2,[pQ]; ld.param.u64 %%rd3,[pD]; ld.param.u32 %%r3,[pN];");
    o(" mov.u32 %%r1,%%ctaid.x; mov.u32 %%r2,%%tid.x; mad.lo.u32 %%r1,%%r1,256,%%r2; shl.b32 %%r3,%%r3,3; setp.lt.u32 %%p1,%%r1,%%r3;");   // p1: in range
    o(" mov.f32 %%f0,0f00000000; mov.f32 %%f1,0f00000000; mov.f32 %%f2,0f00000000; mov.f32 %%f3,0f00000000;");
    o(" mul.wide.u32 %%rd4,%%r1,16; add.u64 %%rd4,%%rd1,%%rd4; @%%p1 ld.global.nc.v4.f32 {%%f0,%%f1,%%f2,%%f3},[%%rd4];");
    o(" abs.f32 %%f4,%%f0; abs.f32 %%f5,%%f1; max.f32 %%f4,%%f4,%%f5; abs.f32 %%f5,%%f2; max.f32 %%f4,%%f4,%%f5; abs.f32 %%f5,%%f3; max.f32 %%f4,%%f4,%%f5;");
    for (int m = 4; m >= 1; m /= 2) o(" shfl.sync.bfly.b32 %%f5,%%f4,%d,31,0xffffffff; max.f32 %%f4,%%f4,%%f5;", m);
    o(" mul.f32 %%f6,%%f4,0f3C010204; setp.gt.f32 %%p0,%%f4,0f00000000; rcp.approx.f32 %%f7,%%f6; selp.f32 %%f7,%%f7,0f00000000,%%p0;");   // d, 1/d
    for (int j = 0; j < 4; j++) o(" mul.f32 %%f8,%%f%d,%%f7; cvt.rni.s32.f32 %%r%d,%%f8;", j, 4 + j);
    o(" add.s32 %%r8,%%r4,%%r5; add.s32 %%r8,%%r8,%%r6; add.s32 %%r8,%%r8,%%r7;");
    for (int m = 4; m >= 1; m /= 2) o(" shfl.sync.bfly.b32 %%r9,%%r8,%d,31,0xffffffff; add.s32 %%r8,%%r8,%%r9;", m);
    o(" bfi.b32 %%r10,%%r5,%%r4,8,8; bfi.b32 %%r10,%%r6,%%r10,16,8; bfi.b32 %%r10,%%r7,%%r10,24,8;");
    o(" mul.wide.u32 %%rd5,%%r1,4; add.u64 %%rd5,%%rd2,%%rd5; @%%p1 st.global.b32 [%%rd5],%%r10;");
    o(" and.b32 %%r11,%%r1,7; setp.eq.and.u32 %%p0,%%r11,0,%%p1;");
    o(" cvt.rn.f32.s32 %%f9,%%r8; mul.f32 %%f9,%%f9,%%f6; shr.u32 %%r12,%%r1,3;");
    o(" mul.wide.u32 %%rd6,%%r12,8; add.u64 %%rd6,%%rd3,%%rd6; @%%p0 st.global.v2.f32 [%%rd6],{%%f6,%%f9};");
    o(" ret;\n}");
    return o.s;
}

// t_mmq*: Y[M][N] = X[M][K] W^T. XQ int8 [M][K], XD float2 [M][K/32]; W rows in the chat GPU layout:
// kind 0 = 4-bit (q-8)*d, kind 1 = int8 q*d, kind 2 = 4-bit q*d - m.
// A block computes (WM*FM*16 tokens) x (WN*FN*8 weight rows) with WM x WN warps; big tiles
// so every byte loaded into shared memory is reused many times (llama.cpp uses 128 x 128 on Ampere).
// 128-wide K steps; the next step is loaded into registers while the current one computes.
// grouped: mixture-of-experts form. Grid z = expert e; its tokens are rows off[e]..off[e+1] of X and Y, its
// weights sit e * stride bytes further on (quants pQs, scales pSs).
std::string gen_mmq(const char* name, int kind, const int* tile, bool grouped = false) {   // tile: warps M, warps N, fragments M, fragments N
    Out o;
    const bool q4 = kind == 0 || kind == 2, q6 = kind == 3, pair = kind == 2 || kind == 3;
    const int WM = tile[0], WN = tile[1], FM = tile[2], FN = tile[3], T = 32 * WM * WN, BM = 16 * FM * WM, BN = 8 * FN * WN, BK = 128, NB = BK / 32;
    const int SA = BK + 16, RB = q4 ? BK / 2 : q6 ? BK / 4 * 3 : BK, SB = RB + 16;   // shared row strides (bytes), padded against bank conflicts
    const int CA = BK / 16, CB = RB / 16;                          // 16-byte chunks per row
    const int EA = BM * CA / T, EB = (BN * CB + T - 1) / T;        // chunks per thread (weights: the last may be partial)
    const int WB = pair ? 8 : 4, SSA = NB * 8 + 8, SSB = NB * WB + WB;   // scale tiles (f32): (d, d*sum) per token row; d or (d, m) per weight row
    const int WG = WB / 2;                                         // weight scales in VRAM are f16
    const int ESA = BM * NB / T, ESB = BN * NB / T;                // scale entries per thread
    const int NC = FM * FN * 4, NR = FM * 2, NW = FN * 2;          // f32 sums, token rows and weight columns per thread
    o(".visible .entry %s(.param .u64 pX, .param .u64 pXD, .param .u64 pW, .param .u64 pS, .param .u64 pY, .param .u32 pM, .param .u32 pN, .param .u32 pK%s)", name,
      grouped ? ", .param .u64 pOff, .param .u64 pQs, .param .u64 pSs" : "");
    if (kMmqRegs) o(".maxnreg %d", kMmqRegs);
    o("{");
    o(" .reg .pred %%p<8>; .reg .pred %%va<%d>, %%vb<%d>, %%vs<%d>; .reg .b32 %%cb<%d>; .reg .b64 %%sb<%d>;", EA, EB, EB, EB, EB);
    o(" .reg .b32 %%r<80>; .reg .b64 %%rd<24>; .reg .b64 %%ga<%d>; .reg .b64 %%gb<%d>; .reg .b64 %%gsa<%d>; .reg .b64 %%gsb<%d>; .reg .b64 %%ssa<%d>; .reg .b64 %%ssb<%d>;", EA, EB, ESA, ESB, ESA, ESB);
    o(" .reg .b32 %%ja<%d>; .reg .b32 %%jb<%d>;", ESA, ESB);
    o(" .reg .f32 %%c<%d>; .reg .b32 %%i<%d>; .reg .b32 %%a<4>; .reg .b32 %%b<%d>; .reg .b32 %%x<%d>;", NC, FN * 4 * (q6 ? 2 : 1), FN * 2, (EA + EB) * 4);
    o(" .reg .f32 %%dx<%d>; .reg .f32 %%xs<%d>; .reg .f32 %%dw<%d>; .reg .f32 %%mw<%d>; .reg .f32 %%f<4>;", NR, NR, NW, NW);
    o(" .reg .f32 %%ya<%d>; .reg .f32 %%yb<%d>; .reg .b16 %%h<2>;", ESA * 2, ESB * 2);
    o(" .shared .align 16 .b8 sA[%d]; .shared .align 16 .b8 sB[%d]; .shared .align 16 .b8 sSA[%d]; .shared .align 16 .b8 sSB[%d];", BM * SA, BN * SB, BM * SSA, BN * SSB);
    o(" ld.param.u64 %%rd1,[pX]; ld.param.u64 %%rd2,[pXD]; ld.param.u64 %%rd3,[pW]; ld.param.u64 %%rd4,[pS]; ld.param.u64 %%rd5,[pY];");
    o(" ld.param.u32 %%r1,[pM]; ld.param.u32 %%r2,[pN]; ld.param.u32 %%r3,[pK]; shr.u32 %%r4,%%r3,5; shr.u32 %%r25,%%r3,%d;", q4 ? 1 : q6 ? 2 : 0);   // r4 = K/32, r25 = weight row bytes
    if (q6) o(" mul.lo.u32 %%r25,%%r25,3;");   // 6 bits per weight
    if (grouped) {
        o(" mov.u32 %%r7,%%ctaid.z; ld.param.u64 %%rd19,[pOff]; mul.wide.u32 %%rd20,%%r7,4; add.u64 %%rd19,%%rd19,%%rd20;");
        o(" ld.global.u32 %%r8,[%%rd19]; ld.global.u32 %%r9,[%%rd19+4]; sub.u32 %%r1,%%r9,%%r8;");   // this expert's rows
        o(" mul.wide.u32 %%rd20,%%r8,%%r3; add.u64 %%rd1,%%rd1,%%rd20; mul.wide.u32 %%rd20,%%r8,%%r4; shl.b64 %%rd20,%%rd20,3; add.u64 %%rd2,%%rd2,%%rd20;");
        o(" mul.wide.u32 %%rd20,%%r8,%%r2; shl.b64 %%rd20,%%rd20,2; add.u64 %%rd5,%%rd5,%%rd20; cvt.u64.u32 %%rd21,%%r7;");
        o(" ld.param.u64 %%rd20,[pQs]; mul.lo.u64 %%rd20,%%rd20,%%rd21; add.u64 %%rd3,%%rd3,%%rd20;");
        o(" ld.param.u64 %%rd20,[pSs]; mul.lo.u64 %%rd20,%%rd20,%%rd21; add.u64 %%rd4,%%rd4,%%rd20;");
    }
    o(" sub.u32 %%r5,%%r1,1; sub.u32 %%r6,%%r2,1;");
    o(" mov.u32 %%r12,%%tid.x; and.b32 %%r13,%%r12,31; shr.u32 %%r14,%%r12,5; shr.u32 %%r60,%%r13,2; and.b32 %%r61,%%r13,3;");   // tid lane warp g t
    o(" div.u32 %%r62,%%r14,%d; mul.lo.u32 %%r62,%%r62,%d; rem.u32 %%r63,%%r14,%d; mul.lo.u32 %%r63,%%r63,%d;", WN, FM * 16, WN, FN * 8);   // wm, wn
    o(" mov.u32 %%r15,%%ctaid.x; mul.lo.u32 %%r15,%%r15,%d; mov.u32 %%r16,%%ctaid.y; mul.lo.u32 %%r16,%%r16,%d;", BN, BM);      // n0, m0
    if (grouped) o(" setp.ge.u32 %%p0,%%r16,%%r1; @%%p0 ret;");   // past this expert's tokens
    o(" mov.u32 %%r0,0; mov.u64 %%rd7,sA; mov.u64 %%rd8,sB;");
    // loaders: chunk tid + T*e -> row (tid / C) + e*T/C, byte (tid % C)*16
    o(" div.u32 %%r18,%%r12,%d; rem.u32 %%r19,%%r12,%d; shl.b32 %%r19,%%r19,4;", CA, CA);
    for (int e = 0; e < EA; e++) {
        o(" add.u32 %%r22,%%r18,%d; add.u32 %%r22,%%r22,%%r16; setp.lt.u32 %%va%d,%%r22,%%r1; min.u32 %%r22,%%r22,%%r5;", e * T / CA, e);
        o(" mad.lo.u32 %%r23,%%r22,%%r3,%%r19; cvt.u64.u32 %%ga%d,%%r23; add.u64 %%ga%d,%%rd1,%%ga%d;", e, e, e);
    }
    o(" mad.lo.u32 %%r24,%%r18,%d,%%r19; cvt.u64.u32 %%rd9,%%r24; add.u64 %%rd9,%%rd7,%%rd9;", SA);
    for (int e = 0; e < EB; e++) {   // weight chunk tid + T*e: row (/ CB), byte (% CB)*16; vs = the chunk exists
        o(" add.u32 %%r20,%%r12,%d; setp.lt.u32 %%vs%d,%%r20,%d; div.u32 %%r22,%%r20,%d; rem.u32 %%cb%d,%%r20,%d; shl.b32 %%cb%d,%%cb%d,4;", e * T, e, BN * CB, CB, e, CB, e, e);
        o(" mad.lo.u32 %%r24,%%r22,%d,%%cb%d; cvt.u64.u32 %%sb%d,%%r24; add.u64 %%sb%d,%%rd8,%%sb%d;", SB, e, e, e, e);
        o(" add.u32 %%r22,%%r22,%%r15; setp.lt.u32 %%vb%d,%%r22,%%r2; and.pred %%vb%d,%%vb%d,%%vs%d; min.u32 %%r22,%%r22,%%r6;", e, e, e, e);
        o(" mad.lo.u32 %%r23,%%r22,%%r25,%%cb%d; cvt.u64.u32 %%gb%d,%%r23; add.u64 %%gb%d,%%rd3,%%gb%d;", e, e, e, e);
    }
    // fragment read bases: A row wm+g, byte 4t; B row wn+g, byte 4t
    o(" shl.b32 %%r27,%%r61,2;");
    o(" add.u32 %%r26,%%r62,%%r60; mad.lo.u32 %%r26,%%r26,%d,%%r27; cvt.u64.u32 %%rd12,%%r26; add.u64 %%rd12,%%rd7,%%rd12;", SA);
    o(" add.u32 %%r28,%%r63,%%r60; mad.lo.u32 %%r28,%%r28,%d,%%r27; cvt.u64.u32 %%rd13,%%r28; add.u64 %%rd13,%%rd8,%%rd13;", SB);
    if (q6) {   // qh word of this lane: row wn+g, byte (t>>1)*4; its 2-bit fields start at bit (t&1)*4 (low half) and +2 (high half)
        o(" shr.u32 %%r70,%%r61,1; shl.b32 %%r70,%%r70,2; add.u32 %%r72,%%r63,%%r60; mad.lo.u32 %%r72,%%r72,%d,%%r70; cvt.u64.u32 %%rd22,%%r72; add.u64 %%rd22,%%rd8,%%rd22;", SB);
        o(" and.b32 %%r70,%%r61,1; shl.b32 %%r70,%%r70,2; add.u32 %%r71,%%r70,2;");
    }
    // scale tiles: entry tid + T*i -> row / NB, block j = % NB. Global rows clamped to the matrix.
    for (int i = 0; i < ESA; i++) {
        o(" add.u32 %%r30,%%r12,%d; div.u32 %%r31,%%r30,%d; rem.u32 %%ja%d,%%r30,%d;", i * T, NB, i, NB);
        o(" add.u32 %%r29,%%r31,%%r16; min.u32 %%r29,%%r29,%%r5; mad.lo.u32 %%r29,%%r29,%%r4,%%ja%d; mul.wide.u32 %%gsa%d,%%r29,8; add.u64 %%gsa%d,%%rd2,%%gsa%d;", i, i, i, i);
        o(" mov.u64 %%ssa%d,sSA; mul.lo.u32 %%r29,%%r31,%d; mad.lo.u32 %%r37,%%ja%d,8,%%r29; cvt.u64.u32 %%rd14,%%r37; add.u64 %%ssa%d,%%ssa%d,%%rd14;", i, SSA, i, i, i);
    }
    for (int i = 0; i < ESB; i++) {
        o(" add.u32 %%r30,%%r12,%d; div.u32 %%r31,%%r30,%d; rem.u32 %%jb%d,%%r30,%d;", i * T, NB, i, NB);
        o(" add.u32 %%r29,%%r31,%%r15; min.u32 %%r29,%%r29,%%r6; mad.lo.u32 %%r29,%%r29,%%r4,%%jb%d; mul.wide.u32 %%gsb%d,%%r29,%d; add.u64 %%gsb%d,%%rd4,%%gsb%d;", i, i, WG, i, i);
        o(" mov.u64 %%ssb%d,sSB; mul.lo.u32 %%r29,%%r31,%d; mad.lo.u32 %%r37,%%jb%d,%d,%%r29; cvt.u64.u32 %%rd14,%%r37; add.u64 %%ssb%d,%%ssb%d,%%rd14;", i, SSB, i, WB, i, i);
    }
    // compute-side scale bases: A row wm+g, B row wn+2t
    o(" mov.u64 %%rd15,sSA; add.u32 %%r29,%%r62,%%r60; mul.lo.u32 %%r29,%%r29,%d; cvt.u64.u32 %%rd14,%%r29; add.u64 %%rd15,%%rd15,%%rd14;", SSA);
    o(" mov.u64 %%rd10,sSB; shl.b32 %%r29,%%r61,1; add.u32 %%r29,%%r29,%%r63; mul.lo.u32 %%r29,%%r29,%d; cvt.u64.u32 %%rd14,%%r29; add.u64 %%rd10,%%rd10,%%rd14;", SSB);
    for (int i = 0; i < NC; i++) o(" mov.f32 %%c%d,0f00000000;", i);
    auto gload = [&](const char* k0) {
        o(" cvt.u64.u32 %%rd16,%s;", k0);
        for (int e = 0; e < EA; e++) {
            o(" add.u32 %%r40,%s,%%r19; setp.lt.u32 %%p2,%%r40,%%r3; and.pred %%p2,%%p2,%%va%d; add.u64 %%rd17,%%ga%d,%%rd16;", k0, e, e);
            o(" mov.b32 %%x%d,0; mov.b32 %%x%d,0; mov.b32 %%x%d,0; mov.b32 %%x%d,0;", e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
            o(" @%%p2 ld.global.nc.v4.b32 {%%x%d,%%x%d,%%x%d,%%x%d},[%%rd17];", e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
        }
        o(" shr.u32 %%r42,%s,5;", k0);   // first 32-block of this K step
        for (int i = 0; i < ESA; i++) {
            o(" add.u32 %%r43,%%r42,%%ja%d; setp.lt.u32 %%p2,%%r43,%%r4; mul.wide.u32 %%rd17,%%r42,8; add.u64 %%rd17,%%gsa%d,%%rd17;", i, i);
            o(" mov.f32 %%ya%d,0f00000000; mov.f32 %%ya%d,0f00000000; @%%p2 ld.global.nc.v2.f32 {%%ya%d,%%ya%d},[%%rd17];", i * 2, i * 2 + 1, i * 2, i * 2 + 1);
        }
        for (int i = 0; i < ESB; i++) {
            o(" add.u32 %%r43,%%r42,%%jb%d; setp.lt.u32 %%p2,%%r43,%%r4; mul.wide.u32 %%rd17,%%r42,%d; add.u64 %%rd17,%%gsb%d,%%rd17;", i, WG, i);
            o(" mov.b16 %%h0,0; mov.b16 %%h1,0;");
            if (pair) o(" @%%p2 ld.global.nc.v2.b16 {%%h0,%%h1},[%%rd17];");
            else o(" @%%p2 ld.global.nc.b16 %%h0,[%%rd17];");
            o(" cvt.f32.f16 %%yb%d,%%h0; cvt.f32.f16 %%yb%d,%%h1;", i * 2, i * 2 + 1);
        }
        o(" shr.u32 %%r41,%s,%d;", k0, q4 ? 1 : q6 ? 2 : 0);   // byte offset of this K step in a weight row
        if (q6) o(" mul.lo.u32 %%r41,%%r41,3;");
        o(" cvt.u64.u32 %%rd16,%%r41;");
        for (int e = 0; e < EB; e++) {
            int x = (EA + e) * 4;
            o(" add.u32 %%r40,%%r41,%%cb%d; setp.lt.u32 %%p2,%%r40,%%r25; and.pred %%p2,%%p2,%%vb%d; add.u64 %%rd17,%%gb%d,%%rd16;", e, e, e);
            o(" mov.b32 %%x%d,0; mov.b32 %%x%d,0; mov.b32 %%x%d,0; mov.b32 %%x%d,0;", x, x + 1, x + 2, x + 3);
            o(" @%%p2 ld.global.nc.v4.b32 {%%x%d,%%x%d,%%x%d,%%x%d},[%%rd17];", x, x + 1, x + 2, x + 3);
        }
    };
    auto sstore = [&]() {
        for (int e = 0; e < EA; e++) o(" st.shared.v4.b32 [%%rd9+%d],{%%x%d,%%x%d,%%x%d,%%x%d};", e * (T / CA) * SA, e * 4, e * 4 + 1, e * 4 + 2, e * 4 + 3);
        for (int e = 0; e < EB; e++) { int x = (EA + e) * 4; o(" @%%vs%d st.shared.v4.b32 [%%sb%d],{%%x%d,%%x%d,%%x%d,%%x%d};", e, e, x, x + 1, x + 2, x + 3); }
        for (int i = 0; i < ESA; i++) o(" st.shared.v2.f32 [%%ssa%d],{%%ya%d,%%ya%d};", i, i * 2, i * 2 + 1);
        for (int i = 0; i < ESB; i++) {
            if (pair) o(" st.shared.v2.f32 [%%ssb%d],{%%yb%d,%%yb%d};", i, i * 2, i * 2 + 1);
            else o(" st.shared.f32 [%%ssb%d],%%yb%d;", i, i * 2);
        }
    };
    o(" mov.u32 %%r32,0;");
    gload("%r32");
    sstore();
    o(" bar.sync 0;");
    o("LK:");
    o(" add.u32 %%r33,%%r32,%d; setp.lt.u32 %%p1,%%r33,%%r3;", BK);
    o(" @!%%p1 bra NOPF;");
    gload("%r33");
    o("NOPF:");
    for (int b = 0; b < NB; b++) {
        o(" add.u32 %%r34,%%r32,%d; setp.ge.u32 %%p3,%%r34,%%r3; @%%p3 bra SKIP%d;", b * 32, b);
        // weight fragments (all FN), unpacked once per block
        for (int nt = 0; nt < FN; nt++) {
            if (q4) {
                o(" ld.shared.b32 %%r36,[%%rd13+%d]; and.b32 %%b%d,%%r36,0x0F0F0F0F; shr.u32 %%r36,%%r36,4; and.b32 %%b%d,%%r36,0x0F0F0F0F;", nt * 8 * SB + b * 16, nt * 2, nt * 2 + 1);
                if (kind == 0)   // q - 8 per byte without borrows: (q + 0x78) ^ 0x80
                    for (int h = 0; h < 2; h++) o(" add.u32 %%b%d,%%b%d,0x78787878; xor.b32 %%b%d,%%b%d,0x80808080;", nt * 2 + h, nt * 2 + h, nt * 2 + h, nt * 2 + h);
            } else if (q6) {   // 48-byte units of two blocks: ql (low 4 bits) at u*48 + half*16, qh (top 2) at u*48 + 32 + half*8
                const int u = b / 2, half = b % 2, ql = nt * 8 * SB + u * 48 + half * 16, qh = nt * 8 * SB + u * 48 + 32 + half * 8;
                o(" ld.shared.b32 %%r36,[%%rd13+%d]; ld.shared.b32 %%r37,[%%rd22+%d];", ql, qh);
                o(" and.b32 %%r73,%%r36,0x0F0F0F0F; shr.u32 %%r74,%%r37,%%r70; and.b32 %%r74,%%r74,0x03030303; shl.b32 %%r74,%%r74,4; or.b32 %%b%d,%%r73,%%r74;", nt * 2);
                o(" shr.u32 %%r73,%%r36,4; and.b32 %%r73,%%r73,0x0F0F0F0F; shr.u32 %%r74,%%r37,%%r71; and.b32 %%r74,%%r74,0x03030303; shl.b32 %%r74,%%r74,4; or.b32 %%b%d,%%r73,%%r74;", nt * 2 + 1);
                for (int h = 0; h < 2; h++)   // q - 32 per byte: (q + 0x60) ^ 0x80
                    o(" add.u32 %%b%d,%%b%d,0x60606060; xor.b32 %%b%d,%%b%d,0x80808080;", nt * 2 + h, nt * 2 + h, nt * 2 + h, nt * 2 + h);
            } else
                o(" ld.shared.b32 %%b%d,[%%rd13+%d]; ld.shared.b32 %%b%d,[%%rd13+%d];", nt * 2, nt * 8 * SB + b * 32, nt * 2 + 1, nt * 8 * SB + b * 32 + 16);
        }
        for (int k = 0; k < NR; k++) o(" ld.shared.v2.f32 {%%dx%d,%%xs%d},[%%rd15+%d];", k, k, ((k / 2) * 16 + (k % 2) * 8) * SSA + b * 8);
        for (int k = 0; k < NW; k++) {
            int off = ((k / 2) * 8 + k % 2) * SSB + b * WB;
            if (kind == 2) o(" ld.shared.v2.f32 {%%dw%d,%%mw%d},[%%rd10+%d]; neg.f32 %%mw%d,%%mw%d;", k, k, off, k, k);
            else if (q6) o(" ld.shared.v2.f32 {%%dw%d,%%mw%d},[%%rd10+%d];", k, k, off);   // scales of elements 0..15 and 16..31
            else o(" ld.shared.f32 %%dw%d,[%%rd10+%d];", k, off);
        }
        // one 16-row token fragment at a time: FN int8 mma, then scale into the f32 sums
        for (int mt = 0; mt < FM; mt++) {
            int base = mt * 16 * SA + b * 32;
            o(" ld.shared.b32 %%a0,[%%rd12+%d]; ld.shared.b32 %%a1,[%%rd12+%d]; ld.shared.b32 %%a2,[%%rd12+%d]; ld.shared.b32 %%a3,[%%rd12+%d];",
              base, base + 8 * SA, base + 16, base + 8 * SA + 16);
            if (q6) {   // the two 16-element halves have their own scales: two k16 mma, sums in i[0..] and i[FN*4..]
                for (int nt = 0; nt < FN; nt++) {
                    const int lo = nt * 4, hi = FN * 4 + nt * 4;
                    o(" mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%%i%d,%%i%d,%%i%d,%%i%d},{%%a0,%%a1},{%%b%d},{%%r0,%%r0,%%r0,%%r0};", lo, lo + 1, lo + 2, lo + 3, nt * 2);
                    o(" mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%%i%d,%%i%d,%%i%d,%%i%d},{%%a2,%%a3},{%%b%d},{%%r0,%%r0,%%r0,%%r0};", hi, hi + 1, hi + 2, hi + 3, nt * 2 + 1);
                }
                for (int nt = 0; nt < FN; nt++)
                    for (int h = 0; h < 2; h++)
                        for (int j = 0; j < 2; j++) {
                            int ii = nt * 4 + h * 2 + j, ci = (mt * FN + nt) * 4 + h * 2 + j, r = mt * 2 + h, w = nt * 2 + j;
                            o(" cvt.rn.f32.s32 %%f0,%%i%d; cvt.rn.f32.s32 %%f1,%%i%d; mul.f32 %%f0,%%f0,%%dw%d; fma.rn.f32 %%f0,%%f1,%%mw%d,%%f0; fma.rn.f32 %%c%d,%%f0,%%dx%d,%%c%d;",
                              ii, FN * 4 + ii, w, w, ci, r, ci);
                        }
                continue;
            }
            for (int nt = 0; nt < FN; nt++)
                o(" mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%%i%d,%%i%d,%%i%d,%%i%d},{%%a0,%%a1,%%a2,%%a3},{%%b%d,%%b%d},{%%r0,%%r0,%%r0,%%r0};",
                  nt * 4, nt * 4 + 1, nt * 4 + 2, nt * 4 + 3, nt * 2, nt * 2 + 1);
            for (int nt = 0; nt < FN; nt++)
                for (int h = 0; h < 2; h++)
                    for (int j = 0; j < 2; j++) {
                        int ii = nt * 4 + h * 2 + j, ci = (mt * FN + nt) * 4 + h * 2 + j, r = mt * 2 + h, w = nt * 2 + j;
                        o(" cvt.rn.f32.s32 %%f0,%%i%d; mul.f32 %%f0,%%f0,%%dx%d; fma.rn.f32 %%c%d,%%f0,%%dw%d,%%c%d;", ii, r, ci, w, ci);
                        if (kind == 2) o(" fma.rn.f32 %%c%d,%%xs%d,%%mw%d,%%c%d;", ci, r, w, ci);
                    }
        }
        o("SKIP%d:", b);
    }
    o(" bar.sync 0;");
    o(" @!%%p1 bra EK;");
    sstore();
    o(" bar.sync 0;");
    o(" mov.u32 %%r32,%%r33; bra LK;");
    o("EK:");
    for (int mt = 0; mt < FM; mt++)
        for (int nt = 0; nt < FN; nt++)
            for (int h = 0; h < 2; h++) {
                int ci = (mt * FN + nt) * 4 + h * 2;
                o(" add.u32 %%r52,%%r16,%%r62; add.u32 %%r52,%%r52,%%r60; add.u32 %%r52,%%r52,%d;", mt * 16 + h * 8);
                o(" add.u32 %%r53,%%r15,%%r63; shl.b32 %%r54,%%r61,1; add.u32 %%r53,%%r53,%%r54; add.u32 %%r53,%%r53,%d;", nt * 8);
                o(" setp.lt.u32 %%p4,%%r52,%%r1; setp.lt.u32 %%p5,%%r53,%%r2; and.pred %%p5,%%p5,%%p4; add.u32 %%r55,%%r53,1; setp.lt.u32 %%p6,%%r55,%%r2; and.pred %%p6,%%p6,%%p4;");
                o(" mad.lo.u32 %%r56,%%r52,%%r2,%%r53; mul.wide.u32 %%rd18,%%r56,4; add.u64 %%rd18,%%rd5,%%rd18;");
                o(" @%%p5 st.global.f32 [%%rd18],%%c%d; @%%p6 st.global.f32 [%%rd18+4],%%c%d;", ci, ci + 1);
            }
    o(" ret;\n}");
    return o.s;
}

// Tensor-core GEMMs live in their own module (needs sm_80+); older GPUs keep the SIMT versions.
std::string gx_ptx_tc() {
    return ".version 7.0\n.target sm_80\n.address_size 64\n" + gen_gemm_tc("t_lin", 0) + gen_gemm_tc("t_nt", 1) + gen_gemm_tc("t_nn", 2) + gen_gemm_tc("t_conv", 3) + gen_lin_fast("t_linf", false) + gen_lin_fast("t_linq8", true) + gen_lin_fast("t_linh", false, true) + gen_lin_fast("t_linq8h", true, true) + gen_qx("t_qx") +
           gen_mmq("t_mmq0", 0, kMmqTiles[0]) + gen_mmq("t_mmq1", 1, kMmqTiles[0]) + gen_mmq("t_mmq2", 2, kMmqTiles[0]) +
           gen_mmq("t_mmqs0", 0, kMmqTiles[1]) + gen_mmq("t_mmqs1", 1, kMmqTiles[1]) + gen_mmq("t_mmqs2", 2, kMmqTiles[1]) +
           gen_mmq("t_mmqg0", 0, kMmqTiles[0], true) + gen_mmq("t_mmqg1", 1, kMmqTiles[0], true) + gen_mmq("t_mmqg2", 2, kMmqTiles[0], true) +
           gen_mmq("t_mmq3", 3, kMmqTiles[0]) + gen_mmq("t_mmqg3", 3, kMmqTiles[0], true) +   // 6-bit: big tile only
           gen_flash("t_fa64", 64) + gen_flash("t_fa128", 128) + gen_flash("t_fa72", 72) +
           gen_flash_kv("t_fkv64", 64, 64) + gen_flash_kv("t_fkv80", 80, 80) + gen_flash_kv("t_fkv96", 96, 96) + gen_flash_kv("t_fkv128", 128, 128) + gen_flash_kv("t_fkv256", 256, 128) +
           gen_flash_kv("t_fkq64", 64, 64, true) + gen_flash_kv("t_fkq80", 80, 80, true) + gen_flash_kv("t_fkq96", 96, 96, true) + gen_flash_kv("t_fkq128", 128, 128, true) + gen_flash_kv("t_fkq256", 256, 128, true);
}

std::string gx_ptx() {
    std::string s = ".version 6.0\n.target sm_61\n.address_size 64\n";
    s += gen_gemm("g_lin", 0) + gen_gemm("g_nt", 1) + gen_gemm("g_nn", 2) + gen_gemm("g_conv", 3);

    // Group norm over NCHW: grid = groups. y = (x-mean)/sqrt(var+eps)*g[c]+b[c], optional SiLU.
    s += ".visible .entry k_gn(.param .u64 pX, .param .u64 pG, .param .u64 pB, .param .u64 pY, .param .u32 pCpg, .param .u32 pHW, .param .f32 pE, .param .u32 pAct)\n{\n"
         " .reg .pred %p<6>; .reg .b32 %r<24>; .reg .f32 %f<24>; .reg .b64 %rd<16>;\n";
    s += kRed;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pG]; ld.param.u64 %rd3,[pB]; ld.param.u64 %rd4,[pY]; ld.param.u32 %r1,[pCpg]; ld.param.u32 %r2,[pHW]; ld.param.f32 %f20,[pE]; ld.param.u32 %r3,[pAct];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x; mul.lo.u32 %r6,%r1,%r2;\n"                 // count
         " mul.wide.u32 %rd5,%r4,%r6; shl.b64 %rd5,%rd5,2; add.u64 %rd6,%rd1,%rd5; add.u64 %rd7,%rd4,%rd5;\n"
         " mov.f32 %f1,0f00000000; mov.u32 %r7,%r5;\n"
         "A1: setp.ge.u32 %p1,%r7,%r6; @%p1 bra A2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f2,[%rd8]; add.f32 %f1,%f1,%f2; add.u32 %r7,%r7,256; bra A1;\nA2:\n";
    s += bred("%f1");
    s += " cvt.rn.f32.u32 %f3,%r6; div.rn.f32 %f4,%f1,%f3;\n"                                   // mean
         " mov.f32 %f1,0f00000000; mov.u32 %r7,%r5;\n"
         "B1: setp.ge.u32 %p1,%r7,%r6; @%p1 bra B2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f2,[%rd8]; sub.f32 %f2,%f2,%f4; fma.rn.f32 %f1,%f2,%f2,%f1; add.u32 %r7,%r7,256; bra B1;\nB2:\n";
    s += bred("%f1");
    s += " div.rn.f32 %f5,%f1,%f3; add.f32 %f5,%f5,%f20; rsqrt.approx.f32 %f5,%f5;\n"
         " mul.lo.u32 %r8,%r4,%r1; setp.ne.u32 %p3,%r3,0; mov.u32 %r7,%r5;\n"
         "C1: setp.ge.u32 %p1,%r7,%r6; @%p1 bra C2;\n"
         " mul.wide.u32 %rd8,%r7,4; add.u64 %rd9,%rd6,%rd8; ld.global.f32 %f2,[%rd9];\n"
         " div.u32 %r9,%r7,%r2; add.u32 %r9,%r9,%r8; mul.wide.u32 %rd10,%r9,4; add.u64 %rd11,%rd2,%rd10; ld.global.f32 %f6,[%rd11]; add.u64 %rd12,%rd3,%rd10; ld.global.f32 %f7,[%rd12];\n"
         " sub.f32 %f2,%f2,%f4; mul.f32 %f2,%f2,%f5; fma.rn.f32 %f2,%f2,%f6,%f7;\n"
         " @%p3 mul.f32 %f8,%f2,0fBFB8AA3B; @%p3 ex2.approx.f32 %f8,%f8; @%p3 add.f32 %f8,%f8,0f3F800000; @%p3 div.approx.f32 %f2,%f2,%f8;\n"
         " add.u64 %rd13,%rd7,%rd8; st.global.f32 [%rd13],%f2; add.u32 %r7,%r7,256; bra C1;\nC2: ret;\n}\n";

    // Layer norm over rows of length D: grid = rows. In-place safe.
    s += ".visible .entry k_ln(.param .u64 pX, .param .u64 pG, .param .u64 pB, .param .u64 pY, .param .u32 pD, .param .f32 pE)\n{\n"
         " .reg .pred %p<6>; .reg .b32 %r<24>; .reg .f32 %f<24>; .reg .b64 %rd<16>;\n";
    s += kRed;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pG]; ld.param.u64 %rd3,[pB]; ld.param.u64 %rd4,[pY]; ld.param.u32 %r1,[pD]; ld.param.f32 %f20,[pE];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x;\n"
         " mul.wide.u32 %rd5,%r4,%r1; shl.b64 %rd5,%rd5,2; add.u64 %rd6,%rd1,%rd5; add.u64 %rd7,%rd4,%rd5;\n"
         " mov.f32 %f1,0f00000000; mov.u32 %r7,%r5;\n"
         "A1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra A2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f2,[%rd8]; add.f32 %f1,%f1,%f2; add.u32 %r7,%r7,256; bra A1;\nA2:\n";
    s += bred("%f1");
    s += " cvt.rn.f32.u32 %f3,%r1; div.rn.f32 %f4,%f1,%f3; mov.f32 %f1,0f00000000; mov.u32 %r7,%r5;\n"
         "B1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra B2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f2,[%rd8]; sub.f32 %f2,%f2,%f4; fma.rn.f32 %f1,%f2,%f2,%f1; add.u32 %r7,%r7,256; bra B1;\nB2:\n";
    s += bred("%f1");
    s += " div.rn.f32 %f5,%f1,%f3; add.f32 %f5,%f5,%f20; rsqrt.approx.f32 %f5,%f5; mov.u32 %r7,%r5;\n"
         "C1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra C2;\n"
         " mul.wide.u32 %rd8,%r7,4; add.u64 %rd9,%rd6,%rd8; ld.global.f32 %f2,[%rd9]; add.u64 %rd10,%rd2,%rd8; ld.global.f32 %f6,[%rd10]; add.u64 %rd11,%rd3,%rd8; ld.global.f32 %f7,[%rd11];\n"
         " sub.f32 %f2,%f2,%f4; mul.f32 %f2,%f2,%f5; fma.rn.f32 %f2,%f2,%f6,%f7; add.u64 %rd12,%rd7,%rd8; st.global.f32 [%rd12],%f2; add.u32 %r7,%r7,256; bra C1;\nC2: ret;\n}\n";

    // Row softmax of x*scale, in place. grid = rows. Causal: column j masked when j > (row % T) + off.
    s += ".visible .entry k_softmax(.param .u64 pX, .param .u32 pCols, .param .f32 pScale, .param .u32 pCausal, .param .u32 pT)\n{\n"
         " .reg .pred %p<6>; .reg .b32 %r<24>; .reg .f32 %f<24>; .reg .b64 %rd<16>;\n";
    s += kRed;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u32 %r1,[pCols]; ld.param.f32 %f20,[pScale]; ld.param.u32 %r2,[pCausal]; ld.param.u32 %r3,[pT];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x; mul.wide.u32 %rd2,%r4,%r1; shl.b64 %rd2,%rd2,2; add.u64 %rd2,%rd1,%rd2;\n"
         " mov.u32 %r6,%r1; setp.ne.u32 %p3,%r2,0; @%p3 rem.u32 %r6,%r4,%r3; @%p3 add.u32 %r6,%r6,1; min.u32 %r6,%r6,%r1;\n"   // active columns
         " mov.f32 %f1,0fF149F2CA; mov.u32 %r7,%r5;\n"
         "A1: setp.ge.u32 %p1,%r7,%r6; @%p1 bra A2; mul.wide.u32 %rd3,%r7,4; add.u64 %rd3,%rd2,%rd3; ld.global.f32 %f2,[%rd3]; mul.f32 %f2,%f2,%f20; max.f32 %f1,%f1,%f2; add.u32 %r7,%r7,256; bra A1;\nA2:\n";
    s += bmax("%f1");
    s += " mov.f32 %f4,0f00000000; mov.u32 %r7,%r5;\n"
         "B1: setp.ge.u32 %p1,%r7,%r6; @%p1 bra B2; mul.wide.u32 %rd3,%r7,4; add.u64 %rd3,%rd2,%rd3; ld.global.f32 %f2,[%rd3]; mul.f32 %f2,%f2,%f20; sub.f32 %f2,%f2,%f1; mul.f32 %f2,%f2,0f3FB8AA3B; ex2.approx.f32 %f2,%f2; st.global.f32 [%rd3],%f2; add.f32 %f4,%f4,%f2; add.u32 %r7,%r7,256; bra B1;\nB2:\n";
    s += bred("%f4");
    s += " rcp.approx.f32 %f5,%f4; mov.u32 %r7,%r5;\n"
         "C1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra C2; mul.wide.u32 %rd3,%r7,4; add.u64 %rd3,%rd2,%rd3;\n"
         " setp.lt.u32 %p2,%r7,%r6; mov.f32 %f2,0f00000000; @%p2 ld.global.f32 %f2,[%rd3]; mul.f32 %f2,%f2,%f5; st.global.f32 [%rd3],%f2; add.u32 %r7,%r7,256; bra C1;\nC2: ret;\n}\n";

    // Activations in place. mode 0 silu, 1 gelu(erf), 2 quick_gelu, 3 gelu(tanh)
    s += ew("k_act", ".param .u64 pX, .param .u32 pMode", " ld.param.u64 %rd1,[pX]; ld.param.u32 %r10,[pMode];\n",
            (std::string(" mul.wide.u32 %rd2,%r5,4; add.u64 %rd2,%rd1,%rd2; ld.global.f32 %f2,[%rd2];\n"
                         " setp.eq.u32 %p2,%r10,1; @%p2 bra GE;\n"
                         " setp.eq.u32 %p3,%r10,4; @%p3 bra MI;\n"
                         " mov.f32 %f3,0f3F800000; setp.eq.u32 %p3,%r10,2; @%p3 mov.f32 %f3,0f3FD9DB23;\n"          // 1 or 1.702
                         " setp.eq.u32 %p5,%r10,3; @%p5 bra TH;\n"
                         " mul.f32 %f4,%f2,%f3; mul.f32 %f4,%f4,0fBFB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; div.approx.f32 %f2,%f2,%f4; bra ST;\n"
                         "TH: mul.f32 %f4,%f2,%f2; mul.f32 %f4,%f4,%f2; fma.rn.f32 %f4,%f4,0f3D372713,%f2; mul.f32 %f4,%f4,0f3F4C422A;\n"   // sqrt(2/pi)(x+0.044715x^3)
                         " mul.f32 %f4,%f4,0fC038AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; div.approx.f32 %f2,%f2,%f4; bra ST;\n"  // x*sigmoid(2u)
                         // mish: x * tanh(softplus(x)); softplus = max(x,0) + log(1+e^-|x|)
                         "MI: abs.f32 %f4,%f2; mul.f32 %f4,%f4,0fBFB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; lg2.approx.f32 %f4,%f4; mul.f32 %f4,%f4,0f3F317218;\n"
                         " max.f32 %f5,%f2,0f00000000; add.f32 %f4,%f4,%f5;\n"
                         " mul.f32 %f5,%f4,0fC038AA3B; ex2.approx.f32 %f5,%f5; mov.f32 %f6,0f3F800000; sub.f32 %f7,%f6,%f5; add.f32 %f8,%f6,%f5; div.approx.f32 %f7,%f7,%f8;\n"
                         " mul.f32 %f2,%f2,%f7; bra ST;\n"
                         "GE:\n") + kGelu + " ST: st.global.f32 [%rd2],%f2;\n").c_str());
    // y += a * x
    s += ew("k_axpy", ".param .u64 pY, .param .u64 pX, .param .f32 pA", " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pX]; ld.param.f32 %f1,[pA];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3; ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5]; fma.rn.f32 %f2,%f3,%f1,%f2; st.global.f32 [%rd4],%f2;\n");
    // y[i] += b[i / HW]   (per-channel add on NCHW)
    s += ew("k_chanadd", ".param .u64 pY, .param .u64 pB, .param .u32 pHW", " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pB]; ld.param.u32 %r10,[pHW];\n",
            " div.u32 %r11,%r5,%r10; mul.wide.u32 %rd3,%r5,4; add.u64 %rd3,%rd1,%rd3; mul.wide.u32 %rd4,%r11,4; add.u64 %rd4,%rd2,%rd4;\n"
            " ld.global.f32 %f2,[%rd3]; ld.global.f32 %f3,[%rd4]; add.f32 %f2,%f2,%f3; st.global.f32 [%rd3],%f2;\n");
    // out[t*F+j] = in[t*2F+j] * gelu(in[t*2F+F+j])
    s += ew("k_geglu", ".param .u64 pI, .param .u64 pO, .param .u32 pF", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd5,[pO]; ld.param.u32 %r10,[pF];\n",
            (std::string(" div.u32 %r11,%r5,%r10; mul.lo.u32 %r12,%r11,%r10; sub.u32 %r12,%r5,%r12; shl.b32 %r13,%r10,1; mad.lo.u32 %r13,%r13,%r11,%r12;\n"
                         " mul.wide.u32 %rd2,%r13,4; add.u64 %rd2,%rd1,%rd2; ld.global.f32 %f9,[%rd2]; mul.wide.u32 %rd3,%r10,4; add.u64 %rd3,%rd2,%rd3; ld.global.f32 %f2,[%rd3];\n") +
             kGelu + " mul.f32 %f2,%f2,%f9; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd5,%rd4; st.global.f32 [%rd4],%f2;\n").c_str());
    // out[c*R + r] = in[r*Cc + c]   (i indexes out)
    s += ew("k_transpose", ".param .u64 pI, .param .u64 pO, .param .u32 pR, .param .u32 pC", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r10,[pR]; ld.param.u32 %r11,[pC];\n",
            " div.u32 %r12,%r5,%r10; mul.lo.u32 %r13,%r12,%r10; sub.u32 %r13,%r5,%r13; mad.lo.u32 %r14,%r13,%r11,%r12;\n"
            " mul.wide.u32 %rd3,%r14,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3]; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // nearest 2x upsample, NCHW: out [C][2H][2W]
    s += ew("k_up2", ".param .u64 pI, .param .u64 pO, .param .u32 pH, .param .u32 pW", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r10,[pH]; ld.param.u32 %r11,[pW];\n",
            " shl.b32 %r12,%r11,1; shl.b32 %r13,%r10,1; mul.lo.u32 %r14,%r12,%r13;\n"
            " div.u32 %r15,%r5,%r14; mul.lo.u32 %r16,%r15,%r14; sub.u32 %r16,%r5,%r16; div.u32 %r17,%r16,%r12; mul.lo.u32 %r18,%r17,%r12; sub.u32 %r18,%r16,%r18;\n"
            " shr.u32 %r17,%r17,1; shr.u32 %r18,%r18,1; mul.lo.u32 %r19,%r15,%r10; add.u32 %r19,%r19,%r17; mad.lo.u32 %r19,%r19,%r11,%r18;\n"
            " mul.wide.u32 %rd3,%r19,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3]; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // out[t*D+d] = f16 table[ids[t]*D+d]
    s += ew("k_embed", ".param .u64 pIds, .param .u64 pT, .param .u64 pO, .param .u32 pD", " ld.param.u64 %rd1,[pIds]; ld.param.u64 %rd2,[pT]; ld.param.u64 %rd3,[pO]; ld.param.u32 %r10,[pD];\n",
            " div.u32 %r11,%r5,%r10; mul.lo.u32 %r12,%r11,%r10; sub.u32 %r12,%r5,%r12; mul.wide.u32 %rd4,%r11,4; add.u64 %rd4,%rd1,%rd4; ld.global.u32 %r13,[%rd4];\n"
            " mad.lo.u32 %r14,%r13,%r10,%r12; mul.wide.u32 %rd5,%r14,2; add.u64 %rd5,%rd2,%rd5; ld.global.b16 %h1,[%rd5]; cvt.f32.f16 %f2,%h1;\n"
            " mul.wide.u32 %rd6,%r5,4; add.u64 %rd6,%rd3,%rd6; st.global.f32 [%rd6],%f2;\n");
    // im2col for pixels [p0, p0+n/K): col[i] with p = p0 + i/K, k = i%K, k = (ci, ky, kx)
    s += ew("k_im2col", ".param .u64 pX, .param .u64 pCol, .param .u32 pH, .param .u32 pW, .param .u32 pKH, .param .u32 pKW, .param .u32 pS, .param .u32 pP, .param .u32 pWo, .param .u32 pK, .param .u32 pP0, .param .u32 pPW",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pCol]; ld.param.u32 %r10,[pH]; ld.param.u32 %r11,[pW]; ld.param.u32 %r12,[pKH]; ld.param.u32 %r13,[pKW];"
            " ld.param.u32 %r14,[pS]; ld.param.u32 %r15,[pP]; ld.param.u32 %r16,[pWo]; ld.param.u32 %r17,[pK]; ld.param.u32 %r18,[pP0]; ld.param.u32 %r29,[pPW];\n",
            " div.u32 %r19,%r5,%r17; mul.lo.u32 %r20,%r19,%r17; sub.u32 %r20,%r5,%r20; add.u32 %r19,%r19,%r18;\n"        // p, k
            " mul.lo.u32 %r21,%r12,%r13; div.u32 %r22,%r20,%r21; mul.lo.u32 %r23,%r22,%r21; sub.u32 %r23,%r20,%r23;\n"    // ci, r
            " div.u32 %r24,%r23,%r13; mul.lo.u32 %r25,%r24,%r13; sub.u32 %r25,%r23,%r25;\n"                               // ky, kx
            " div.u32 %r26,%r19,%r16; mul.lo.u32 %r27,%r26,%r16; sub.u32 %r27,%r19,%r27;\n"                               // oy, ox
            " mad.lo.u32 %r26,%r26,%r14,%r24; sub.u32 %r26,%r26,%r15; mad.lo.u32 %r27,%r27,%r14,%r25; sub.u32 %r27,%r27,%r29;\n"  // iy, ix
            " mov.f32 %f2,0f00000000; setp.lt.u32 %p2,%r26,%r10; setp.lt.u32 %p3,%r27,%r11; and.pred %p2,%p2,%p3;\n"
            " mad.lo.u32 %r28,%r22,%r10,%r26; mad.lo.u32 %r28,%r28,%r11,%r27; mul.wide.u32 %rd3,%r28,4; add.u64 %rd3,%rd1,%rd3; @%p2 ld.global.f32 %f2,[%rd3];\n"
            " mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // out[c*ldo + r0 + r] = in[r*C + c] (+ beta * out)   i over C*R
    s += ew("k_tradd", ".param .u64 pI, .param .u64 pO, .param .u32 pR, .param .u32 pC, .param .u32 pLdo, .param .u32 pR0, .param .f32 pBeta",
            " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r10,[pR]; ld.param.u32 %r11,[pC]; ld.param.u32 %r12,[pLdo]; ld.param.u32 %r13,[pR0]; ld.param.f32 %f9,[pBeta];\n",
            " div.u32 %r14,%r5,%r10; mul.lo.u32 %r15,%r14,%r10; sub.u32 %r15,%r5,%r15;\n"                                // c, r
            " mad.lo.u32 %r16,%r15,%r11,%r14; mul.wide.u32 %rd3,%r16,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3];\n"
            " add.u32 %r17,%r13,%r15; mad.lo.u32 %r17,%r14,%r12,%r17; mul.wide.u32 %rd4,%r17,4; add.u64 %rd4,%rd2,%rd4;\n"
            " setp.ne.f32 %p2,%f9,0f00000000; @%p2 ld.global.f32 %f3,[%rd4]; @%p2 fma.rn.f32 %f2,%f3,%f9,%f2; st.global.f32 [%rd4],%f2;\n");
    // ---- video engine kernels ----
    // RMS norm over rows of length D with weight: y = x / sqrt(mean(x^2)+eps) * w. grid = rows.
    s += ".visible .entry k_rmsrow(.param .u64 pX, .param .u64 pG, .param .u64 pY, .param .u32 pD, .param .f32 pE)\n{\n"
         " .reg .pred %p<6>; .reg .b32 %r<24>; .reg .f32 %f<24>; .reg .b64 %rd<16>;\n";
    s += kRed;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pG]; ld.param.u64 %rd4,[pY]; ld.param.u32 %r1,[pD]; ld.param.f32 %f20,[pE];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x; mul.wide.u32 %rd5,%r4,%r1; shl.b64 %rd5,%rd5,2; add.u64 %rd6,%rd1,%rd5; add.u64 %rd7,%rd4,%rd5;\n"
         " mov.f32 %f1,0f00000000; mov.u32 %r7,%r5;\n"
         "A1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra A2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f2,[%rd8]; fma.rn.f32 %f1,%f2,%f2,%f1; add.u32 %r7,%r7,256; bra A1;\nA2:\n";
    s += bred("%f1");
    s += " cvt.rn.f32.u32 %f3,%r1; div.rn.f32 %f5,%f1,%f3; add.f32 %f5,%f5,%f20; rsqrt.approx.f32 %f5,%f5; mov.u32 %r7,%r5;\n"
         "C1: setp.ge.u32 %p1,%r7,%r1; @%p1 bra C2; mul.wide.u32 %rd8,%r7,4; add.u64 %rd9,%rd6,%rd8; ld.global.f32 %f2,[%rd9]; add.u64 %rd10,%rd2,%rd8; ld.global.f32 %f6,[%rd10];\n"
         " mul.f32 %f2,%f2,%f5; mul.f32 %f2,%f2,%f6; add.u64 %rd12,%rd7,%rd8; st.global.f32 [%rd12],%f2; add.u32 %r7,%r7,256; bra C1;\nC2: ret;\n}\n";
    // Channel-first L2 norm (Wan VAE RMS_norm): y[c][n] = x[c][n] / max(||x[:,n]||, 1e-12) * sqrt(C) * g[c], optional SiLU. i over N.
    s += ew("k_rmsc", ".param .u64 pX, .param .u64 pG, .param .u64 pY, .param .u32 pC, .param .u32 pNN, .param .u32 pAct",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pG]; ld.param.u64 %rd3,[pY]; ld.param.u32 %r10,[pC]; ld.param.u32 %r11,[pNN]; ld.param.u32 %r12,[pAct];\n",
            " mov.f32 %f1,0f00000000; mov.u32 %r13,0; mul.wide.u32 %rd4,%r5,4; add.u64 %rd5,%rd1,%rd4; add.u64 %rd6,%rd3,%rd4; mul.wide.u32 %rd7,%r11,4;\n"
            " mov.u64 %rd8,%rd5;\nL1: setp.ge.u32 %p2,%r13,%r10; @%p2 bra L2; ld.global.f32 %f2,[%rd8]; fma.rn.f32 %f1,%f2,%f2,%f1; add.u64 %rd8,%rd8,%rd7; add.u32 %r13,%r13,1; bra L1;\nL2:\n"
            " sqrt.rn.f32 %f1,%f1; max.f32 %f1,%f1,0f2B8CBCCC; cvt.rn.f32.u32 %f3,%r10; sqrt.rn.f32 %f3,%f3; div.rn.f32 %f4,%f3,%f1;\n"
            " setp.ne.u32 %p3,%r12,0; mov.u32 %r13,0; mov.u64 %rd8,%rd5; mov.u64 %rd9,%rd6; mov.u64 %rd10,%rd2;\n"
            "L3: setp.ge.u32 %p2,%r13,%r10; @%p2 bra L4; ld.global.f32 %f2,[%rd8]; ld.global.f32 %f5,[%rd10]; mul.f32 %f2,%f2,%f4; mul.f32 %f2,%f2,%f5;\n"
            " @%p3 mul.f32 %f6,%f2,0fBFB8AA3B; @%p3 ex2.approx.f32 %f6,%f6; @%p3 add.f32 %f6,%f6,0f3F800000; @%p3 div.approx.f32 %f2,%f2,%f6;\n"
            " st.global.f32 [%rd9],%f2; add.u64 %rd8,%rd8,%rd7; add.u64 %rd9,%rd9,%rd7; add.u64 %rd10,%rd10,4; add.u32 %r13,%r13,1; bra L3;\nL4:\n");
    // int8 quantization with one scale per 32 consecutive weights (like Q8_0). i over groups.
    s += ew("k_q8row", ".param .u64 pI, .param .u64 pO, .param .u64 pS", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO]; ld.param.u64 %rd3,[pS];\n",
            " mul.wide.u32 %rd4,%r5,64; add.u64 %rd4,%rd1,%rd4; mul.wide.u32 %rd5,%r5,32; add.u64 %rd5,%rd2,%rd5;\n"
            " mov.f32 %f1,0f00000000; mov.u32 %r10,0;\n"
            "A1: mul.wide.u32 %rd6,%r10,2; add.u64 %rd6,%rd4,%rd6; ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f2,%h1; abs.f32 %f2,%f2; max.f32 %f1,%f1,%f2;\n"
            " add.u32 %r10,%r10,1; setp.lt.u32 %p2,%r10,32; @%p2 bra A1;\n"
            " mul.f32 %f3,%f1,0f3C010204; setp.gt.f32 %p3,%f1,0f00000000; div.rn.f32 %f4,0f42FE0000,%f1; selp.f32 %f4,%f4,0f00000000,%p3;\n"
            " mul.wide.u32 %rd7,%r5,4; add.u64 %rd7,%rd3,%rd7; st.global.f32 [%rd7],%f3; mov.u32 %r10,0;\n"
            "B1: mul.wide.u32 %rd6,%r10,2; add.u64 %rd6,%rd4,%rd6; ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f2,%h1; mul.f32 %f2,%f2,%f4; cvt.rni.s32.f32 %r11,%f2;\n"
            " cvt.u64.u32 %rd8,%r10; add.u64 %rd8,%rd5,%rd8; st.global.s8 [%rd8],%r11; add.u32 %r10,%r10,1; setp.lt.u32 %p2,%r10,32; @%p2 bra B1;\n");
    // y[i] *= x[i]
    s += ew("k_mul", ".param .u64 pY, .param .u64 pX", " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pX];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3; ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5]; mul.f32 %f2,%f2,%f3; st.global.f32 [%rd4],%f2;\n");
    // x[i] += y[i] * g[i % C]
    s += ew("k_gadd", ".param .u64 pX, .param .u64 pY, .param .u64 pG, .param .u32 pC",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pY]; ld.param.u64 %rd3,[pG]; ld.param.u32 %r10,[pC];\n",
            " rem.u32 %r11,%r5,%r10; mul.wide.u32 %rd4,%r5,4; add.u64 %rd5,%rd1,%rd4; add.u64 %rd6,%rd2,%rd4; mul.wide.u32 %rd7,%r11,4; add.u64 %rd7,%rd3,%rd7;\n"
            " ld.global.f32 %f2,[%rd5]; ld.global.f32 %f3,[%rd6]; ld.global.f32 %f4,[%rd7]; fma.rn.f32 %f2,%f3,%f4,%f2; st.global.f32 [%rd5],%f2;\n");
    // rotary on adjacent pairs: x [L][heads*hd], table cs [L][hd/2][2]; i over L*heads*hd/2
    s += ew("k_rope3", ".param .u64 pX, .param .u64 pCS, .param .u32 pHeads, .param .u32 pHalf",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pCS]; ld.param.u32 %r10,[pHeads]; ld.param.u32 %r11,[pHalf];\n",
            " mul.lo.u32 %r12,%r10,%r11; div.u32 %r13,%r5,%r12; rem.u32 %r14,%r5,%r11;\n"     // l, j
            " mad.lo.u32 %r15,%r13,%r11,%r14; mul.wide.u32 %rd3,%r15,8; add.u64 %rd3,%rd2,%rd3; ld.global.f32 %f5,[%rd3]; ld.global.f32 %f6,[%rd3+4];\n"
            " mul.wide.u32 %rd4,%r5,8; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd4+4];\n"
            " mul.f32 %f7,%f2,%f5; mul.f32 %f8,%f3,%f6; sub.f32 %f7,%f7,%f8; mul.f32 %f9,%f2,%f6; fma.rn.f32 %f9,%f3,%f5,%f9;\n"
            " st.global.f32 [%rd4],%f7; st.global.f32 [%rd4+4],%f9;\n");
    // 3D im2col (stride 1): x [Ci][Tin][H][W]; output frame t0 uses input frames t0..t0+KT-1; spatial pad P.
    // col[i], p = P0 + i/K over H*W, k = ((ci*KT+kt)*KH+ky)*KW+kx
    s += ew("k_im2col3", ".param .u64 pX, .param .u64 pCol, .param .u32 pTin, .param .u32 pH, .param .u32 pW, .param .u32 pKT, .param .u32 pKH, .param .u32 pKW, .param .u32 pP, .param .u32 pT0, .param .u32 pK, .param .u32 pP0",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pCol]; ld.param.u32 %r10,[pTin]; ld.param.u32 %r11,[pH]; ld.param.u32 %r12,[pW]; ld.param.u32 %r13,[pKT];"
            " ld.param.u32 %r14,[pKH]; ld.param.u32 %r15,[pKW]; ld.param.u32 %r16,[pP]; ld.param.u32 %r17,[pT0]; ld.param.u32 %r18,[pK]; ld.param.u32 %r19,[pP0];\n",
            " div.u32 %r20,%r5,%r18; mul.lo.u32 %r21,%r20,%r18; sub.u32 %r21,%r5,%r21; add.u32 %r20,%r20,%r19;\n"      // p, k
            " mul.lo.u32 %r22,%r14,%r15; div.u32 %r23,%r21,%r22; mul.lo.u32 %r24,%r23,%r22; sub.u32 %r24,%r21,%r24;\n"   // (ci*KT+kt), r
            " div.u32 %r25,%r23,%r13; mul.lo.u32 %r26,%r25,%r13; sub.u32 %r26,%r23,%r26;\n"                               // ci, kt
            " div.u32 %r27,%r24,%r15; mul.lo.u32 %r28,%r27,%r15; sub.u32 %r28,%r24,%r28;\n"                               // ky, kx
            " div.u32 %r29,%r20,%r12; mul.lo.u32 %r30,%r29,%r12; sub.u32 %r30,%r20,%r30;\n"                               // oy, ox
            " add.u32 %r29,%r29,%r27; sub.u32 %r29,%r29,%r16; add.u32 %r30,%r30,%r28; sub.u32 %r30,%r30,%r16; add.u32 %r26,%r26,%r17;\n"
            " mov.f32 %f2,0f00000000; setp.lt.u32 %p2,%r29,%r11; setp.lt.u32 %p3,%r30,%r12; and.pred %p2,%p2,%p3;\n"
            " mad.lo.u32 %r31,%r25,%r10,%r26; mad.lo.u32 %r31,%r31,%r11,%r29; mad.lo.u32 %r31,%r31,%r12,%r30; mul.wide.u32 %rd3,%r31,4; add.u64 %rd3,%rd1,%rd3; @%p2 ld.global.f32 %f2,[%rd3];\n"
            " mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // DupUp3D: x [Cin][T][H][W] -> out [Cout][T*ft-drop][H*fs][W*fs]; i over out
    s += ew("k_dupup", ".param .u64 pX, .param .u64 pO, .param .u32 pRep, .param .u32 pT, .param .u32 pH, .param .u32 pW, .param .u32 pFt, .param .u32 pFs, .param .u32 pDrop",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r10,[pRep]; ld.param.u32 %r11,[pT]; ld.param.u32 %r12,[pH]; ld.param.u32 %r13,[pW];"
            " ld.param.u32 %r14,[pFt]; ld.param.u32 %r15,[pFs]; ld.param.u32 %r16,[pDrop];\n",
            " mul.lo.u32 %r17,%r13,%r15; mul.lo.u32 %r18,%r12,%r15; mul.lo.u32 %r19,%r11,%r14; sub.u32 %r19,%r19,%r16;\n"   // Wo, Ho, To
            " div.u32 %r20,%r5,%r17; mul.lo.u32 %r21,%r20,%r17; sub.u32 %r21,%r5,%r21;\n"                                  // rest, wo
            " div.u32 %r22,%r20,%r18; mul.lo.u32 %r23,%r22,%r18; sub.u32 %r23,%r20,%r23;\n"                                // rest, ho
            " div.u32 %r24,%r22,%r19; mul.lo.u32 %r25,%r24,%r19; sub.u32 %r25,%r22,%r25; add.u32 %r25,%r25,%r16;\n"       // co, to (+drop)
            " div.u32 %r26,%r25,%r14; mul.lo.u32 %r27,%r26,%r14; sub.u32 %r27,%r25,%r27;\n"                                // t, a
            " div.u32 %r28,%r23,%r15; mul.lo.u32 %r29,%r28,%r15; sub.u32 %r29,%r23,%r29;\n"                                // h, b
            " div.u32 %r30,%r21,%r15; mul.lo.u32 %r31,%r30,%r15; sub.u32 %r31,%r21,%r31;\n"                                // w, c
            " mul.lo.u32 %r6,%r15,%r15; mul.lo.u32 %r7,%r6,%r14; mul.lo.u32 %r8,%r24,%r7; mad.lo.u32 %r8,%r27,%r6,%r8; mad.lo.u32 %r8,%r29,%r15,%r8; add.u32 %r8,%r8,%r31;\n"  // j
            " div.u32 %r8,%r8,%r10; mad.lo.u32 %r8,%r8,%r11,%r26; mad.lo.u32 %r8,%r8,%r12,%r28; mad.lo.u32 %r8,%r8,%r13,%r30;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3]; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // out[c][2t+s][p] = x[s*C+c][t][p]; i over C*2T*HW
    s += ew("k_tinter", ".param .u64 pX, .param .u64 pO, .param .u32 pC, .param .u32 pT, .param .u32 pHW",
            " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r10,[pC]; ld.param.u32 %r11,[pT]; ld.param.u32 %r12,[pHW];\n",
            " div.u32 %r13,%r5,%r12; mul.lo.u32 %r14,%r13,%r12; sub.u32 %r14,%r5,%r14; shl.b32 %r15,%r11,1;\n"
            " div.u32 %r16,%r13,%r15; mul.lo.u32 %r17,%r16,%r15; sub.u32 %r17,%r13,%r17; shr.u32 %r18,%r17,1; and.b32 %r19,%r17,1;\n"   // c, to, t, s
            " mad.lo.u32 %r20,%r19,%r10,%r16; mad.lo.u32 %r20,%r20,%r11,%r18; mad.lo.u32 %r20,%r20,%r12,%r14;\n"
            " mul.wide.u32 %rd3,%r20,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3]; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    // fp8 e4m3 -> f16, times a scale
    s += ew("k_f8", ".param .u64 pI, .param .u64 pO, .param .f32 pS", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO]; ld.param.f32 %f9,[pS];\n",
            " cvt.u64.u32 %rd3,%r5; add.u64 %rd3,%rd1,%rd3; ld.global.u8 %r10,[%rd3];\n"
            " shr.u32 %r11,%r10,3; and.b32 %r11,%r11,15; and.b32 %r12,%r10,7; and.b32 %r13,%r10,128;\n"
            " setp.eq.u32 %p2,%r11,0; cvt.rn.f32.u32 %f2,%r12; mul.f32 %f2,%f2,0f3B000000;\n"                        // subnormal: m * 2^-9
            " add.u32 %r14,%r11,120; shl.b32 %r14,%r14,23; shl.b32 %r15,%r12,20; or.b32 %r14,%r14,%r15; mov.b32 %f3,%r14;\n"
            " selp.f32 %f2,%f2,%f3,%p2; setp.ne.u32 %p3,%r13,0; @%p3 neg.f32 %f2,%f2; mul.f32 %f2,%f2,%f9;\n"
            " cvt.rn.f16.f32 %h1,%f2; mul.wide.u32 %rd4,%r5,2; add.u64 %rd4,%rd2,%rd4; st.global.b16 [%rd4],%h1;\n");
    // f16 -> f32
    s += ew("k_h2f", ".param .u64 pI, .param .u64 pO", " ld.param.u64 %rd1,[pI]; ld.param.u64 %rd2,[pO];\n",
            " mul.wide.u32 %rd3,%r5,2; add.u64 %rd3,%rd1,%rd3; ld.global.b16 %h1,[%rd3]; cvt.f32.f16 %f2,%h1; mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");
    return s;
}
