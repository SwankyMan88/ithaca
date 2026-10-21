#include "cuda_tier.h"
#include "common.h"
#include <cstdio>
#include <algorithm>
#include <future>

std::string gpu_ops_ptx();

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
#define CUAPI __stdcall

static CUresult(CUAPI* cuInit)(unsigned);
static CUresult(CUAPI* cuDeviceGet)(CUdevice*, int);
static CUresult(CUAPI* cuDeviceGetCount)(int*);
static CUresult(CUAPI* cuDeviceGetName)(char*, int, CUdevice);
static CUresult(CUAPI* cuCtxCreate)(void**, unsigned, CUdevice);
static CUresult(CUAPI* cuCtxSetCurrent)(void*);
static CUresult(CUAPI* cuModuleLoadDataEx)(void**, const void*, unsigned, int*, void**);
static CUresult(CUAPI* cuModuleGetFunction)(void**, void*, const char*);
static CUresult(CUAPI* cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult(CUAPI* cuMemFree)(CUdeviceptr);
static CUresult(CUAPI* cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
static CUresult(CUAPI* cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
static CUresult(CUAPI* cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, void*);
static CUresult(CUAPI* cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, void*);
static CUresult(CUAPI* cuStreamCreate)(void**, unsigned);
static CUresult(CUAPI* cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, void*);
static CUresult(CUAPI* cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, void*);
static CUresult(CUAPI* cuStreamSynchronize)(void*);
static CUresult(CUAPI* cuStreamBeginCapture)(void*, int);
static CUresult(CUAPI* cuStreamEndCapture)(void*, void**);
static CUresult(CUAPI* cuGraphInstantiateWithFlags)(void**, void*, unsigned long long);
static CUresult(CUAPI* cuGraphLaunch)(void*, void*);
static CUresult(CUAPI* cuGraphExecDestroy)(void*);
static CUresult(CUAPI* cuGraphDestroy)(void*);
static CUresult(CUAPI* cuMemGetInfo)(size_t*, size_t*);
static CUresult(CUAPI* cuCtxSynchronize)();
static CUresult(CUAPI* cuMemHostAlloc)(void**, size_t, unsigned);
static CUresult(CUAPI* cuEventCreate)(void**, unsigned);
static CUresult(CUAPI* cuEventRecord)(void*, void*);
static CUresult(CUAPI* cuEventSynchronize)(void*);
static CUresult(CUAPI* cuLaunchKernel)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                       unsigned, void*, void**, void**);

// Kernels are generated as PTX text here and JIT-compiled by the driver.
// GPU weight layout (repacked at upload): [f32 scale per block][pad][quants, 16 B (Q4) or 32 B (Q8) per block].
// Activations are quantized to int8 blocks on the GPU first (k_quant), then rows are dotted with dp4a.
// 8 warps per CTA, one warp per output row, 8 tokens per CTA so small batches reuse each weight load.
static const int kTokTile = 8;   // tokens per CTA: draft checks (<= 8 tokens) read each weight once
static const int kRows = 4;   // output rows per warp: each activation block loaded once serves all of them
static std::string gen_kernel(const char* name, int mode) {
    const bool q4 = mode != 0, qm = mode == 2;
    const int NW = q4 ? 4 : 8;   // weight registers per row block (packed nibbles are unpacked in place below)
    std::string k;
    char b[400];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); k += b; k += "\n"; };
    L(".visible .entry %s(.param .u64 pS, .param .u64 pQ, .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS, .param .u64 pY,"
      " .param .u32 pC, .param .u32 pR, .param .u32 pN)", name);
    L("{\n .reg .pred %%p<8>; .reg .pred %%pt<8>; .reg .pred %%pr<%d>; .reg .b32 %%r<96>; .reg .b32 %%w<%d>; .reg .b32 %%x<8>; .reg .f32 %%f<16>;"
      " .reg .f32 %%sc<%d>; .reg .f32 %%mn<%d>; .reg .f32 %%a<%d>; .reg .b64 %%rd<24>;", kRows, kRows * 8, kRows, kRows, kTokTile * kRows);
    k += " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pQ]; ld.param.u64 %rd3,[pXQ]; ld.param.u64 %rd4,[pXD]; ld.param.u64 %rd5,[pXS]; ld.param.u64 %rd6,[pY];\n";
    k += " ld.param.u32 %r1,[pC]; ld.param.u32 %r2,[pR]; ld.param.u32 %r3,[pN];\n";
    k += " mov.u32 %r4,%tid.x; shr.u32 %r5,%r4,5; and.b32 %r6,%r4,31;\n";                         // warp, lane
    L(" mov.u32 %%r7,%%ctaid.x; mul.lo.u32 %%r7,%%r7,%d; mad.lo.u32 %%r7,%%r5,%d,%%r7;", 8 * kRows, kRows);   // first row of this warp
    k += " setp.ge.u32 %p1,%r7,%r2; @%p1 bra DONE;\n";
    k += " mov.u32 %r8,%ctaid.y; shl.b32 %r8,%r8,3;\n";                                             // first token of tile
    k += " shr.u32 %r9,%r1,5;\n";                                                                    // nb
    for (int j = 0; j < kRows; j++)
        L(" add.u32 %%r%d,%%r7,%d; setp.lt.u32 %%pr%d,%%r%d,%%r2; mul.lo.u32 %%r%d,%%r%d,%%r9;", 60 + j, j, j, 60 + j, 64 + j, 60 + j);   // row j, row*nb
    for (int t = 0; t < kTokTile; t++) {
        L(" add.u32 %%r%d,%%r8,%d; setp.lt.u32 %%pt%d,%%r%d,%%r3; mul.lo.u32 %%r%d,%%r%d,%%r9;", 40 + t, t, t, 40 + t, 50 + t, 40 + t);
        for (int j = 0; j < kRows; j++) L(" mov.f32 %%a%d,0f00000000;", t * kRows + j);
    }
    k += " mov.u32 %r11,%r6;\n";
    k += "LOOP:\n setp.ge.u32 %p2,%r11,%r9; @%p2 bra END;\n";
    for (int j = 0; j < kRows; j++) {
        const int w0 = j * 8;
        L(" add.u32 %%r12,%%r%d,%%r11; mov.f32 %%sc%d,0f00000000; mov.f32 %%mn%d,0f00000000;", 64 + j, j, j);
        for (int i = 0; i < 8; i++) L(" mov.b32 %%w%d,0;", w0 + i);
        if (qm) L(" mul.wide.u32 %%rd7,%%r12,8; add.u64 %%rd7,%%rd1,%%rd7; @%%pr%d ld.global.nc.v2.f32 {%%sc%d,%%mn%d},[%%rd7];", j, j, j);
        else L(" mul.wide.u32 %%rd7,%%r12,4; add.u64 %%rd7,%%rd1,%%rd7; @%%pr%d ld.global.nc.f32 %%sc%d,[%%rd7];", j, j);
        if (q4) L(" mul.wide.u32 %%rd8,%%r12,16; add.u64 %%rd8,%%rd2,%%rd8; @%%pr%d ld.global.nc.v4.u32 {%%w%d,%%w%d,%%w%d,%%w%d},[%%rd8];", j, w0, w0 + 1, w0 + 2, w0 + 3);
        else L(" mul.wide.u32 %%rd8,%%r12,32; add.u64 %%rd8,%%rd2,%%rd8; @%%pr%d ld.global.nc.v4.u32 {%%w%d,%%w%d,%%w%d,%%w%d},[%%rd8]; @%%pr%d ld.global.nc.v4.u32 {%%w%d,%%w%d,%%w%d,%%w%d},[%%rd8+16];",
               j, w0, w0 + 1, w0 + 2, w0 + 3, j, w0 + 4, w0 + 5, w0 + 6, w0 + 7);
        if (q4)   // nibbles: w0-3 = low (elements 0..15), w4-7 = high (16..31)
            for (int i = 0; i < 4; i++) L(" shr.u32 %%w%d,%%w%d,4; and.b32 %%w%d,%%w%d,0x0F0F0F0F; and.b32 %%w%d,%%w%d,0x0F0F0F0F;", w0 + 4 + i, w0 + i, w0 + 4 + i, w0 + 4 + i, w0 + i, w0 + i);
    }
    (void)NW;
    for (int t = 0; t < kTokTile; t++) {
        L(" @!%%pt%d bra SKIP%d;", t, t);
        L(" add.u32 %%r13,%%r%d,%%r11;", 50 + t);
        k += " mul.wide.u32 %rd9,%r13,32; add.u64 %rd9,%rd3,%rd9; ld.global.v4.u32 {%x0,%x1,%x2,%x3},[%rd9]; ld.global.v4.u32 {%x4,%x5,%x6,%x7},[%rd9+16];\n";
        k += " mul.wide.u32 %rd10,%r13,4; add.u64 %rd11,%rd4,%rd10; ld.global.f32 %f2,[%rd11]; add.u64 %rd12,%rd5,%rd10; ld.global.s32 %r14,[%rd12];\n";
        if (qm) k += " cvt.rn.f32.s32 %f10,%r14;\n";
        for (int j = 0; j < kRows; j++) {
            const int w0 = j * 8, acc = t * kRows + j;
            k += " mov.u32 %r15,0;\n";
            for (int i = 0; i < 8; i++) L(" dp4a.%s.s32 %%r15,%%w%d,%%x%d,%%r15;", q4 ? "u32" : "s32", w0 + i, i);
            if (qm) L(" cvt.rn.f32.s32 %%f3,%%r15; mul.f32 %%f3,%%f3,%%sc%d; mul.f32 %%f4,%%f10,%%mn%d; sub.f32 %%f3,%%f3,%%f4; fma.rn.f32 %%a%d,%%f3,%%f2,%%a%d;", j, j, acc, acc);
            else {
                if (q4) k += " mad.lo.s32 %r15,%r14,-8,%r15;\n";
                L(" cvt.rn.f32.s32 %%f3,%%r15; mul.f32 %%f3,%%f3,%%f2; fma.rn.f32 %%a%d,%%f3,%%sc%d,%%a%d;", acc, j, acc);
            }
        }
        L("SKIP%d:", t);
    }
    k += " add.u32 %r11,%r11,32; bra LOOP;\nEND:\n";
    for (int t = 0; t < kTokTile; t++)
        for (int j = 0; j < kRows; j++)
            for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f8,%%a%d,%d,31,0xffffffff; add.f32 %%a%d,%%a%d,%%f8;", t * kRows + j, o, t * kRows + j, t * kRows + j);
    k += " setp.ne.u32 %p3,%r6,0; @%p3 bra DONE;\n";
    for (int t = 0; t < kTokTile; t++)
        for (int j = 0; j < kRows; j++)
            L(" and.pred %%p4,%%pt%d,%%pr%d; @%%p4 mad.lo.u32 %%r16,%%r%d,%%r2,%%r%d; @%%p4 mul.wide.u32 %%rd13,%%r16,4; @%%p4 add.u64 %%rd13,%%rd6,%%rd13; @%%p4 st.global.f32 [%%rd13],%%a%d;",
              t, j, 40 + t, 60 + j, t * kRows + j);
    k += "DONE:\n ret;\n}\n";
    return k;
}

// ---- weight converters: raw GGUF blocks (on the GPU) -> the matmul layout. One warp per output block of 32.
// modes: 0 Q4_0, 1 Q8_0, 2 Q4_K (-> Q4M, exact), 3 Q6_K, 4 Q5_K, 5 F16, 6 BF16, 7 F32 (3..7 -> Q8),
//        8 Q4_1 (-> Q4M, exact), 9 Q5_0, 10 Q5_1 (-> Q8)
static const char* kCvNames[11] = {"cv_q4_0", "cv_q8_0", "cv_q4_k", "cv_q6_k", "cv_q5_k", "cv_f16", "cv_bf16", "cv_f32", "cv_q4_1", "cv_q5_0", "cv_q5_1"};
static std::string gen_cv(int mode) {
    std::string k; char b[400];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); k += b; k += "\n"; };
    L(".visible .entry %s(.param .u64 pSrc, .param .u64 pS, .param .u64 pQ, .param .u64 pM, .param .u32 pN, .param .u32 pB0)", kCvNames[mode]);
    k += "{\n .reg .pred %p<8>; .reg .b32 %r<64>; .reg .b64 %rd<32>; .reg .f32 %f<32>; .reg .b16 %h<4>;\n";
    k += " ld.param.u64 %rd1,[pSrc]; ld.param.u64 %rd2,[pS]; ld.param.u64 %rd3,[pQ]; ld.param.u64 %rd4,[pM]; ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pB0];\n";
    k += " mov.u32 %r3,%tid.x; shr.u32 %r4,%r3,5; and.b32 %r5,%r3,31;\n";
    k += " mov.u32 %r6,%ctaid.x; shl.b32 %r6,%r6,3; add.u32 %r6,%r6,%r4; setp.ge.u32 %p1,%r6,%r1; @%p1 bra DONE;\n";
    k += " add.u32 %r7,%r6,%r2; setp.eq.u32 %p2,%r5,0; cvt.u64.u32 %rd30,%r5;\n";   // global block, lane 0, lane (64-bit)
    k += " mul.wide.u32 %rd5,%r7,4; add.u64 %rd5,%rd2,%rd5;\n";                       // &S[gb]
    auto kscale = [&](const char* base) {   // Q4_K/Q5_K 6-bit scale and min of sub-block j (%r21) -> %r24 sc, %r25 m
        L(" add.u64 %%rd10,%s,4; and.b32 %%r22,%%r21,3; cvt.u64.u32 %%rd11,%%r21; add.u64 %%rd12,%%rd10,%%rd11; ld.global.u8 %%r26,[%%rd12]; ld.global.u8 %%r27,[%%rd12+4];", base);
        k += " cvt.u64.u32 %rd13,%r22; add.u64 %rd13,%rd10,%rd13; ld.global.u8 %r28,[%rd13];\n";   // s[j], s[j+4], s[j&3]
        k += " and.b32 %r24,%r26,63; and.b32 %r25,%r27,63;\n";
        k += " and.b32 %r29,%r27,15; shr.u32 %r30,%r28,6; shl.b32 %r30,%r30,4; or.b32 %r29,%r29,%r30;\n";
        k += " shr.u32 %r31,%r27,4; shr.u32 %r32,%r26,6; shl.b32 %r32,%r32,4; or.b32 %r31,%r31,%r32;\n";
        k += " setp.ge.u32 %p3,%r21,4; selp.b32 %r24,%r29,%r24,%p3; selp.b32 %r25,%r31,%r25,%p3;\n";
    };
    if (mode == 0 || mode == 1) {
        int sz = mode == 0 ? 18 : 34, nq = mode == 0 ? 16 : 32;
        L(" mul.wide.u32 %%rd6,%%r6,%d; add.u64 %%rd6,%%rd1,%%rd6;", sz);
        L(" setp.lt.u32 %%p3,%%r5,%d; @!%%p3 bra NOQ; add.u64 %%rd7,%%rd6,%%rd30; ld.global.u8 %%r8,[%%rd7+2];", nq);
        L(" mul.wide.u32 %%rd8,%%r7,%d; add.u64 %%rd8,%%rd3,%%rd8; add.u64 %%rd8,%%rd8,%%rd30; st.global.u8 [%%rd8],%%r8;", nq);
        k += "NOQ:\n @!%p2 bra DONE; ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f1,%h1; st.global.f32 [%rd5],%f1;\n";
    } else if (mode == 8) {   // Q4_1: value = d*q + m  ->  S = d, M = -m
        k += " mul.wide.u32 %rd6,%r6,20; add.u64 %rd6,%rd1,%rd6;\n";
        k += " setp.lt.u32 %p3,%r5,16; @!%p3 bra NOQ; add.u64 %rd7,%rd6,%rd30; ld.global.u8 %r8,[%rd7+4];\n";
        k += " mul.wide.u32 %rd8,%r7,16; add.u64 %rd8,%rd3,%rd8; add.u64 %rd8,%rd8,%rd30; st.global.u8 [%rd8],%r8;\n";
        k += "NOQ:\n @!%p2 bra DONE; ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f1,%h1; ld.global.b16 %h2,[%rd6+2]; cvt.f32.f16 %f2,%h2; neg.f32 %f2,%f2;\n";
        k += " mul.wide.u32 %rd9,%r7,8; add.u64 %rd9,%rd2,%rd9; st.global.v2.f32 [%rd9],{%f1,%f2};\n";
    } else if (mode == 2) {
        k += " shr.u32 %r20,%r6,3; and.b32 %r21,%r6,7; mul.wide.u32 %rd6,%r20,144; add.u64 %rd6,%rd1,%rd6;\n";
        k += " shr.u32 %r9,%r21,1; shl.b32 %r9,%r9,5; add.u32 %r9,%r9,%r5; cvt.u64.u32 %rd7,%r9; add.u64 %rd7,%rd6,%rd7; ld.global.u8 %r10,[%rd7+16];\n";
        k += " and.b32 %r11,%r21,1; shl.b32 %r11,%r11,2; shr.u32 %r10,%r10,%r11; and.b32 %r10,%r10,15;\n";
        k += " shfl.sync.down.b32 %r12,%r10,16,31,0xffffffff; shl.b32 %r12,%r12,4; or.b32 %r12,%r12,%r10;\n";
        k += " setp.lt.u32 %p3,%r5,16; @!%p3 bra NOQ; mul.wide.u32 %rd8,%r7,16; add.u64 %rd8,%rd3,%rd8; add.u64 %rd8,%rd8,%rd30; st.global.u8 [%rd8],%r12;\n";
        k += "NOQ:\n @!%p2 bra DONE;\n";
        kscale("%rd6");
        k += " ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f1,%h1; ld.global.b16 %h2,[%rd6+2]; cvt.f32.f16 %f2,%h2;\n";
        k += " cvt.rn.f32.u32 %f3,%r24; mul.f32 %f3,%f3,%f1; cvt.rn.f32.u32 %f4,%r25; mul.f32 %f4,%f4,%f2;\n";
        k += " mul.wide.u32 %rd9,%r7,8; add.u64 %rd9,%rd2,%rd9; st.global.v2.f32 [%rd9],{%f3,%f4};\n";
    } else {
        // compute this lane's value into %f10
        if (mode == 3) {
            k += " shr.u32 %r20,%r6,3; and.b32 %r21,%r6,7; mul.wide.u32 %rd6,%r20,210; add.u64 %rd6,%rd1,%rd6;\n";
            k += " shr.u32 %r22,%r21,2; and.b32 %r23,%r21,3;\n";                                   // n, quarter
            k += " shl.b32 %r9,%r22,6; add.u32 %r9,%r9,%r5; and.b32 %r10,%r23,1; shl.b32 %r10,%r10,5; add.u32 %r9,%r9,%r10;\n";
            k += " cvt.u64.u32 %rd7,%r9; add.u64 %rd7,%rd6,%rd7; ld.global.u8 %r11,[%rd7];\n";      // ql
            k += " shr.u32 %r12,%r23,1; shl.b32 %r12,%r12,2; shr.u32 %r11,%r11,%r12; and.b32 %r11,%r11,15;\n";
            k += " shl.b32 %r13,%r22,5; add.u32 %r13,%r13,%r5; cvt.u64.u32 %rd8,%r13; add.u64 %rd8,%rd6,%rd8; ld.global.u8 %r14,[%rd8+128];\n";
            k += " shl.b32 %r15,%r23,1; shr.u32 %r14,%r14,%r15; and.b32 %r14,%r14,3; shl.b32 %r14,%r14,4; or.b32 %r11,%r11,%r14; sub.s32 %r11,%r11,32;\n";
            k += " shl.b32 %r16,%r22,3; shr.u32 %r17,%r5,4; add.u32 %r16,%r16,%r17; shl.b32 %r17,%r23,1; add.u32 %r16,%r16,%r17;\n";
            k += " cvt.u64.u32 %rd9,%r16; add.u64 %rd9,%rd6,%rd9; ld.global.s8 %r18,[%rd9+192];\n";
            k += " ld.global.b16 %h1,[%rd6+208]; cvt.f32.f16 %f1,%h1;\n";
            k += " mul.lo.s32 %r19,%r18,%r11; cvt.rn.f32.s32 %f10,%r19; mul.f32 %f10,%f10,%f1;\n";
        } else if (mode == 4) {
            k += " shr.u32 %r20,%r6,3; and.b32 %r21,%r6,7; mul.wide.u32 %rd6,%r20,176; add.u64 %rd6,%rd1,%rd6;\n";
            k += " shr.u32 %r9,%r21,1; shl.b32 %r9,%r9,5; add.u32 %r9,%r9,%r5; cvt.u64.u32 %rd7,%r9; add.u64 %rd7,%rd6,%rd7; ld.global.u8 %r10,[%rd7+48];\n";
            k += " and.b32 %r11,%r21,1; shl.b32 %r11,%r11,2; shr.u32 %r10,%r10,%r11; and.b32 %r10,%r10,15;\n";
            k += " add.u64 %rd8,%rd6,%rd30; ld.global.u8 %r12,[%rd8+16]; shr.u32 %r12,%r12,%r21; and.b32 %r12,%r12,1; shl.b32 %r12,%r12,4; add.u32 %r10,%r10,%r12;\n";
            kscale("%rd6");
            k += " ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f1,%h1; ld.global.b16 %h2,[%rd6+2]; cvt.f32.f16 %f2,%h2;\n";
            k += " cvt.rn.f32.u32 %f3,%r24; mul.f32 %f3,%f3,%f1; cvt.rn.f32.u32 %f4,%r25; mul.f32 %f4,%f4,%f2;\n";
            k += " cvt.rn.f32.u32 %f5,%r10; mul.f32 %f10,%f3,%f5; sub.f32 %f10,%f10,%f4;\n";
        } else if (mode == 9 || mode == 10) {   // Q5_0: d*(q-16), Q5_1: d*q + m; 5th bit from qh
            const int sz = mode == 9 ? 22 : 24, qh = mode == 9 ? 2 : 4, qs = qh + 4;
            L(" mul.wide.u32 %%rd6,%%r6,%d; add.u64 %%rd6,%%rd1,%%rd6;", sz);
            L(" and.b32 %%r9,%%r5,15; cvt.u64.u32 %%rd7,%%r9; add.u64 %%rd7,%%rd6,%%rd7; ld.global.u8 %%r10,[%%rd7+%d];", qs);
            k += " shr.u32 %r11,%r5,4; shl.b32 %r11,%r11,2; shr.u32 %r10,%r10,%r11; and.b32 %r10,%r10,15;\n";
            L(" shr.u32 %%r12,%%r5,3; cvt.u64.u32 %%rd8,%%r12; add.u64 %%rd8,%%rd6,%%rd8; ld.global.u8 %%r13,[%%rd8+%d];", qh);
            k += " and.b32 %r14,%r5,7; shr.u32 %r13,%r13,%r14; and.b32 %r13,%r13,1; shl.b32 %r13,%r13,4; or.b32 %r10,%r10,%r13;\n";
            k += " ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f1,%h1;\n";
            if (mode == 9) k += " sub.s32 %r10,%r10,16; cvt.rn.f32.s32 %f10,%r10; mul.f32 %f10,%f10,%f1;\n";
            else k += " ld.global.b16 %h2,[%rd6+2]; cvt.f32.f16 %f2,%h2; cvt.rn.f32.s32 %f10,%r10; fma.rn.f32 %f10,%f10,%f1,%f2;\n";
        } else if (mode == 5 || mode == 6) {
            k += " shl.b32 %r9,%r6,5; add.u32 %r9,%r9,%r5; mul.wide.u32 %rd6,%r9,2; add.u64 %rd6,%rd1,%rd6;\n";
            if (mode == 5) k += " ld.global.b16 %h1,[%rd6]; cvt.f32.f16 %f10,%h1;\n";
            else k += " ld.global.u16 %r10,[%rd6]; shl.b32 %r10,%r10,16; mov.b32 %f10,%r10;\n";
        } else {
            k += " shl.b32 %r9,%r6,5; add.u32 %r9,%r9,%r5; mul.wide.u32 %rd6,%r9,4; add.u64 %rd6,%rd1,%rd6; ld.global.f32 %f10,[%rd6];\n";
        }
        // symmetric int8 per block of 32
        k += " abs.f32 %f11,%f10;\n";
        for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f12,%%f11,%d,31,0xffffffff; max.f32 %%f11,%%f11,%%f12;", o);
        k += " mul.f32 %f13,%f11,0f3C010204; setp.gt.f32 %p4,%f11,0f00000000; div.rn.f32 %f14,0f42FE0000,%f11; selp.f32 %f14,%f14,0f00000000,%p4;\n";
        k += " mul.f32 %f15,%f10,%f14; cvt.rni.s32.f32 %r40,%f15;\n";
        k += " mul.wide.u32 %rd15,%r7,32; add.u64 %rd15,%rd3,%rd15; add.u64 %rd15,%rd15,%rd30; st.global.s8 [%rd15],%r40;\n";
        k += " @%p2 st.global.f32 [%rd5],%f13;\n";
    }
    k += "DONE:\n ret;\n}\n";
    return k;
}

// Expert matmul (decode): grid = (ceil(R/8), slots), 256 threads, one warp per output row.
// Row r of slot s reads expert ids[s]: weight block index = (ids[s]*R + r)*nb + b; input row = s*xstep.
static std::string gen_kernel_ex(const char* name, int mode) {
    const bool q4 = mode != 0, qm = mode == 2;
    std::string k; char b[320];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); k += b; k += "\n"; };
    L(".visible .entry %s(.param .u64 pS, .param .u64 pQ, .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS, .param .u64 pY,"
      " .param .u32 pC, .param .u32 pR, .param .u64 pE, .param .u32 pXST)", name);
    k += "{\n .reg .pred %p<4>; .reg .b32 %r<40>; .reg .b32 %w<8>; .reg .b32 %x<8>; .reg .f32 %f<24>; .reg .b64 %rd<24>;\n";
    k += " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pQ]; ld.param.u64 %rd3,[pXQ]; ld.param.u64 %rd4,[pXD]; ld.param.u64 %rd5,[pXS]; ld.param.u64 %rd6,[pY];\n";
    k += " ld.param.u32 %r1,[pC]; ld.param.u32 %r2,[pR]; ld.param.u64 %rd15,[pE]; ld.param.u32 %r3,[pXST];\n";
    k += " mov.u32 %r4,%tid.x; shr.u32 %r5,%r4,5; and.b32 %r6,%r4,31;\n";
    k += " mov.u32 %r7,%ctaid.x; shl.b32 %r7,%r7,3; add.u32 %r7,%r7,%r5;\n";
    k += " setp.ge.u32 %p1,%r7,%r2; @%p1 bra DONE;\n";
    k += " mov.u32 %r8,%ctaid.y; mul.wide.u32 %rd16,%r8,4; add.u64 %rd16,%rd15,%rd16; ld.global.u32 %r20,[%rd16];\n";   // slot, expert
    k += " shr.u32 %r9,%r1,5; mad.lo.u32 %r21,%r20,%r2,%r7; mul.lo.u32 %r10,%r21,%r9;\n";                                // nb, weight block base
    k += " mul.lo.u32 %r22,%r8,%r3; mul.lo.u32 %r24,%r22,%r9;\n";                                                         // input block base
    k += " mov.f32 %f20,0f00000000; mov.u32 %r11,%r6;\n";
    k += "LOOP:\n setp.ge.u32 %p2,%r11,%r9; @%p2 bra END;\n";
    k += " add.u32 %r12,%r10,%r11;\n";
    if (qm) k += " mul.wide.u32 %rd7,%r12,8; add.u64 %rd7,%rd1,%rd7; ld.global.nc.v2.f32 {%f1,%f9},[%rd7];\n";
    else k += " mul.wide.u32 %rd7,%r12,4; add.u64 %rd7,%rd1,%rd7; ld.global.nc.f32 %f1,[%rd7];\n";
    if (q4) k += " mul.wide.u32 %rd8,%r12,16; add.u64 %rd8,%rd2,%rd8; ld.global.nc.v4.u32 {%w0,%w1,%w2,%w3},[%rd8];\n";
    else k += " mul.wide.u32 %rd8,%r12,32; add.u64 %rd8,%rd2,%rd8; ld.global.nc.v4.u32 {%w0,%w1,%w2,%w3},[%rd8]; ld.global.nc.v4.u32 {%w4,%w5,%w6,%w7},[%rd8+16];\n";
    if (q4) for (int i = 0; i < 4; i++) L(" shr.u32 %%w%d,%%w%d,4; and.b32 %%w%d,%%w%d,0x0F0F0F0F; and.b32 %%w%d,%%w%d,0x0F0F0F0F;", 4 + i, i, 4 + i, 4 + i, i, i);
    k += " add.u32 %r13,%r24,%r11;\n";
    k += " mul.wide.u32 %rd9,%r13,32; add.u64 %rd9,%rd3,%rd9; ld.global.v4.u32 {%x0,%x1,%x2,%x3},[%rd9]; ld.global.v4.u32 {%x4,%x5,%x6,%x7},[%rd9+16];\n";
    k += " mul.wide.u32 %rd10,%r13,4; add.u64 %rd11,%rd4,%rd10; ld.global.f32 %f2,[%rd11]; add.u64 %rd12,%rd5,%rd10; ld.global.s32 %r14,[%rd12];\n";
    k += " mov.u32 %r15,0;\n";
    for (int i = 0; i < 8; i++) L(" dp4a.%s.s32 %%r15,%%w%d,%%x%d,%%r15;", q4 ? "u32" : "s32", i, i);
    if (qm) k += " cvt.rn.f32.s32 %f3,%r15; mul.f32 %f3,%f3,%f1; cvt.rn.f32.s32 %f10,%r14; mul.f32 %f10,%f10,%f9; sub.f32 %f3,%f3,%f10; fma.rn.f32 %f20,%f3,%f2,%f20;\n";
    else {
        if (q4) k += " mad.lo.s32 %r15,%r14,-8,%r15;\n";
        k += " cvt.rn.f32.s32 %f3,%r15; mul.f32 %f3,%f3,%f2; fma.rn.f32 %f20,%f3,%f1,%f20;\n";
    }
    k += " add.u32 %r11,%r11,32; bra LOOP;\nEND:\n";
    for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f8,%%f20,%d,31,0xffffffff; add.f32 %%f20,%%f20,%%f8;", o);
    k += " setp.ne.u32 %p3,%r6,0; @%p3 bra DONE;\n";
    k += " mad.lo.u32 %r16,%r8,%r2,%r7; mul.wide.u32 %rd13,%r16,4; add.u64 %rd13,%rd6,%rd13; st.global.f32 [%rd13],%f20;\n";
    k += "DONE:\n ret;\n}\n";
    return k;
}

// Router: one thread. softmax over E logits, pick top K, write ids + weights*scale (renormalized over the K if norm).
static const char* kTopkPTX =
    ".visible .entry k_topk(.param .u64 pL, .param .u32 pE, .param .u32 pK, .param .u32 pNorm, .param .f32 pScale, .param .u64 pI, .param .u64 pW)\n{\n"
    " .reg .pred %p<8>; .reg .b32 %r<24>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n"
    " mov.u32 %r1,%tid.x; setp.ne.u32 %p1,%r1,0; @%p1 bra DONE;\n"
    " ld.param.u64 %rd1,[pL]; ld.param.u32 %r2,[pE]; ld.param.u32 %r3,[pK]; ld.param.u32 %r4,[pNorm]; ld.param.f32 %f1,[pScale];\n"
    " ld.param.u64 %rd2,[pI]; ld.param.u64 %rd3,[pW];\n"
    // max and sum of exp
    " mov.f32 %f2,0fF149F2CA; mov.u32 %r5,0;\n"
    "M1:\n setp.ge.u32 %p2,%r5,%r2; @%p2 bra M1E; mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f3,[%rd5]; max.f32 %f2,%f2,%f3;\n"
    " add.u32 %r5,%r5,1; bra M1;\nM1E:\n"
    " mov.f32 %f4,0f00000000; mov.u32 %r5,0;\n"
    "M2:\n setp.ge.u32 %p2,%r5,%r2; @%p2 bra M2E; mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f3,[%rd5];\n"
    " sub.f32 %f3,%f3,%f2; mul.f32 %f3,%f3,0f3FB8AA3B; ex2.approx.f32 %f3,%f3; add.f32 %f4,%f4,%f3; add.u32 %r5,%r5,1; bra M2;\nM2E:\n"
    // K passes of argmax over unused experts
    " mov.f32 %f5,0f00000000; mov.u32 %r7,0;\n"
    "K1:\n setp.ge.u32 %p3,%r7,%r3; @%p3 bra K1E;\n"
    " mov.f32 %f6,0fF149F2CA; mov.u32 %r8,0; mov.u32 %r5,0;\n"
    "K2:\n setp.ge.u32 %p2,%r5,%r2; @%p2 bra K2E;\n"
    " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f3,[%rd5]; setp.gt.f32 %p5,%f3,%f6; @%p5 mov.f32 %f6,%f3; @%p5 mov.u32 %r8,%r5;\n"
    "K2N:\n add.u32 %r5,%r5,1; bra K2;\nK2E:\n"
    " mul.wide.u32 %rd6,%r8,4; add.u64 %rd6,%rd1,%rd6; mov.f32 %f11,0fF149F2CA; st.global.f32 [%rd6],%f11;"   // taken: the logits are scratch
    " sub.f32 %f7,%f6,%f2; mul.f32 %f7,%f7,0f3FB8AA3B; ex2.approx.f32 %f7,%f7; add.f32 %f5,%f5,%f7;\n"
    " mul.wide.u32 %rd7,%r7,4; add.u64 %rd8,%rd2,%rd7; st.global.u32 [%rd8],%r8; add.u64 %rd9,%rd3,%rd7; st.global.f32 [%rd9],%f7;\n"
    " add.u32 %r7,%r7,1; bra K1;\nK1E:\n"
    " setp.ne.u32 %p6,%r4,0; selp.f32 %f8,%f5,%f4,%p6; div.rn.f32 %f9,%f1,%f8; mov.u32 %r7,0;\n"
    "K3:\n setp.ge.u32 %p3,%r7,%r3; @%p3 bra DONE; mul.wide.u32 %rd7,%r7,4; add.u64 %rd9,%rd3,%rd7; ld.global.f32 %f10,[%rd9]; mul.f32 %f10,%f10,%f9; st.global.f32 [%rd9],%f10;\n"
    " add.u32 %r7,%r7,1; bra K3;\n"
    "DONE:\n ret;\n}\n"
    // X[j] += sum_s w[s]*T[s*D+j]
    ".visible .entry k_wsum(.param .u64 pX, .param .u64 pT, .param .u64 pW, .param .u32 pK, .param .u32 pD)\n{\n"
    " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<8>; .reg .b64 %rd<12>;\n"
    " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pT]; ld.param.u64 %rd3,[pW]; ld.param.u32 %r1,[pK]; ld.param.u32 %r2,[pD];\n"
    " mov.u32 %r3,%ctaid.x; mov.u32 %r4,%ntid.x; mov.u32 %r5,%tid.x; mad.lo.u32 %r6,%r3,%r4,%r5; setp.ge.u32 %p1,%r6,%r2; @%p1 bra DONE;\n"
    " mul.wide.u32 %rd4,%r6,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f1,[%rd4]; mov.u32 %r7,0;\n"
    "L:\n setp.ge.u32 %p2,%r7,%r1; @%p2 bra E; mul.wide.u32 %rd5,%r7,4; add.u64 %rd5,%rd3,%rd5; ld.global.f32 %f2,[%rd5];\n"
    " mad.lo.u32 %r8,%r7,%r2,%r6; mul.wide.u32 %rd6,%r8,4; add.u64 %rd6,%rd2,%rd6; ld.global.f32 %f3,[%rd6]; fma.rn.f32 %f1,%f2,%f3,%f1;\n"
    " add.u32 %r7,%r7,1; bra L;\nE:\n st.global.f32 [%rd4],%f1;\n"
    "DONE:\n ret;\n}\n";

// x (f32) -> int8 blocks of 32 with f32 scale and int32 sum. grid = (nb, n_tok), 32 threads.
static const char* kQuantPTX =
    ".visible .entry k_quant(.param .u64 pX, .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS, .param .u32 pC)\n{\n"
    " .reg .pred %p<4>; .reg .b32 %r<20>; .reg .f32 %f<12>; .reg .b64 %rd<12>;\n"
    " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pXQ]; ld.param.u64 %rd3,[pXD]; ld.param.u64 %rd4,[pXS]; ld.param.u32 %r1,[pC];\n"
    " mov.u32 %r2,%ctaid.x; mov.u32 %r3,%ctaid.y; mov.u32 %r4,%tid.x; shr.u32 %r5,%r1,5;\n"
    " mul.lo.u32 %r6,%r3,%r1; shl.b32 %r7,%r2,5; add.u32 %r6,%r6,%r7; add.u32 %r6,%r6,%r4;\n"   // element index
    " mul.wide.u32 %rd5,%r6,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f1,[%rd5];\n"
    " abs.f32 %f2,%f1;\n"
    " shfl.sync.bfly.b32 %f3,%f2,16,31,0xffffffff; max.f32 %f2,%f2,%f3;\n"
    " shfl.sync.bfly.b32 %f3,%f2,8,31,0xffffffff; max.f32 %f2,%f2,%f3;\n"
    " shfl.sync.bfly.b32 %f3,%f2,4,31,0xffffffff; max.f32 %f2,%f2,%f3;\n"
    " shfl.sync.bfly.b32 %f3,%f2,2,31,0xffffffff; max.f32 %f2,%f2,%f3;\n"
    " shfl.sync.bfly.b32 %f3,%f2,1,31,0xffffffff; max.f32 %f2,%f2,%f3;\n"
    " mul.f32 %f4,%f2,0f3C010204;\n"                                   // d = amax/127
    " setp.gt.f32 %p1,%f2,0f00000000; div.rn.f32 %f5,0f42FE0000,%f2; selp.f32 %f5,%f5,0f00000000,%p1;\n"
    " mul.f32 %f6,%f1,%f5; cvt.rni.s32.f32 %r8,%f6;\n"
    " mov.u32 %r9,%r8;\n"
    " shfl.sync.bfly.b32 %r10,%r9,16,31,0xffffffff; add.s32 %r9,%r9,%r10;\n"
    " shfl.sync.bfly.b32 %r10,%r9,8,31,0xffffffff; add.s32 %r9,%r9,%r10;\n"
    " shfl.sync.bfly.b32 %r10,%r9,4,31,0xffffffff; add.s32 %r9,%r9,%r10;\n"
    " shfl.sync.bfly.b32 %r10,%r9,2,31,0xffffffff; add.s32 %r9,%r9,%r10;\n"
    " shfl.sync.bfly.b32 %r10,%r9,1,31,0xffffffff; add.s32 %r9,%r9,%r10;\n"
    " cvt.u64.u32 %rd6,%r6; add.u64 %rd6,%rd2,%rd6; st.global.s8 [%rd6],%r8;\n"
    " setp.ne.u32 %p2,%r4,0; @%p2 bra DONE;\n"
    " mad.lo.u32 %r11,%r3,%r5,%r2; mul.wide.u32 %rd7,%r11,4;\n"
    " add.u64 %rd8,%rd3,%rd7; st.global.f32 [%rd8],%f4; add.u64 %rd9,%rd4,%rd7; st.global.s32 [%rd9],%r9;\n"
    "DONE:\n ret;\n}\n";

template <class T> static bool load_sym(HMODULE h, T& fp, const char* n) {
    fp = (T)(void*)GetProcAddress(h, n);
    return fp != nullptr;
}


// Set ITHACA_PROF=1 to time every kernel (synchronizes after each launch, slows things down).
static bool g_prof = getenv("ITHACA_PROF") != nullptr;
static double g_pt[32]; static long g_pc[32]; static long g_ptot;
static void prof_end(int slot, double t0) {
    cuStreamSynchronize(nullptr); cuCtxSynchronize();
    g_pt[slot] += now_sec() - t0; g_pc[slot]++;
    if (++g_ptot % 4000 == 0) {
        static const char* nm[32] = {"add", "bias", "silu", "silu_split", "f16", "copy2d", "rmsnorm", "rope", "attn", "kvstore", "attn_comb", "gelu", "gather", "scatter", "attn_gqa",
                                     "", "", "", "", "", "mm_q4", "mm_q8", "mm_q4m"};
        fprintf(stderr, "[prof]");
        for (int i = 0; i < 32; i++) if (g_pc[i]) fprintf(stderr, " %s %.1fus x%ld |", nm[i], g_pt[i] / g_pc[i] * 1e6, g_pc[i]);
        fprintf(stderr, "\n");
    }
}

bool CudaTier::init(std::string& info) {
    HMODULE h = LoadLibraryA("nvcuda.dll");
    if (!h) { info = "no NVIDIA driver (nvcuda.dll) - CPU only"; return false; }
    bool g = load_sym(h, cuInit, "cuInit") && load_sym(h, cuDeviceGet, "cuDeviceGet") &&
             load_sym(h, cuDeviceGetCount, "cuDeviceGetCount") && load_sym(h, cuDeviceGetName, "cuDeviceGetName") &&
             load_sym(h, cuCtxCreate, "cuCtxCreate_v2") && load_sym(h, cuCtxSetCurrent, "cuCtxSetCurrent") &&
             load_sym(h, cuModuleLoadDataEx, "cuModuleLoadDataEx") && load_sym(h, cuModuleGetFunction, "cuModuleGetFunction") &&
             load_sym(h, cuMemAlloc, "cuMemAlloc_v2") && load_sym(h, cuMemFree, "cuMemFree_v2") &&
             load_sym(h, cuMemcpyHtoD, "cuMemcpyHtoD_v2") && load_sym(h, cuMemcpyDtoH, "cuMemcpyDtoH_v2") &&
             load_sym(h, cuMemGetInfo, "cuMemGetInfo_v2") && load_sym(h, cuLaunchKernel, "cuLaunchKernel") && load_sym(h, cuCtxSynchronize, "cuCtxSynchronize") && load_sym(h, cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") &&
             load_sym(h, cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2") && load_sym(h, cuStreamCreate, "cuStreamCreate") && load_sym(h, cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2") && load_sym(h, cuMemsetD8Async, "cuMemsetD8Async") &&
             load_sym(h, cuStreamSynchronize, "cuStreamSynchronize") && load_sym(h, cuMemHostAlloc, "cuMemHostAlloc") &&
             load_sym(h, cuEventCreate, "cuEventCreate") && load_sym(h, cuEventRecord, "cuEventRecord") && load_sym(h, cuEventSynchronize, "cuEventSynchronize");
    has_graphs_ = load_sym(h, cuStreamBeginCapture, "cuStreamBeginCapture_v2") && load_sym(h, cuStreamEndCapture, "cuStreamEndCapture") &&
                  load_sym(h, cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags") && load_sym(h, cuGraphLaunch, "cuGraphLaunch") &&
                  load_sym(h, cuGraphExecDestroy, "cuGraphExecDestroy") && load_sym(h, cuGraphDestroy, "cuGraphDestroy");
    if (!g) { info = "NVIDIA driver too old"; return false; }
    int cnt = 0;
    if (cuInit(0) || cuDeviceGetCount(&cnt) || cnt < 1) { info = "no CUDA device"; return false; }
    CUdevice dev; cuDeviceGet(&dev, 0);
    char nm[256] = {0}; cuDeviceGetName(nm, 255, dev); name = nm;
    if (cuCtxCreate(&ctx_, 0, dev)) { info = "cuCtxCreate failed"; return false; }
    char log[8192] = {0};
    int opts[2] = {5 /*CU_JIT_ERROR_LOG_BUFFER*/, 6 /*CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES*/};
    void* vals[2] = {log, (void*)(uintptr_t)sizeof(log)};
    void* mod = nullptr;
    std::string ptx = ".version 6.0\n.target sm_61\n.address_size 64\n" + gen_kernel("mv_q4", 1) + gen_kernel("mv_q8", 0) + gen_kernel("mv_q4m", 2) + kQuantPTX + gpu_ops_ptx() +
                      gen_kernel_ex("mx_q4", 1) + gen_kernel_ex("mx_q8", 0) + gen_kernel_ex("mx_q4m", 2) + kTopkPTX;
    for (int m = 0; m < 11; m++) ptx += gen_cv(m);
    if (getenv("ITHACA_DUMP_PTX")) { FILE* f = fopen(getenv("ITHACA_DUMP_PTX"), "wb"); if (f) { fwrite(ptx.data(), 1, ptx.size(), f); fclose(f); } }
    if (cuModuleLoadDataEx(&mod, ptx.c_str(), 2, opts, vals)) { info = std::string("PTX JIT failed: ") + log; return false; }
    for (int m = 0; m < 11; m++) cuModuleGetFunction(&cv_[m], mod, kCvNames[m]);
    cuModuleGetFunction(&fe_[K_Q4_0], mod, "mx_q4"); cuModuleGetFunction(&fe_[K_Q8_0], mod, "mx_q8"); cuModuleGetFunction(&fe_[K_Q4M], mod, "mx_q4m");
    cuModuleGetFunction(&ftopk_, mod, "k_topk"); cuModuleGetFunction(&fwsum_, mod, "k_wsum");
    if (cuModuleGetFunction(&fn_[K_Q4_0], mod, "mv_q4") || cuModuleGetFunction(&fn_[K_Q8_0], mod, "mv_q8") || cuModuleGetFunction(&fn_[K_Q4M], mod, "mv_q4m")) {
        info = "kernel lookup failed"; return false;
    }
    static const char* opn[N_OPS] = {"k_add", "k_bias", "k_silu", "k_silu_split", "k_f16", "k_copy2d", "k_rmsnorm", "k_rope", "k_attn", "k_kvstore", "k_attn_comb", "k_gelu", "k_gather", "k_scatter", "k_kvstore_q8"};
    if (cuModuleGetFunction(&quant_, mod, "k_quant")) { info = "kernel lookup failed: k_quant"; return false; }
    for (int i = 0; i < N_OPS; i++)
        if (cuModuleGetFunction(&ops_[i], mod, opn[i])) { info = std::string("kernel lookup failed: ") + opn[i]; return false; }
    if (cuStreamCreate(&st_, 1 /*non-blocking*/)) { info = "stream create failed"; return false; }
    cuStreamCreate(&up_, 1);
    ok_ = true;
    info = "GPU: " + name;
    return true;
}

void CudaTier::bind_thread() { if (ok_) cuCtxSetCurrent(ctx_); }

bool CudaTier::mem_info(size_t& f, size_t& t) {
    if (!ok_) return false;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuMemGetInfo(&f, &t) == 0;
}

uint64_t CudaTier::alloc(size_t bytes) {
    if (!ok_) return 0;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    CUdeviceptr p = 0;
    if (cuMemAlloc(&p, bytes ? bytes : 4)) return 0;
    return p;
}

bool CudaTier::htod(uint64_t dst, const void* src, size_t bytes) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuMemcpyHtoD(dst, src, bytes) == 0;
}

bool CudaTier::dtoh(void* dst, uint64_t src, size_t bytes) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    if (cuMemcpyDtoHAsync(dst, src, bytes, st_)) return false;
    return cuStreamSynchronize(st_) == 0;
}

bool CudaTier::htod_async(uint64_t dst, const void* src, size_t bytes) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuMemcpyHtoDAsync(dst, src, bytes, st_) == 0;
}

bool CudaTier::begin_capture() {
    if (!has_graphs_ || g_prof || getenv("ITHACA_NO_GRAPH")) return false;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuStreamBeginCapture(st_, 2 /*relaxed*/) == 0;
}

void* CudaTier::end_capture() {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    void* g = nullptr; void* e = nullptr;
    if (cuStreamEndCapture(st_, &g) || !g) return nullptr;
    if (cuGraphInstantiateWithFlags(&e, g, 0)) e = nullptr;
    cuGraphDestroy(g);
    return e;
}

bool CudaTier::graph_launch(void* exec) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuGraphLaunch(exec, st_) == 0;
}

void CudaTier::graph_destroy(void* exec) {
    if (!exec) return;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    cuGraphExecDestroy(exec);
}

bool CudaTier::launch(Op op, unsigned gx, unsigned gy, unsigned block, std::initializer_list<void*> args) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    void* a[24]; int n = 0;
    for (void* p : args) a[n++] = p;
    double t0 = g_prof ? now_sec() : 0;
    bool ok = cuLaunchKernel(ops_[op], gx, gy, 1, block, 1, 1, 0, st_, a, nullptr) == 0;
    if (g_prof) prof_end(op, t0);
    return ok;
}

bool CudaTier::ew(Op op, unsigned n, std::initializer_list<void*> args) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    void* a[24]; int k = 0;
    for (void* p : args) a[k++] = p;
    a[k++] = &n;
    double t0 = g_prof ? now_sec() : 0;
    bool ok = cuLaunchKernel(ops_[op], (n + 255) / 256, 1, 1, 256, 1, 1, 0, st_, a, nullptr) == 0;
    if (g_prof) prof_end(op, t0);
    return ok;
}

void* CudaTier::load_module(const std::string& ptx, std::string& log) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    char buf[16384] = {0};
    int opts[2] = {5, 6};
    void* vals[2] = {buf, (void*)(uintptr_t)sizeof(buf)};
    void* mod = nullptr;
    if (cuModuleLoadDataEx(&mod, ptx.c_str(), 2, opts, vals)) { log = buf; return nullptr; }
    return mod;
}

void* CudaTier::function(void* mod, const char* name) {
    void* f = nullptr;
    return cuModuleGetFunction(&f, mod, name) ? nullptr : f;
}

bool CudaTier::launch_raw(void* fn, unsigned gx, unsigned gy, unsigned gz, unsigned block, unsigned shmem, void** args) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuLaunchKernel(fn, gx, gy, gz, block, 1, 1, shmem, st_, args, nullptr) == 0;
}

bool CudaTier::dtod(uint64_t dst, uint64_t src, size_t bytes) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuMemcpyDtoDAsync(dst, src, bytes, st_) == 0;
}

bool CudaTier::zero(uint64_t dst, size_t bytes) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuMemsetD8Async(dst, 0, bytes, st_) == 0;
}

bool CudaTier::sync() {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    return cuStreamSynchronize(st_) == 0;
}

bool CudaTier::reserve_act(size_t elems) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    if (elems <= cap_q_) return true;
    if (xq_) { cuMemFree(xq_); cuMemFree(xd_); cuMemFree(xs_); }
    xq_ = xd_ = xs_ = 0; cap_q_ = 0;
    if (cuMemAlloc(&xq_, elems) || cuMemAlloc(&xd_, elems / 32 * 4 + 4) || cuMemAlloc(&xs_, elems / 32 * 4 + 4)) return false;
    cap_q_ = elems;
    return true;
}

bool CudaTier::matmul_dev(uint64_t W, uint64_t qoff, int kind, uint64_t dX, int n, int cols, int rows, uint64_t dY) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    if ((size_t)n * cols > cap_q_ && !reserve_act((size_t)n * cols)) return false;
    double t0 = g_prof ? now_sec() : 0;
    unsigned c = (unsigned)cols, r = (unsigned)rows, un = (unsigned)n;
    void* qa[5] = {&dX, &xq_, &xd_, &xs_, &c};
    if (cuLaunchKernel(quant_, c / 32, un, 1, 32, 1, 1, 0, st_, qa, nullptr)) return false;
    uint64_t Q = W + qoff;
    void* args[9] = {&W, &Q, &xq_, &xd_, &xs_, &dY, &c, &r, &un};
    bool ok = cuLaunchKernel(fn_[kind], (r + 8 * kRows - 1) / (8 * kRows), (un + kTokTile - 1) / kTokTile, 1, 256, 1, 1, 0, st_, args, nullptr) == 0;
    if (g_prof) prof_end(20 + kind, t0);
    return ok;
}

bool CudaTier::matmul_experts(uint64_t W, uint64_t qoff, int kind, uint64_t ids, int slots, uint64_t dX, int xrows, int xstep, int cols, int R, uint64_t dY) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    if (!fe_[kind]) return false;
    if ((size_t)xrows * cols > cap_q_ && !reserve_act((size_t)xrows * cols)) return false;
    unsigned c = (unsigned)cols, r = (unsigned)R, xn = (unsigned)xrows, xs = (unsigned)xstep;
    void* qa[5] = {&dX, &xq_, &xd_, &xs_, &c};
    if (cuLaunchKernel(quant_, c / 32, xn, 1, 32, 1, 1, 0, st_, qa, nullptr)) return false;
    uint64_t Q = W + qoff;
    void* args[10] = {&W, &Q, &xq_, &xd_, &xs_, &dY, &c, &r, &ids, &xs};
    return cuLaunchKernel(fe_[kind], (r + 7) / 8, (unsigned)slots, 1, 256, 1, 1, 0, st_, args, nullptr) == 0;
}
bool CudaTier::topk(uint64_t logits, int E, int K, bool norm, float scale, uint64_t ids, uint64_t w) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    unsigned e = E, k = K, nm = norm ? 1 : 0;
    void* a[7] = {&logits, &e, &k, &nm, &scale, &ids, &w};
    return ftopk_ && cuLaunchKernel(ftopk_, 1, 1, 1, 32, 1, 1, 0, st_, a, nullptr) == 0;
}
bool CudaTier::wsum(uint64_t X, uint64_t T, uint64_t w, int K, int D) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    unsigned k = K, d = D;
    void* a[5] = {&X, &T, &w, &k, &d};
    return fwsum_ && cuLaunchKernel(fwsum_, (d + 255) / 256, 1, 1, 256, 1, 1, 0, st_, a, nullptr) == 0;
}

uint64_t CudaTier::alloc_upload(const void* host, size_t bytes) {
    if (!ok_) return 0;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    CUdeviceptr p = 0;
    CUresult e = cuMemAlloc(&p, bytes);
    if (e) { fprintf(stderr, "[gpu] alloc %zu MB failed (%d)\n", bytes >> 20, e); return 0; }
    e = cuMemcpyHtoD(p, host, bytes);
    if (e) { fprintf(stderr, "[gpu] upload failed (%d)\n", e); cuMemFree(p); return 0; }
    return p;
}

// GGUF type ids (see gguf.h) -> converter mode
static int cv_mode(int gtype) {
    switch (gtype) { case 2: return 0; case 8: return 1; case 12: return 2; case 14: return 3; case 13: return 4; case 1: return 5; case 30: return 6; case 0: return 7;
                     case 3: return 8; case 6: return 9; case 7: return 10; }
    return -1;
}
bool CudaTier::can_load(int gtype) { return cv_mode(gtype) >= 0; }
size_t CudaTier::gpu_bytes(int gtype, int rows, int cols) {
    size_t nblk = (size_t)rows * (cols / 32);
    int m = cv_mode(gtype);
    if (m == 0) return (nblk * 4 + 255) / 256 * 256 + nblk * 16;
    if (m == 2 || m == 8) return (nblk * 8 + 255) / 256 * 256 + nblk * 16;
    return (nblk * 4 + 255) / 256 * 256 + nblk * 32;
}

uint64_t CudaTier::load_weight(void* file, uint64_t off, size_t nbytes, int gtype, int rows, int cols, int& kind, uint64_t& qoff) {
    const int mode = cv_mode(gtype);
    if (!ok_ || mode < 0 || !cv_[mode] || cols % 32) return 0;
    static const int usrc[11] = {18, 34, 144, 210, 176, 64, 64, 128, 20, 22, 24}, ublk[11] = {1, 1, 8, 8, 8, 1, 1, 1, 1, 1, 1};
    const size_t kChunk = 8u << 20;
    const size_t nblk = (size_t)rows * (cols / 32), units = nblk / ublk[mode];
    if (units * usrc[mode] != nbytes) return 0;
    const bool q4m = mode == 2 || mode == 8;
    kind = mode == 0 ? K_Q4_0 : q4m ? K_Q4M : K_Q8_0;
    qoff = (nblk * (q4m ? 8 : 4) + 255) / 256 * 256;
    uint64_t dst = alloc(gpu_bytes(gtype, rows, cols));
    if (!dst) return 0;
    {
        std::lock_guard<std::recursive_mutex> l(m_);
        cuCtxSetCurrent(ctx_);
        for (int i = 0; i < kSlots; i++) {
            if (!pin_[i] && cuMemHostAlloc(&pin_[i], kChunk, 0)) pin_[i] = nullptr;
            if (!ev_[i]) cuEventCreate(&ev_[i], 2);
            if (!raw_[i] && cuMemAlloc(&raw_[i], kChunk)) raw_[i] = 0;
            if (!pin_[i] || !raw_[i]) { cuMemFree(dst); return 0; }
        }
        if (!up_) { cuMemFree(dst); return 0; }
    }
    uint64_t S = dst, Q = dst + qoff, M = dst + nblk * 4;
    // Pipeline: up to kSlots chunk reads in flight (queue depth for SSDs, parallel copies out of the
    // OS cache when warm) while earlier chunks upload and convert on the GPU.
    const size_t per = kChunk / usrc[mode], nchunks = (units + per - 1) / per;
    auto read_chunk = [this, file, off, per, units, mode](size_t c, int slot) -> bool {
        size_t u0 = c * per, bytes = std::min(per, units - u0) * usrc[mode], done = 0;
        uint64_t pos = off + u0 * usrc[mode];
        while (done < bytes) {
            OVERLAPPED ov{}; ov.Offset = (DWORD)(pos + done); ov.OffsetHigh = (DWORD)((pos + done) >> 32);
            DWORD got = 0;
            if (!ReadFile((HANDLE)file, (char*)pin_[slot] + done, (DWORD)(bytes - done), &got, &ov) || !got) return false;
            done += got;
        }
        return true;
    };
    std::future<bool> rd[kSlots];
    for (size_t c = 0; c < nchunks && c < (size_t)kSlots; c++) {
        { std::lock_guard<std::recursive_mutex> l(m_); cuCtxSetCurrent(ctx_); cuEventSynchronize(ev_[c]); }
        rd[c] = std::async(std::launch::async, read_chunk, c, (int)c);
    }
    bool ok = true;
    for (size_t c = 0; c < nchunks; c++) {
        const int i = (int)(c % kSlots);
        ok = rd[i].get() && ok;
        if (!ok) continue;   // drain the remaining reads before bailing out
        size_t u0 = c * per, nu = std::min(per, units - u0), bytes = nu * usrc[mode];
        {
            std::lock_guard<std::recursive_mutex> l(m_);
            cuCtxSetCurrent(ctx_);
            unsigned N = (unsigned)(nu * ublk[mode]), B0 = (unsigned)(u0 * ublk[mode]);
            void* a[6] = {&raw_[i], &S, &Q, &M, &N, &B0};
            ok = cuMemcpyHtoDAsync(raw_[i], pin_[i], bytes, up_) == 0 && cuEventRecord(ev_[i], up_) == 0 &&
                 cuLaunchKernel(cv_[mode], (N + 7) / 8, 1, 1, 256, 1, 1, 0, up_, a, nullptr) == 0;
        }
        size_t nx = c + kSlots;
        if (ok && nx < nchunks) {
            // the slot's pinned buffer is free once its copy finished; the GPU side keeps going meanwhile
            { std::lock_guard<std::recursive_mutex> l(m_); cuCtxSetCurrent(ctx_); cuEventSynchronize(ev_[i]); }
            rd[i] = std::async(std::launch::async, read_chunk, nx, i);
        }
    }
    for (auto& f : rd) if (f.valid()) f.wait();
    {
        std::lock_guard<std::recursive_mutex> l(m_);
        cuCtxSetCurrent(ctx_);
        if (cuStreamSynchronize(up_)) ok = false;
    }
    if (!ok) { free(dst); return 0; }
    return dst;
}

void CudaTier::free(uint64_t p) {
    if (!ok_ || !p) return;
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    cuMemFree(p);
}

bool CudaTier::matmul(uint64_t W, uint64_t qoff, int kind, const float* X, int n, int cols, int rows, float* Y) {
    std::lock_guard<std::recursive_mutex> l(m_);
    cuCtxSetCurrent(ctx_);
    size_t bx = (size_t)n * cols * 4, by = (size_t)n * rows * 4;
    if (bx > cap_x_) { if (dx_) cuMemFree(dx_); cap_x_ = 0; if (cuMemAlloc(&dx_, bx)) return false; cap_x_ = bx; }
    if (by > cap_y_) { if (dy_) cuMemFree(dy_); cap_y_ = 0; if (cuMemAlloc(&dy_, by)) return false; cap_y_ = by; }
    if (cuMemcpyHtoDAsync(dx_, X, bx, st_)) return false;
    if (!matmul_dev(W, qoff, kind, dx_, n, cols, rows, dy_)) return false;
    return cuMemcpyDtoHAsync(Y, dy_, by, st_) == 0 && cuStreamSynchronize(st_) == 0;
}
