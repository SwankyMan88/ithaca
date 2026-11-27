// Fused decode kernels: each one does the work of several small launches, so a token spends its time reading
// weights instead of starting kernels. They repeat the exact float operations of the kernels they replace
// (same reduction order, same roundings), so the output stays the same token for token.
#include <cstdio>
#include <string>

std::string fuse_ops_ptx();

namespace {
// Block-wide reduction (256 threads = 8 warps) of register R; result broadcast to all threads. Same order as k_rmsnorm.
std::string red(const std::string& R, const char* op) {
    std::string s;
    char b[200];
    for (int o = 16; o >= 1; o /= 2) {
        snprintf(b, sizeof b, " shfl.sync.bfly.b32 %%ft,%s,%d,31,0xffffffff; %s %s,%s,%%ft;\n", R.c_str(), o, op, R.c_str(), R.c_str());
        s += b;
    }
    s += " @%plane0 st.shared.f32 [%rdw]," + R + ";\n bar.sync 0;\n ld.shared.f32 " + R + ",[%rdr];\n";
    for (int k = 1; k < 8; k++) {
        snprintf(b, sizeof b, " ld.shared.f32 %%ft,[%%rdr+%d]; %s %s,%s,%%ft;\n", k * 4, op, R.c_str(), R.c_str());
        s += b;
    }
    s += " bar.sync 0;\n";
    return s;
}
const char* kPro =
    " .reg .pred %plane0; .reg .f32 %ft; .reg .b64 %rdw,%rdr; .reg .b32 %rlane,%rwarp;\n"
    " .shared .align 4 .f32 red[8];\n"
    " mov.u32 %rlane,%tid.x; shr.u32 %rwarp,%rlane,5; and.b32 %rlane,%rlane,31;\n"
    " setp.eq.u32 %plane0,%rlane,0;\n"
    " mov.u64 %rdr,red; mul.wide.u32 %rdw,%rwarp,4; add.u64 %rdw,%rdr,%rdw;\n";
// int8 block quantization of %qv over the 32 lanes of a warp (as k_quant): q in %qi, d (amax/127) in %qd, sum of q in %qs
const char* kQuant32 =
    " abs.f32 %qa,%qv;\n"
    " shfl.sync.bfly.b32 %qt,%qa,16,31,0xffffffff; max.f32 %qa,%qa,%qt;\n"
    " shfl.sync.bfly.b32 %qt,%qa,8,31,0xffffffff; max.f32 %qa,%qa,%qt;\n"
    " shfl.sync.bfly.b32 %qt,%qa,4,31,0xffffffff; max.f32 %qa,%qa,%qt;\n"
    " shfl.sync.bfly.b32 %qt,%qa,2,31,0xffffffff; max.f32 %qa,%qa,%qt;\n"
    " shfl.sync.bfly.b32 %qt,%qa,1,31,0xffffffff; max.f32 %qa,%qa,%qt;\n"
    " mul.f32 %qd,%qa,0f3C010204;\n"
    " setp.gt.f32 %qp,%qa,0f00000000; div.rn.f32 %qr,0f42FE0000,%qa; selp.f32 %qr,%qr,0f00000000,%qp;\n"
    " mul.f32 %qr,%qv,%qr; cvt.rni.s32.f32 %qi,%qr; mov.u32 %qs,%qi;\n"
    " shfl.sync.bfly.b32 %qu,%qs,16,31,0xffffffff; add.s32 %qs,%qs,%qu;\n"
    " shfl.sync.bfly.b32 %qu,%qs,8,31,0xffffffff; add.s32 %qs,%qs,%qu;\n"
    " shfl.sync.bfly.b32 %qu,%qs,4,31,0xffffffff; add.s32 %qs,%qs,%qu;\n"
    " shfl.sync.bfly.b32 %qu,%qs,2,31,0xffffffff; add.s32 %qs,%qs,%qu;\n"
    " shfl.sync.bfly.b32 %qu,%qs,1,31,0xffffffff; add.s32 %qs,%qs,%qu;\n";
const char* kQuantRegs = " .reg .f32 %qa,%qt,%qd,%qr,%qv; .reg .pred %qp; .reg .b32 %qi,%qs,%qu;\n";
}

// ---- k_normq<K>: one CTA (256 threads) per token row, D <= 256 K. Thread t owns elements t + 256 k (k < K), kept in
// registers through every phase, so all loads go out at once instead of one loop trip at a time.
//  flags & 1: X += res * T first (flags & 2: T rmsnorm'd with Wp before that, in place of a k_rmsnorm on T)
//  then XN = rmsnorm(X) * Wn (stored when flags & 4) and quantized into XQ/XD/XS like k_quant does.
// The sums run over k in order, exactly like the loops of k_rmsnorm, and block b's 32 elements are lanes of warp b % 8.
std::string gen_normq(int K) {
    std::string s;
    char b[400];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); s += b; s += "\n"; };
    L(".visible .entry k_normq%d(.param .u64 pX, .param .u64 pT, .param .u64 pWp, .param .u64 pWn, .param .u64 pXN,"
      " .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS, .param .u32 pD, .param .f32 pE, .param .f32 pR, .param .u32 pF)\n{", K);
    L(" .reg .pred %%p<8>; .reg .pred %%pk<%d>; .reg .b32 %%r<24>; .reg .f32 %%f<24>; .reg .b64 %%rd<24>; .reg .f32 %%x<%d>; .reg .f32 %%t<%d>; .reg .f32 %%w<%d>; .reg .f32 %%v<%d>;", K, K, K, K, K);
    s += kPro; s += kQuantRegs;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pT]; ld.param.u64 %rd3,[pWp]; ld.param.u64 %rd4,[pWn]; ld.param.u64 %rd5,[pXN];\n"
         " ld.param.u64 %rd6,[pXQ]; ld.param.u64 %rd7,[pXD]; ld.param.u64 %rd8,[pXS]; ld.param.u32 %r1,[pD]; ld.param.f32 %f10,[pE]; ld.param.f32 %f11,[pR]; ld.param.u32 %r20,[pF];\n"
         " mov.u32 %r2,%ctaid.x; mov.u32 %r3,%tid.x;\n"
         " mul.wide.u32 %rd9,%r2,%r1; shl.b64 %rd9,%rd9,2; add.u64 %rd10,%rd1,%rd9; add.u64 %rd11,%rd2,%rd9; add.u64 %rd12,%rd5,%rd9;\n"
         " and.b32 %r21,%r20,1; setp.ne.u32 %p5,%r21,0; and.b32 %r21,%r20,2; setp.ne.u32 %p6,%r21,0; and.b32 %r21,%r20,4; setp.ne.u32 %p7,%r21,0;\n"
         " and.pred %p4,%p5,%p6; cvt.rn.f32.u32 %f5,%r1; mov.f32 %f12,0f3F800000;\n"
         " mul.wide.u32 %rd13,%r3,4; add.u64 %rd14,%rd10,%rd13; add.u64 %rd15,%rd11,%rd13; add.u64 %rd16,%rd3,%rd13; add.u64 %rd17,%rd4,%rd13; add.u64 %rd18,%rd12,%rd13;\n";
    for (int k = 0; k < K; k++)   // element t + 256 k exists?
        L(" add.u32 %%r4,%%r3,%d; setp.lt.u32 %%pk%d,%%r4,%%r1; mov.f32 %%x%d,0f00000000; mov.f32 %%t%d,0f00000000; mov.f32 %%w%d,0f00000000; mov.f32 %%v%d,0f00000000;", 256 * k, k, k, k, k, k);
    for (int k = 0; k < K; k++) L(" @%%pk%d ld.global.f32 %%x%d,[%%rd14+%d]; @%%pk%d ld.global.nc.f32 %%v%d,[%%rd17+%d];", k, k, 1024 * k, k, k, 1024 * k);   // X and Wn
    s += " @!%p5 bra NOT;\n";
    for (int k = 0; k < K; k++) L(" @%%pk%d ld.global.f32 %%t%d,[%%rd15+%d];", k, k, 1024 * k);
    s += " @!%p6 bra NOA;\n";
    for (int k = 0; k < K; k++) L(" @%%pk%d ld.global.nc.f32 %%w%d,[%%rd16+%d];", k, k, 1024 * k);
    s += " mov.f32 %f1,0f00000000;\n";
    for (int k = 0; k < K; k++) L(" @%%pk%d fma.rn.f32 %%f1,%%t%d,%%t%d,%%f1;", k, k, k);
    s += red("%f1", "add.f32");
    s += " div.rn.f32 %f3,%f1,%f5; add.f32 %f3,%f3,%f10; rsqrt.approx.f32 %f12,%f3;\n";
    for (int k = 0; k < K; k++) L(" mul.f32 %%t%d,%%t%d,%%f12; mul.f32 %%t%d,%%t%d,%%w%d;", k, k, k, k, k);
    s += "NOA:\n";
    for (int k = 0; k < K; k++) L(" fma.rn.f32 %%x%d,%%t%d,%%f11,%%x%d; @%%pk%d st.global.f32 [%%rd14+%d],%%x%d;", k, k, k, k, 1024 * k, k);
    s += "NOT:\n";
    s += " mov.f32 %f1,0f00000000;\n";
    for (int k = 0; k < K; k++) L(" @%%pk%d fma.rn.f32 %%f1,%%x%d,%%x%d,%%f1;", k, k, k);
    s += red("%f1", "add.f32");
    s += " div.rn.f32 %f3,%f1,%f5; add.f32 %f13,%f3,%f10; rsqrt.approx.f32 %f13,%f13;\n"
         " shr.u32 %r5,%r1,5; mul.lo.u32 %r10,%r2,%r5; mul.lo.u32 %r11,%r2,%r1; add.u32 %r11,%r11,%r3;\n";   // nb, first block of the row, XQ index
    for (int k = 0; k < K; k++) {
        L(" @!%%pk%d bra QE%d;", k, k);   // uniform per warp: D is a multiple of 32
        L(" mul.f32 %%x%d,%%x%d,%%f13; mul.f32 %%qv,%%x%d,%%v%d; @%%p7 st.global.f32 [%%rd18+%d],%%qv;", k, k, k, k, 1024 * k);
        s += kQuant32;
        L(" add.u32 %%r8,%%r11,%d; cvt.u64.u32 %%rd19,%%r8; add.u64 %%rd19,%%rd6,%%rd19; st.global.s8 [%%rd19],%%qi;", 256 * k);
        L(" @!%%plane0 bra QE%d; add.u32 %%r9,%%r10,%%rwarp; add.u32 %%r9,%%r9,%d; mul.wide.u32 %%rd20,%%r9,4; add.u64 %%rd21,%%rd7,%%rd20; st.global.f32 [%%rd21],%%qd; add.u64 %%rd21,%%rd8,%%rd20; st.global.s32 [%%rd21],%%qs;", k, 8 * k);
        L("QE%d:", k);
    }
    s += " ret;\n}\n";
    return s;
}

std::string fuse_ops_ptx() {
    std::string s;
    s += gen_normq(8) + gen_normq(16) + gen_normq(32);

    // ---- k_qkpost: everything between the q/k/v matmuls and attention, one CTA (256 threads) per head.
    // grid = (n_head + 2 n_kv, n_tok). Per head: + bias, rmsnorm over head_dim (q/k), RoPE (q/k), then k/v into
    // the cache (f16, or int8 with one scale per head when Ks != 0). Needs head_dim <= 256.
    // flags: 1 neox pairing, 2 apply RoPE
    s += ".visible .entry k_qkpost(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pBq, .param .u64 pBk, .param .u64 pBv,"
         " .param .u64 pQn, .param .u64 pKn, .param .u64 pCS, .param .u64 pKc, .param .u64 pVc, .param .u64 pKs, .param .u64 pVs, .param .u64 pPos,"
         " .param .u32 pNH, .param .u32 pNKV, .param .u32 pHD, .param .u32 pHalf, .param .f32 pE, .param .u32 pF)\n{\n"
         " .reg .pred %p<16>; .reg .b32 %r<40>; .reg .f32 %f<24>; .reg .b64 %rd<40>; .reg .b16 %h1;\n"
         " .shared .align 4 .f32 sh[256];\n";
    s += kPro;
    s += " ld.param.u64 %rd1,[pQ]; ld.param.u64 %rd2,[pK]; ld.param.u64 %rd3,[pV]; ld.param.u64 %rd4,[pBq]; ld.param.u64 %rd5,[pBk]; ld.param.u64 %rd6,[pBv];\n"
         " ld.param.u64 %rd7,[pQn]; ld.param.u64 %rd8,[pKn]; ld.param.u64 %rd9,[pCS]; ld.param.u64 %rd10,[pKc]; ld.param.u64 %rd11,[pVc];\n"
         " ld.param.u64 %rd12,[pKs]; ld.param.u64 %rd13,[pVs]; ld.param.u64 %rd14,[pPos];\n"
         " ld.param.u32 %r4,[pNH]; ld.param.u32 %r5,[pNKV]; ld.param.u32 %r6,[pHD]; ld.param.u32 %r7,[pHalf]; ld.param.f32 %f10,[pE]; ld.param.u32 %r20,[pF];\n"
         " mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y; mov.u32 %r3,%tid.x;\n"
         " ld.global.u32 %r8,[%rd14]; add.u32 %r8,%r8,%r2;\n"                                 // cache position of this token
         " mul.lo.u32 %r9,%r4,%r6; mul.lo.u32 %r10,%r5,%r6;\n"                                // qd, kvd
         " setp.lt.u32 %p1,%r1,%r4; add.u32 %r11,%r4,%r5; setp.lt.u32 %p2,%r1,%r11;\n"         // p1: q head, p2: q or k head
         " sub.u32 %r12,%r1,%r4; sub.u32 %r13,%r12,%r5; selp.u32 %r12,%r12,%r13,%p2; selp.u32 %r12,%r1,%r12,%p1;\n"   // head index within its kind
         " selp.u32 %r14,%r9,%r10,%p1; mul.lo.u32 %r15,%r2,%r14; mad.lo.u32 %r15,%r12,%r6,%r15;\n"                    // element offset of the head
         " selp.u64 %rd15,%rd2,%rd3,%p2; selp.u64 %rd15,%rd1,%rd15,%p1;\n"                     // src
         " selp.u64 %rd16,%rd5,%rd6,%p2; selp.u64 %rd16,%rd4,%rd16,%p1;\n"                     // bias
         " mov.u64 %rd17,0; selp.u64 %rd17,%rd8,%rd17,%p2; selp.u64 %rd17,%rd7,%rd17,%p1;\n"   // norm weight (none for v)
         " setp.lt.u32 %p3,%r3,%r6;\n"
         " mul.wide.u32 %rd18,%r15,4; add.u64 %rd18,%rd15,%rd18; mul.wide.u32 %rd19,%r3,4; add.u64 %rd20,%rd18,%rd19;\n"
         " mov.f32 %f1,0f00000000; @%p3 ld.global.f32 %f1,[%rd20];\n"
         " setp.ne.u64 %p4,%rd16,0; @!%p4 bra NBIAS;\n"
         " mul.lo.u32 %r16,%r12,%r6; add.u32 %r16,%r16,%r3; mul.wide.u32 %rd21,%r16,4; add.u64 %rd21,%rd16,%rd21;\n"
         " @%p3 ld.global.f32 %f2,[%rd21]; @%p3 add.f32 %f1,%f1,%f2;\n"
         "NBIAS:\n"
         " setp.ne.u64 %p5,%rd17,0; @!%p5 bra NNORM;\n"
         " mov.f32 %f3,0f00000000; @%p3 fma.rn.f32 %f3,%f1,%f1,%f3;\n";
    s += red("%f3", "add.f32");
    s += " cvt.rn.f32.u32 %f5,%r6; div.rn.f32 %f4,%f3,%f5; add.f32 %f4,%f4,%f10; rsqrt.approx.f32 %f4,%f4;\n"
         " add.u64 %rd22,%rd17,%rd19; mov.f32 %f6,0f00000000; @%p3 ld.global.f32 %f6,[%rd22]; mul.f32 %f1,%f1,%f4; mul.f32 %f1,%f1,%f6;\n"
         "NNORM:\n"
         " and.b32 %r21,%r20,2; setp.ne.u32 %p6,%r21,0; and.pred %p6,%p6,%p2; @!%p6 bra NROPE;\n"   // rope on q and k heads
         " mov.u64 %rd23,sh; add.u64 %rd24,%rd23,%rd19; st.shared.f32 [%rd24],%f1; bar.sync 0;\n"
         " and.b32 %r21,%r20,1; setp.ne.u32 %p7,%r21,0; setp.lt.u32 %p8,%r3,%r7;\n"
         " shl.b32 %r22,%r3,1; add.u32 %r23,%r22,1; add.u32 %r24,%r3,%r7;\n"
         " selp.u32 %r25,%r3,%r22,%p7; selp.u32 %r26,%r24,%r23,%p7;\n"                         // the pair (a, b)
         " mul.wide.u32 %rd25,%r25,4; add.u64 %rd25,%rd23,%rd25; mul.wide.u32 %rd26,%r26,4; add.u64 %rd26,%rd23,%rd26;\n"
         " mad.lo.u32 %r27,%r2,%r7,%r3; mul.wide.u32 %rd27,%r27,8; add.u64 %rd27,%rd9,%rd27;\n"
         " @%p8 ld.global.f32 %f11,[%rd27]; @%p8 ld.global.f32 %f12,[%rd27+4];\n"
         " @%p8 ld.shared.f32 %f13,[%rd25]; @%p8 ld.shared.f32 %f14,[%rd26];\n"
         " bar.sync 0;\n"
         " @%p8 mul.rn.f32 %f15,%f13,%f11; @%p8 mul.rn.f32 %f16,%f14,%f12; @%p8 sub.rn.f32 %f15,%f15,%f16;\n"
         " @%p8 mul.rn.f32 %f17,%f13,%f12; @%p8 fma.rn.f32 %f17,%f14,%f11,%f17;\n"
         " @%p8 st.shared.f32 [%rd25],%f15; @%p8 st.shared.f32 [%rd26],%f17;\n"
         " bar.sync 0; ld.shared.f32 %f1,[%rd24];\n"
         "NROPE:\n"
         " @%p3 st.global.f32 [%rd20],%f1;\n"
         " @%p1 bra DONE;\n"                                                                    // q heads are done
         " selp.u64 %rd28,%rd10,%rd11,%p2; selp.u64 %rd29,%rd12,%rd13,%p2;\n"                    // cache, scales
         " mul.lo.u32 %r28,%r8,%r10; mad.lo.u32 %r28,%r12,%r6,%r28; add.u32 %r28,%r28,%r3;\n"   // pos*kvd + h*hd + i
         " setp.ne.u64 %p9,%rd12,0; @%p9 bra Q8;\n"
         " cvt.rn.f16.f32 %h1,%f1; mul.wide.u32 %rd30,%r28,2; add.u64 %rd30,%rd28,%rd30; @%p3 st.global.b16 [%rd30],%h1;\n"
         " bra DONE;\n"
         "Q8:\n"
         " abs.f32 %f7,%f1;\n";
    s += red("%f7", "max.f32");
    s += " mul.f32 %f8,%f7,0f3C010204; setp.gt.f32 %p10,%f7,0f00000000; div.rn.f32 %f9,0f42FE0000,%f7; selp.f32 %f9,%f9,0f00000000,%p10;\n"
         " mul.f32 %f18,%f1,%f9; cvt.rni.s32.f32 %r29,%f18; cvt.u64.u32 %rd31,%r28; add.u64 %rd31,%rd28,%rd31; @%p3 st.global.s8 [%rd31],%r29;\n"
         " setp.ne.u32 %p11,%r3,0; @%p11 bra DONE;\n"
         " mad.lo.u32 %r30,%r8,%r5,%r12; mul.wide.u32 %rd32,%r30,4; add.u64 %rd32,%rd29,%rd32; st.global.f32 [%rd32],%f8;\n"
         "DONE:\n ret;\n}\n";

    // ---- k_attn_combq: k_attn_comb, plus the output quantized into the int8 input of the o projection.
    // grid = (n_head, n_tok), head_dim threads (a multiple of 32).
    s += ".visible .entry k_attn_combq(.param .u64 pPM, .param .u64 pPL, .param .u64 pPO, .param .u64 pO, .param .u32 pNH, .param .u32 pHD, .param .u32 pSPL,"
         " .param .u64 pXQ, .param .u64 pXD, .param .u64 pXS)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .b64 %rd<16>; .reg .b32 %rl;\n";
    s += kQuantRegs;
    s += " ld.param.u64 %rd1,[pPM]; ld.param.u64 %rd2,[pPL]; ld.param.u64 %rd3,[pPO]; ld.param.u64 %rd4,[pO];\n"
         " ld.param.u32 %r1,[pNH]; ld.param.u32 %r2,[pHD]; ld.param.u32 %r3,[pSPL];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ctaid.y; mov.u32 %r6,%tid.x;\n"
         " setp.ge.u32 %p1,%r6,%r2; @%p1 bra DONE;\n"
         " mad.lo.u32 %r7,%r5,%r1,%r4; mul.lo.u32 %r8,%r7,%r3;\n"
         " mov.f32 %f1,0fF149F2CA; mov.u32 %r9,0;\n"
         "M1:\n add.u32 %r10,%r8,%r9; mul.wide.u32 %rd5,%r10,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f2,[%rd5]; max.f32 %f1,%f1,%f2;\n"
         " add.u32 %r9,%r9,1; setp.lt.u32 %p2,%r9,%r3; @%p2 bra M1;\n"
         " mov.f32 %f3,0f00000000; mov.f32 %f4,0f00000000; mov.u32 %r9,0;\n"
         "M2:\n add.u32 %r10,%r8,%r9; mul.wide.u32 %rd5,%r10,4; add.u64 %rd6,%rd1,%rd5; ld.global.f32 %f2,[%rd6];\n"
         " sub.f32 %f2,%f2,%f1; mul.f32 %f2,%f2,0f3FB8AA3B; ex2.approx.f32 %f2,%f2;\n"
         " add.u64 %rd7,%rd2,%rd5; ld.global.f32 %f5,[%rd7]; fma.rn.f32 %f3,%f2,%f5,%f3;\n"
         " mad.lo.u32 %r11,%r10,%r2,%r6; mul.wide.u32 %rd8,%r11,4; add.u64 %rd8,%rd3,%rd8; ld.global.f32 %f6,[%rd8]; fma.rn.f32 %f4,%f2,%f6,%f4;\n"
         " add.u32 %r9,%r9,1; setp.lt.u32 %p2,%r9,%r3; @%p2 bra M2;\n"
         " div.rn.f32 %f4,%f4,%f3;\n"
         " mad.lo.u32 %r12,%r7,%r2,%r6; mul.wide.u32 %rd9,%r12,4; add.u64 %rd9,%rd4,%rd9; st.global.f32 [%rd9],%f4;\n"
         " mov.f32 %qv,%f4;\n";
    s += kQuant32;
    s += " ld.param.u64 %rd10,[pXQ]; ld.param.u64 %rd11,[pXD]; ld.param.u64 %rd12,[pXS];\n"
         " cvt.u64.u32 %rd13,%r12; add.u64 %rd13,%rd10,%rd13; st.global.s8 [%rd13],%qi;\n"
         " and.b32 %rl,%r6,31; setp.ne.u32 %p3,%rl,0; @%p3 bra DONE;\n"
         " shr.u32 %r13,%r12,5; mul.wide.u32 %rd14,%r13,4; add.u64 %rd15,%rd11,%rd14; st.global.f32 [%rd15],%qd; add.u64 %rd15,%rd12,%rd14; st.global.s32 [%rd15],%qs;\n"
         "DONE:\n ret;\n}\n";
    // ---- k_moe_comb: X[t*D+j] += sum_k W[t*K+k] * T[Inv[t*K+k]*D+j], k in order (the fma of k_wsum, as decode does)
    s += ".visible .entry k_moe_comb(.param .u64 pX, .param .u64 pT, .param .u64 pInv, .param .u64 pW, .param .u32 pK, .param .u32 pD, .param .u32 pN)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<8>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pT]; ld.param.u64 %rd3,[pInv]; ld.param.u64 %rd4,[pW]; ld.param.u32 %r1,[pK]; ld.param.u32 %r2,[pD]; ld.param.u32 %r3,[pN];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ntid.x; mov.u32 %r6,%tid.x; mad.lo.u32 %r7,%r4,%r5,%r6; setp.ge.u32 %p1,%r7,%r3; @%p1 bra DONE;\n"
         " div.u32 %r8,%r7,%r2; mul.lo.u32 %r9,%r8,%r2; sub.u32 %r9,%r7,%r9; mul.lo.u32 %r10,%r8,%r1;\n"     // token, column, first pair
         " mul.wide.u32 %rd5,%r7,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f1,[%rd5]; mov.u32 %r11,0;\n"
         "L:\n setp.ge.u32 %p2,%r11,%r1; @%p2 bra E; add.u32 %r12,%r10,%r11; mul.wide.u32 %rd6,%r12,4; add.u64 %rd7,%rd3,%rd6; ld.global.u32 %r13,[%rd7];\n"
         " add.u64 %rd8,%rd4,%rd6; ld.global.f32 %f2,[%rd8]; mad.lo.u32 %r14,%r13,%r2,%r9; mul.wide.u32 %rd9,%r14,4; add.u64 %rd9,%rd2,%rd9; ld.global.f32 %f3,[%rd9];\n"
         " fma.rn.f32 %f1,%f2,%f3,%f1; add.u32 %r11,%r11,1; bra L;\nE:\n st.global.f32 [%rd5],%f1;\n"
         "DONE:\n ret;\n}\n";
    return s;
}
