// PTX for the non-matmul parts of a transformer layer, so whole layers can stay in VRAM.
#include <cstdio>
#include <string>

std::string gpu_ops_ptx();

namespace {
// Elementwise kernel: index i = ctaid.x*ntid.x + tid.x in %r5, exits when i >= pN (%r1).
std::string ew(const char* name, const char* params, const char* loads, const char* body) {
    std::string s = ".visible .entry ";
    s += name; s += "("; s += params; s += ", .param .u32 pN)\n{\n";
    s += " .reg .pred %p<4>; .reg .b32 %r<24>; .reg .f32 %f<16>; .reg .b64 %rd<16>; .reg .b16 %h<4>;\n";
    s += " ld.param.u32 %r1,[pN];\n";
    s += loads;
    s += " mov.u32 %r2,%ctaid.x; mov.u32 %r3,%ntid.x; mov.u32 %r4,%tid.x; mad.lo.u32 %r5,%r2,%r3,%r4;\n";
    s += " setp.ge.u32 %p1,%r5,%r1; @%p1 bra DONE;\n";
    s += body;
    s += "DONE:\n ret;\n}\n";
    return s;
}

// Block-wide reduction (256 threads = 8 warps) of register R; result broadcast to all threads.
std::string bred(const std::string& R, const char* op) {
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

const char* kRedPrologue =
    " .reg .pred %plane0; .reg .f32 %ft; .reg .b64 %rdw,%rdr; .reg .b32 %rlane,%rwarp;\n"
    " .shared .align 4 .f32 red[8];\n"
    " mov.u32 %rlane,%tid.x; shr.u32 %rwarp,%rlane,5; and.b32 %rlane,%rlane,31;\n"
    " setp.eq.u32 %plane0,%rlane,0;\n"
    " mov.u64 %rdr,red; mul.wide.u32 %rdw,%rwarp,4; add.u64 %rdw,%rdr,%rdw;\n";

// Attention over an f16 KV cache. grid = (n_head, n_tok), 256 threads (8 warps).
// Scores: one warp per key (lanes split head_dim, coalesced). Output: keys split across warps,
// partial sums combined through shared memory. Position comes from device memory (graph-friendly).
std::string gen_attn() {
    std::string s;
    char b[256];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); s += b; s += "\n"; };
    s += ".visible .entry k_attn(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pO, .param .u64 pSC,"
         " .param .u32 pNH, .param .u32 pHD, .param .u32 pKVD, .param .u32 pGRP, .param .u64 pPOS, .param .u32 pCTX, .param .f32 pSCALE,"
         " .param .u32 pSPL, .param .u64 pPM, .param .u64 pPL, .param .u64 pPO, .param .u32 pWIN, .param .u32 pNC)\n{\n"
         " .reg .pred %p<8>; .reg .pred %pk<8>; .reg .b32 %r<48>; .reg .f32 %f<32>; .reg .f32 %q<8>; .reg .f32 %a<8>;"
         " .reg .b64 %rd<40>; .reg .b16 %hs;\n"
         " .shared .align 4 .f32 part[2048];\n";
    s += kRedPrologue;
    s += " ld.param.u64 %rd1,[pQ]; ld.param.u64 %rd2,[pK]; ld.param.u64 %rd3,[pV]; ld.param.u64 %rd4,[pO]; ld.param.u64 %rd5,[pSC];\n"
         " ld.param.u32 %r10,[pNH]; ld.param.u32 %r11,[pHD]; ld.param.u32 %r12,[pKVD]; ld.param.u32 %r13,[pGRP];\n"
         " ld.param.u64 %rd20,[pPOS]; ld.global.u32 %r14,[%rd20]; ld.param.u32 %r15,[pCTX]; ld.param.f32 %f20,[pSCALE];\n"
         " ld.param.u32 %r34,[pSPL]; ld.param.u64 %rd30,[pPM]; ld.param.u64 %rd31,[pPL]; ld.param.u64 %rd32,[pPO];\n"
         " mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ctaid.y; mov.u32 %r3,%tid.x;\n"
         " div.u32 %r35,%r2,%r34; mul.lo.u32 %r36,%r35,%r34; sub.u32 %r36,%r2,%r36; mov.u32 %r2,%r35;\n"  // t, split s
         " add.u32 %r4,%r14,%r2; add.u32 %r4,%r4,1;\n"                                   // len (causal)
         // non-causal block (image tokens see each other): len = pos + NC; sliding window: start = len - WIN
         " ld.param.u32 %r42,[pNC]; setp.ne.u32 %p7,%r42,0; @%p7 add.u32 %r4,%r14,%r42;\n"
         " ld.param.u32 %r43,[pWIN]; mov.u32 %r44,0; setp.ne.u32 %p6,%r43,0; @%p6 setp.gt.u32 %p6,%r4,%r43; @%p6 sub.u32 %r44,%r4,%r43;\n"
         " sub.u32 %r45,%r4,%r44;\n"
         " add.u32 %r37,%r45,%r34; sub.u32 %r37,%r37,1; div.u32 %r37,%r37,%r34;\n"       // chunk
         " mad.lo.u32 %r38,%r36,%r37,%r44; add.u32 %r39,%r38,%r37; min.u32 %r39,%r39,%r4;\n"   // [j0, j1)
         " div.u32 %r5,%r1,%r13;\n"                                                       // kvh
         " mad.lo.u32 %r6,%r2,%r10,%r1;\n"                                                // t*nh+h
         " mul.lo.u32 %r7,%r6,%r11; mul.wide.u32 %rd6,%r7,4; add.u64 %rd7,%rd1,%rd6; add.u64 %rd8,%rd4,%rd6;\n"
         " mul.lo.u32 %r8,%r5,%r11; mul.wide.u32 %rd9,%r8,2; add.u64 %rd10,%rd2,%rd9; add.u64 %rd11,%rd3,%rd9;\n"
         " mul.wide.u32 %rd12,%r6,%r15; shl.b64 %rd12,%rd12,2; add.u64 %rd12,%rd5,%rd12;\n"
         " shl.b32 %r9,%r12,1;\n";
    // per-lane dims: dim_k = lane + 32k
    for (int k = 0; k < 8; k++) {
        L(" add.u32 %%r%d,%%rlane,%d; setp.lt.u32 %%pk%d,%%r%d,%%r11;", 20 + k, 32 * k, k, 20 + k);
        L(" mov.f32 %%q%d,0f00000000; mul.wide.u32 %%rd%d,%%r%d,4; add.u64 %%rd%d,%%rd7,%%rd%d; @%%pk%d ld.global.f32 %%q%d,[%%rd%d];",
          k, 21, 20 + k, 21, 21, k, k, 21);
    }
    // phase 1: scores
    s += " mov.f32 %f1,0fF149F2CA; add.u32 %r16,%r38,%rwarp;\n"
         "P1:\n setp.ge.u32 %p1,%r16,%r39; @%p1 bra E1;\n"
         " mul.wide.u32 %rd13,%r16,%r9; add.u64 %rd13,%rd10,%rd13;\n"
         " mov.f32 %f2,0f00000000;\n";
    for (int k = 0; k < 8; k++)
        L(" mul.wide.u32 %%rd14,%%r%d,2; add.u64 %%rd14,%%rd13,%%rd14; @%%pk%d ld.global.b16 %%hs,[%%rd14]; @%%pk%d cvt.f32.f16 %%f3,%%hs; @%%pk%d fma.rn.f32 %%f2,%%q%d,%%f3,%%f2;",
          20 + k, k, k, k, k);
    for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f4,%%f2,%d,31,0xffffffff; add.f32 %%f2,%%f2,%%f4;", o);
    s += " mul.f32 %f2,%f2,%f20;\n"
         " mul.wide.u32 %rd16,%r16,4; add.u64 %rd16,%rd12,%rd16; @%plane0 st.global.f32 [%rd16],%f2; max.f32 %f1,%f1,%f2;\n"
         " add.u32 %r16,%r16,8; bra P1;\nE1:\n";
    s += bred("%f1", "max.f32");
    // phase 2: exp + sum
    s += " mov.f32 %f5,0f00000000; add.u32 %r16,%r38,%r3;\n"
         "P2:\n setp.ge.u32 %p1,%r16,%r39; @%p1 bra E2;\n"
         " mul.wide.u32 %rd16,%r16,4; add.u64 %rd16,%rd12,%rd16; ld.global.f32 %f6,[%rd16];\n"
         " sub.f32 %f6,%f6,%f1; mul.f32 %f6,%f6,0f3FB8AA3B; ex2.approx.f32 %f6,%f6; st.global.f32 [%rd16],%f6; add.f32 %f5,%f5,%f6;\n"
         " add.u32 %r16,%r16,256; bra P2;\nE2:\n";
    s += bred("%f5", "add.f32");
    s += " rcp.approx.f32 %f7,%f5;\n";
    // phase 3: weighted sum of V, keys split across warps
    for (int k = 0; k < 8; k++) L(" mov.f32 %%a%d,0f00000000;", k);
    s += " add.u32 %r16,%r38,%rwarp;\n"
         "P3:\n setp.ge.u32 %p1,%r16,%r39; @%p1 bra E3;\n"
         " mul.wide.u32 %rd16,%r16,4; add.u64 %rd16,%rd12,%rd16; ld.global.f32 %f9,[%rd16];\n"
         " mul.wide.u32 %rd18,%r16,%r9; add.u64 %rd18,%rd11,%rd18;\n";
    for (int k = 0; k < 8; k++)
        L(" mul.wide.u32 %%rd19,%%r%d,2; add.u64 %%rd19,%%rd18,%%rd19; @%%pk%d ld.global.b16 %%hs,[%%rd19]; @%%pk%d cvt.f32.f16 %%f10,%%hs; @%%pk%d fma.rn.f32 %%a%d,%%f9,%%f10,%%a%d;",
          20 + k, k, k, k, k, k);
    s += " add.u32 %r16,%r16,8; bra P3;\nE3:\n";
    s += " mov.u64 %rd25,part; mul.lo.u32 %r30,%rwarp,%r11;\n";
    for (int k = 0; k < 8; k++)
        L(" add.u32 %%r31,%%r30,%%r%d; mul.wide.u32 %%rd26,%%r31,4; add.u64 %%rd26,%%rd25,%%rd26; @%%pk%d st.shared.f32 [%%rd26],%%a%d;", 20 + k, k, k);
    s += " bar.sync 0;\n"
         " setp.ge.u32 %p3,%r3,%r11; @%p3 bra DONE;\n"
         " mov.f32 %f8,0f00000000; mov.u32 %r32,0;\n"
         "P4:\n mad.lo.u32 %r33,%r32,%r11,%r3; mul.wide.u32 %rd27,%r33,4; add.u64 %rd27,%rd25,%rd27; ld.shared.f32 %f11,[%rd27]; add.f32 %f8,%f8,%f11;\n"
         " add.u32 %r32,%r32,1; setp.lt.u32 %p4,%r32,8; @%p4 bra P4;\n"
         " setp.ne.u32 %p5,%r34,1; @%p5 bra PART;\n"
         " mul.f32 %f8,%f8,%f7; mul.wide.u32 %rd28,%r3,4; add.u64 %rd28,%rd8,%rd28; st.global.f32 [%rd28],%f8; bra DONE;\n"
         "PART:\n mad.lo.u32 %r40,%r6,%r34,%r36;\n"                                   // partial slot
         " mad.lo.u32 %r41,%r40,%r11,%r3; mul.wide.u32 %rd33,%r41,4; add.u64 %rd33,%rd32,%rd33; st.global.f32 [%rd33],%f8;\n"
         " setp.ne.u32 %p6,%r3,0; @%p6 bra DONE;\n"
         " mul.wide.u32 %rd34,%r40,4; add.u64 %rd35,%rd30,%rd34; st.global.f32 [%rd35],%f1; add.u64 %rd36,%rd31,%rd34; st.global.f32 [%rd36],%f5;\n"
         "DONE:\n ret;\n}\n";
    // Merge split partials: out = sum_s e^(m_s-m) o_s / sum_s e^(m_s-m) l_s. grid = (n_head, n_tok), threads = head_dim.
    s += ".visible .entry k_attn_comb(.param .u64 pPM, .param .u64 pPL, .param .u64 pPO, .param .u64 pO, .param .u32 pNH, .param .u32 pHD, .param .u32 pSPL)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pPM]; ld.param.u64 %rd2,[pPL]; ld.param.u64 %rd3,[pPO]; ld.param.u64 %rd4,[pO];\n"
         " ld.param.u32 %r1,[pNH]; ld.param.u32 %r2,[pHD]; ld.param.u32 %r3,[pSPL];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ctaid.y; mov.u32 %r6,%tid.x;\n"
         " setp.ge.u32 %p1,%r6,%r2; @%p1 bra DONE;\n"
         " mad.lo.u32 %r7,%r5,%r1,%r4; mul.lo.u32 %r8,%r7,%r3;\n"                    // (t*nh+h), base slot
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
         "DONE:\n ret;\n}\n";
    return s;
}

// Decode attention for grouped-query models: grid = (n_kv_heads, splits), 256 threads (8 warps).
// A CTA reads each cached key/value of its kv head once and serves all G query heads that share it,
// with an online softmax in registers (no score buffer). Each split writes (max, sum, out) partials for
// k_attn_comb. Generated per model for its exact G (query heads per kv head) and NT = ceil(head_dim / 32),
// so the accumulators stay in registers.
std::string gen_attn_gqa(int G, int NT) {
    std::string s;
    char b[300];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); s += b; s += "\n"; };
    s += ".visible .entry k_attn_gqa(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pPM, .param .u64 pPL, .param .u64 pPO,"
         " .param .u32 pHD, .param .u32 pKVD, .param .u32 pGRP, .param .u64 pPOS, .param .f32 pSCALE, .param .u32 pSPL, .param .u32 pWIN)\n{\n"
         " .reg .pred %p<8>; .reg .pred %ps<8>; .reg .b32 %r<48>; .reg .f32 %f<24>; .reg .b64 %rd<24>; .reg .b16 %h;\n";
    L(" .reg .f32 %%q<%d>; .reg .f32 %%o<%d>; .reg .f32 %%m<%d>; .reg .f32 %%l<%d>; .reg .f32 %%k<%d>; .reg .f32 %%v<%d>;", G * NT, G * NT, G, G, NT, NT);
    L(" .shared .align 4 .f32 sO[%d]; .shared .align 4 .f32 sM[%d]; .shared .align 4 .f32 sL[%d];", 8 * G * NT * 32, 8 * G, 8 * G);
    s += " ld.param.u64 %rd1,[pQ]; ld.param.u64 %rd2,[pK]; ld.param.u64 %rd3,[pV]; ld.param.u64 %rd4,[pPM]; ld.param.u64 %rd5,[pPL]; ld.param.u64 %rd6,[pPO];\n"
         " ld.param.u32 %r1,[pHD]; ld.param.u32 %r2,[pKVD]; ld.param.u64 %rd7,[pPOS]; ld.param.f32 %f1,[pSCALE];\n"
         " ld.param.u32 %r4,[pSPL]; ld.param.u32 %r5,[pWIN];\n"
         " mov.u32 %r6,%ctaid.x; mov.u32 %r7,%ctaid.y; mov.u32 %r8,%tid.x; shr.u32 %r9,%r8,5; and.b32 %r10,%r8,31;\n"
         " ld.global.u32 %r11,[%rd7]; add.u32 %r11,%r11,1;\n"
         " mov.u32 %r12,0; setp.ne.u32 %p1,%r5,0; @%p1 setp.gt.u32 %p1,%r11,%r5; @%p1 sub.u32 %r12,%r11,%r5;\n"
         " sub.u32 %r13,%r11,%r12; add.u32 %r13,%r13,%r4; sub.u32 %r13,%r13,1; div.u32 %r13,%r13,%r4;\n"
         " mad.lo.u32 %r14,%r7,%r13,%r12; add.u32 %r15,%r14,%r13; min.u32 %r15,%r15,%r11;\n"
         " mul.f32 %f1,%f1,0f3FB8AA3B;\n";
    L(" mul.lo.u32 %%r16,%%r6,%d; mul.lo.u32 %%r17,%%r6,%%r1;", G);
    for (int t = 0; t < NT; t++) L(" add.u32 %%r%d,%%r10,%d; setp.lt.u32 %%ps%d,%%r%d,%%r1;", 20 + t, 32 * t, t, 20 + t);
    for (int g = 0; g < G; g++) {
        L(" add.u32 %%r18,%%r16,%d; mul.lo.u32 %%r18,%%r18,%%r1; mov.f32 %%m%d,0fF149F2CA; mov.f32 %%l%d,0f00000000;", g, g, g);
        for (int t = 0; t < NT; t++)
            L(" add.u32 %%r19,%%r18,%%r%d; mul.wide.u32 %%rd8,%%r19,4; add.u64 %%rd8,%%rd1,%%rd8; mov.f32 %%q%d,0f00000000; @%%ps%d ld.global.f32 %%q%d,[%%rd8]; mul.f32 %%q%d,%%q%d,%%f1; mov.f32 %%o%d,0f00000000;",
              20 + t, g * NT + t, t, g * NT + t, g * NT + t, g * NT + t, g * NT + t);
    }
    s += " add.u32 %r30,%r14,%r9;\n"
         "LJ:\n setp.ge.u32 %p3,%r30,%r15; @%p3 bra EJ;\n"
         " mad.lo.u32 %r31,%r30,%r2,%r17;\n";
    for (int t = 0; t < NT; t++)
        L(" add.u32 %%r32,%%r31,%%r%d; mul.wide.u32 %%rd9,%%r32,2; add.u64 %%rd10,%%rd2,%%rd9; add.u64 %%rd11,%%rd3,%%rd9;"
          " mov.f32 %%k%d,0f00000000; mov.f32 %%v%d,0f00000000; @%%ps%d ld.global.nc.b16 %%h,[%%rd10]; @%%ps%d cvt.f32.f16 %%k%d,%%h; @%%ps%d ld.global.nc.b16 %%h,[%%rd11]; @%%ps%d cvt.f32.f16 %%v%d,%%h;",
          20 + t, t, t, t, t, t, t, t, t);
    for (int g = 0; g < G; g++) {
        L(" mul.f32 %%f2,%%q%d,%%k0;", g * NT);
        for (int t = 1; t < NT; t++) L(" fma.rn.f32 %%f2,%%q%d,%%k%d,%%f2;", g * NT + t, t);
        for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f3,%%f2,%d,31,0xffffffff; add.f32 %%f2,%%f2,%%f3;", o);
        L(" max.f32 %%f4,%%m%d,%%f2; sub.f32 %%f5,%%m%d,%%f4; ex2.approx.f32 %%f5,%%f5; sub.f32 %%f6,%%f2,%%f4; ex2.approx.f32 %%f6,%%f6;", g, g);
        L(" fma.rn.f32 %%l%d,%%l%d,%%f5,%%f6; mov.f32 %%m%d,%%f4;", g, g, g);
        for (int t = 0; t < NT; t++) L(" mul.f32 %%o%d,%%o%d,%%f5; fma.rn.f32 %%o%d,%%f6,%%v%d,%%o%d;", g * NT + t, g * NT + t, g * NT + t, t, g * NT + t);
    }
    s += " add.u32 %r30,%r30,8; bra LJ;\nEJ:\n";
    s += " mov.u64 %rd12,sO; mov.u64 %rd13,sM; mov.u64 %rd14,sL; setp.eq.u32 %p4,%r10,0;\n";
    for (int g = 0; g < G; g++) {
        L(" mov.u32 %%r33,%d; mad.lo.u32 %%r33,%%r9,%d,%%r33; mul.lo.u32 %%r33,%%r33,%%r1;", g, G);
        for (int t = 0; t < NT; t++)
            L(" add.u32 %%r34,%%r33,%%r%d; mul.wide.u32 %%rd15,%%r34,4; add.u64 %%rd15,%%rd12,%%rd15; @%%ps%d st.shared.f32 [%%rd15],%%o%d;", 20 + t, t, g * NT + t);
        L(" mov.u32 %%r35,%d; mad.lo.u32 %%r35,%%r9,%d,%%r35; mul.wide.u32 %%rd16,%%r35,4;"
          " add.u64 %%rd17,%%rd13,%%rd16; @%%p4 st.shared.f32 [%%rd17],%%m%d; add.u64 %%rd18,%%rd14,%%rd16; @%%p4 st.shared.f32 [%%rd18],%%l%d;", g, G, g, g);
    }
    s += " bar.sync 0;\n";
    L(" mul.lo.u32 %%r36,%%r1,%d; mov.u32 %%r37,%%r8;", G);
    s += "LC:\n setp.ge.u32 %p5,%r37,%r36; @%p5 bra DONE;\n"
         " div.u32 %r38,%r37,%r1; rem.u32 %r39,%r37,%r1;\n"
         " mov.f32 %f7,0fF149F2CA;\n";
    for (int w = 0; w < 8; w++) L(" mov.u32 %%r40,%d; add.u32 %%r40,%%r40,%%r38; mul.wide.u32 %%rd19,%%r40,4; add.u64 %%rd19,%%rd13,%%rd19; ld.shared.f32 %%f8,[%%rd19]; max.f32 %%f7,%%f7,%%f8;", w * G);
    s += " mov.f32 %f9,0f00000000; mov.f32 %f10,0f00000000;\n";
    for (int w = 0; w < 8; w++) {
        L(" mov.u32 %%r40,%d; add.u32 %%r40,%%r40,%%r38; mul.wide.u32 %%rd19,%%r40,4; add.u64 %%rd20,%%rd13,%%rd19; ld.shared.f32 %%f8,[%%rd20];", w * G);
        s += " sub.f32 %f11,%f8,%f7; ex2.approx.f32 %f11,%f11; add.u64 %rd21,%rd14,%rd19; ld.shared.f32 %f12,[%rd21]; fma.rn.f32 %f9,%f12,%f11,%f9;\n";
        L(" mov.u32 %%r41,%d; add.u32 %%r41,%%r41,%%r38; mad.lo.u32 %%r41,%%r41,%%r1,%%r39; mul.wide.u32 %%rd22,%%r41,4; add.u64 %%rd22,%%rd12,%%rd22; ld.shared.f32 %%f13,[%%rd22]; fma.rn.f32 %%f10,%%f13,%%f11,%%f10;", w * G);
    }
    s += " add.u32 %r42,%r16,%r38; mad.lo.u32 %r43,%r42,%r4,%r7;\n"
         " mad.lo.u32 %r44,%r43,%r1,%r39; mul.wide.u32 %rd23,%r44,4; add.u64 %rd23,%rd6,%rd23; st.global.f32 [%rd23],%f10;\n"
         " setp.ne.u32 %p6,%r39,0; @%p6 bra NX;\n"
         " mul.f32 %f14,%f7,0f3F317218; mul.wide.u32 %rd23,%r43,4; add.u64 %rd8,%rd4,%rd23; st.global.f32 [%rd8],%f14; add.u64 %rd9,%rd5,%rd23; st.global.f32 [%rd9],%f9;\n"
         "NX:\n add.u32 %r37,%r37,256; bra LC;\n"
         "DONE:\n ret;\n}\n";
    return s;
}
}  // namespace

std::string gpu_ops_ptx() {
    std::string s;
    // y[i] += s * x[i]
    s += ew("k_add", ".param .u64 pY, .param .u64 pX, .param .f32 pS",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pX]; ld.param.f32 %f1,[pS];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3;\n"
            " ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5]; fma.rn.f32 %f2,%f3,%f1,%f2; st.global.f32 [%rd4],%f2;\n");
    // y[i] += b[i % D]
    s += ew("k_bias", ".param .u64 pY, .param .u64 pB, .param .u32 pD",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pB]; ld.param.u32 %r7,[pD];\n",
            " rem.u32 %r6,%r5,%r7;\n"
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; mul.wide.u32 %rd5,%r6,4; add.u64 %rd5,%rd2,%rd5;\n"
            " ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5]; add.f32 %f2,%f2,%f3; st.global.f32 [%rd4],%f2;\n");
    // g[i] = silu(g[i]) * u[i]
    s += ew("k_silu", ".param .u64 pG, .param .u64 pU",
            " ld.param.u64 %rd1,[pG]; ld.param.u64 %rd2,[pU];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3;\n"
            " ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5];\n"
            " mul.f32 %f4,%f2,0fBFB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000;\n"
            " div.approx.f32 %f5,%f2,%f4; mul.f32 %f5,%f5,%f3; st.global.f32 [%rd4],%f5;\n");
    // o[t*F+j] = silu(T[t*2F+j]) * T[t*2F+F+j]   (fused gate/up)
    s += ew("k_silu_split", ".param .u64 pT, .param .u64 pO, .param .u32 pF",
            " ld.param.u64 %rd1,[pT]; ld.param.u64 %rd2,[pO]; ld.param.u32 %r7,[pF];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " shl.b32 %r10,%r7,1; mul.lo.u32 %r10,%r10,%r8; add.u32 %r10,%r10,%r9;\n"
            " mul.wide.u32 %rd3,%r10,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3];\n"
            " mul.wide.u32 %rd4,%r7,4; add.u64 %rd4,%rd3,%rd4; ld.global.f32 %f3,[%rd4];\n"
            " mul.f32 %f4,%f2,0fBFB8AA3B; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000;\n"
            " div.approx.f32 %f5,%f2,%f4; mul.f32 %f5,%f5,%f3;\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd2,%rd5; st.global.f32 [%rd5],%f5;\n");
    // g[i] = gelu_tanh(g[i]) * u[i]   (Gemma)
    const char* kGelu =
        " mul.f32 %f6,%f2,%f2; mul.f32 %f6,%f6,%f2; fma.rn.f32 %f6,%f6,0f3D372713,%f2; mul.f32 %f6,%f6,0f3F4C422A;\n"   // u = 0.79788*(x + 0.044715x^3)
        " mul.f32 %f7,%f6,0fC038AA3B; ex2.approx.f32 %f7,%f7; add.f32 %f7,%f7,0f3F800000; div.approx.f32 %f7,%f2,%f7;\n";  // x*sigmoid(2u)
    std::string gelu_body = std::string(" mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3;\n"
                                        " ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5];\n") + kGelu +
                            " mul.f32 %f5,%f7,%f3; st.global.f32 [%rd4],%f5;\n";
    s += ew("k_gelu", ".param .u64 pG, .param .u64 pU", " ld.param.u64 %rd1,[pG]; ld.param.u64 %rd2,[pU];\n", gelu_body.c_str());
    // dst[r*D+c] = src[idx[r]*D+c]
    s += ew("k_gather", ".param .u64 pS, .param .u64 pI, .param .u64 pD, .param .u32 pDim",
            " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pI]; ld.param.u64 %rd6,[pD]; ld.param.u32 %r7,[pDim];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd3,%rd2,%rd3; ld.global.u32 %r10,[%rd3];\n"
            " mad.lo.u32 %r11,%r10,%r7,%r9; mul.wide.u32 %rd4,%r11,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f2,[%rd4];\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd6,%rd5; st.global.f32 [%rd5],%f2;\n");
    // dst[idx[r]*D+c] += w[r] * src[r*D+c]
    s += ew("k_scatter", ".param .u64 pY, .param .u64 pS, .param .u64 pI, .param .u64 pW, .param .u32 pDim",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pS]; ld.param.u64 %rd6,[pI]; ld.param.u64 %rd7,[pW]; ld.param.u32 %r7,[pDim];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd4,%rd6,%rd3; ld.global.u32 %r10,[%rd4]; add.u64 %rd8,%rd7,%rd3; ld.global.f32 %f3,[%rd8];\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd2,%rd5; ld.global.f32 %f2,[%rd5];\n"
            " mad.lo.u32 %r11,%r10,%r7,%r9; mul.wide.u32 %rd9,%r11,4; add.u64 %rd9,%rd1,%rd9; ld.global.f32 %f4,[%rd9];\n"
            " fma.rn.f32 %f4,%f2,%f3,%f4; st.global.f32 [%rd9],%f4;\n");

    // d[i] = (f16) s[i]
    s += ew("k_f16", ".param .u64 pS, .param .u64 pD",
            " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pD];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3];\n"
            " cvt.rn.f16.f32 %h1,%f2; mul.wide.u32 %rd4,%r5,2; add.u64 %rd4,%rd2,%rd4; st.global.b16 [%rd4],%h1;\n");
    // D[r*DS+c] = S[r*SS+c], i = r*W + c
    s += ew("k_copy2d", ".param .u64 pS, .param .u32 pSS, .param .u64 pD, .param .u32 pDS, .param .u32 pW",
            " ld.param.u64 %rd1,[pS]; ld.param.u32 %r7,[pSS]; ld.param.u64 %rd2,[pD]; ld.param.u32 %r8,[pDS]; ld.param.u32 %r9,[pW];\n",
            " div.u32 %r10,%r5,%r9; mul.lo.u32 %r11,%r10,%r9; sub.u32 %r11,%r5,%r11;\n"
            " mad.lo.u32 %r12,%r10,%r7,%r11; mad.lo.u32 %r13,%r10,%r8,%r11;\n"
            " mul.wide.u32 %rd3,%r12,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3];\n"
            " mul.wide.u32 %rd4,%r13,4; add.u64 %rd4,%rd2,%rd4; st.global.f32 [%rd4],%f2;\n");

    // RMSNorm, one CTA (256 threads) per row. In-place safe.
    s += ".visible .entry k_rmsnorm(.param .u64 pX, .param .u64 pW, .param .u64 pY, .param .u32 pD, .param .f32 pE)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n";
    s += kRedPrologue;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pW]; ld.param.u64 %rd3,[pY]; ld.param.u32 %r1,[pD]; ld.param.f32 %f10,[pE];\n"
         " mov.u32 %r2,%ctaid.x; mov.u32 %r3,%tid.x;\n"
         " mul.wide.u32 %rd4,%r2,%r1; shl.b64 %rd4,%rd4,2; add.u64 %rd5,%rd1,%rd4; add.u64 %rd6,%rd3,%rd4;\n"
         " mov.f32 %f1,0f00000000; mov.u32 %r4,%r3;\n"
         "L1:\n setp.ge.u32 %p1,%r4,%r1; @%p1 bra E1;\n"
         " mul.wide.u32 %rd7,%r4,4; add.u64 %rd8,%rd5,%rd7; ld.global.f32 %f2,[%rd8]; fma.rn.f32 %f1,%f2,%f2,%f1;\n"
         " add.u32 %r4,%r4,256; bra L1;\nE1:\n";
    s += bred("%f1", "add.f32");
    s += " cvt.rn.f32.u32 %f5,%r1; div.rn.f32 %f3,%f1,%f5; add.f32 %f3,%f3,%f10; rsqrt.approx.f32 %f3,%f3;\n"
         " mov.u32 %r4,%r3;\n"
         "L2:\n setp.ge.u32 %p1,%r4,%r1; @%p1 bra E2;\n"
         " mul.wide.u32 %rd7,%r4,4; add.u64 %rd8,%rd5,%rd7; ld.global.f32 %f2,[%rd8];\n"
         " add.u64 %rd9,%rd2,%rd7; ld.global.f32 %f6,[%rd9]; mul.f32 %f2,%f2,%f3; mul.f32 %f2,%f2,%f6;\n"
         " add.u64 %rd10,%rd6,%rd7; st.global.f32 [%rd10],%f2;\n"
         " add.u32 %r4,%r4,256; bra L2;\nE2:\n ret;\n}\n";

    // RoPE with a host-computed cos/sin table CS[tok][half][2]. grid.x = token.
    s += ".visible .entry k_rope(.param .u64 pX, .param .u64 pCS, .param .u32 pH, .param .u32 pHD, .param .u32 pHalf,"
         " .param .u32 pNeox, .param .u32 pStride)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<24>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pCS]; ld.param.u32 %r1,[pH]; ld.param.u32 %r2,[pHD];\n"
         " ld.param.u32 %r3,[pHalf]; ld.param.u32 %r4,[pNeox]; ld.param.u32 %r5,[pStride];\n"
         " mov.u32 %r6,%ctaid.x; mov.u32 %r7,%tid.x; mul.lo.u32 %r8,%r1,%r3;\n"
         " setp.ne.u32 %p2,%r4,0;\n"
         "L1:\n setp.ge.u32 %p1,%r7,%r8; @%p1 bra DONE;\n"
         " div.u32 %r9,%r7,%r3; mul.lo.u32 %r10,%r9,%r3; sub.u32 %r10,%r7,%r10;\n"          // h, i
         " mul.lo.u32 %r11,%r6,%r5; mad.lo.u32 %r11,%r9,%r2,%r11;\n"                          // row base (elements)
         " shl.b32 %r12,%r10,1; add.u32 %r13,%r12,1;\n"                                       // 2i, 2i+1
         " add.u32 %r14,%r10,%r3;\n"                                                          // i+half
         " selp.u32 %r15,%r10,%r12,%p2; selp.u32 %r16,%r14,%r13,%p2;\n"
         " add.u32 %r15,%r15,%r11; add.u32 %r16,%r16,%r11;\n"
         " mul.wide.u32 %rd3,%r15,4; add.u64 %rd3,%rd1,%rd3; mul.wide.u32 %rd4,%r16,4; add.u64 %rd4,%rd1,%rd4;\n"
         " mad.lo.u32 %r17,%r6,%r3,%r10; mul.wide.u32 %rd5,%r17,8; add.u64 %rd5,%rd2,%rd5;\n"
         " ld.global.f32 %f1,[%rd5]; ld.global.f32 %f2,[%rd5+4];\n"
         " ld.global.f32 %f3,[%rd3]; ld.global.f32 %f4,[%rd4];\n"
         " mul.f32 %f5,%f3,%f1; mul.f32 %f6,%f4,%f2; sub.f32 %f5,%f5,%f6;\n"
         " mul.f32 %f7,%f3,%f2; fma.rn.f32 %f7,%f4,%f1,%f7;\n"
         " st.global.f32 [%rd3],%f5; st.global.f32 [%rd4],%f7;\n"
         " mov.u32 %r18,%ntid.x; add.u32 %r7,%r7,%r18; bra L1;\n"
         "DONE:\n ret;\n}\n";

    // KV store: cache[pos*kvd + i] = (f16) src[i], pos read from device memory.
    s += ew("k_kvstore", ".param .u64 pS, .param .u64 pD, .param .u64 pPos, .param .u32 pKVD",
            " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pD]; ld.param.u64 %rd6,[pPos]; ld.param.u32 %r7,[pKVD];\n"
            " ld.global.u32 %r8,[%rd6];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd3,%rd1,%rd3; ld.global.f32 %f2,[%rd3];\n"
            " mad.lo.u32 %r9,%r8,%r7,%r5;\n"
            " cvt.rn.f16.f32 %h1,%f2; mul.wide.u32 %rd4,%r9,2; add.u64 %rd4,%rd2,%rd4; st.global.b16 [%rd4],%h1;\n");
    s += gen_attn();
    return s;
}

// Per-model decode attention module (see gen_attn_gqa).
std::string gqa_attn_ptx(int G, int NT) { return ".version 6.0\n.target sm_61\n.address_size 64\n" + gen_attn_gqa(G, NT); }
