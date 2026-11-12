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
std::string gen_attn_gqa(int G, int NT, bool q8) {
    std::string s;
    char b[300];
    auto L = [&](const char* fmt, auto... a) { snprintf(b, sizeof b, fmt, a...); s += b; s += "\n"; };
    s += ".visible .entry k_attn_gqa(.param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u64 pPM, .param .u64 pPL, .param .u64 pPO,"
         " .param .u32 pHD, .param .u32 pKVD, .param .u32 pGRP, .param .u64 pPOS, .param .f32 pSCALE, .param .u32 pSPL, .param .u32 pWIN,"
         " .param .u64 pKs, .param .u64 pVs, .param .u32 pNKV, .param .u64 pSink, .param .f32 pCap)\n{\n"
         " .reg .f32 %cp<4>; .reg .pred %pcap;\n"
         " ld.param.f32 %cp0,[pCap]; setp.gt.f32 %pcap,%cp0,0f00000000; @%pcap rcp.approx.f32 %cp1,%cp0; @%pcap add.f32 %cp1,%cp1,%cp1; @%pcap mul.f32 %cp2,%cp0,0f3FB8AA3B;\n"
         " .reg .pred %p<8>; .reg .pred %ps<8>; .reg .b32 %r<56>; .reg .f32 %f<24>; .reg .b64 %rd<32>; .reg .b16 %h;\n";
    L(" .reg .f32 %%q<%d>; .reg .f32 %%o<%d>; .reg .f32 %%m<%d>; .reg .f32 %%l<%d>; .reg .f32 %%k<%d>; .reg .f32 %%v<%d>;", G * NT, G * NT, G, G, NT, NT);
    L(" .shared .align 4 .f32 sO[%d]; .shared .align 4 .f32 sM[%d]; .shared .align 4 .f32 sL[%d];", 8 * G * NT * 32, 8 * G, 8 * G);
    s += " ld.param.u64 %rd1,[pQ]; ld.param.u64 %rd2,[pK]; ld.param.u64 %rd3,[pV]; ld.param.u64 %rd4,[pPM]; ld.param.u64 %rd5,[pPL]; ld.param.u64 %rd6,[pPO];\n"
         " ld.param.u32 %r1,[pHD]; ld.param.u32 %r2,[pKVD]; ld.param.u64 %rd7,[pPOS]; ld.param.f32 %f1,[pSCALE];\n"
         " ld.param.u32 %r4,[pSPL]; ld.param.u32 %r5,[pWIN]; ld.param.u64 %rd24,[pKs]; ld.param.u64 %rd25,[pVs]; ld.param.u32 %r50,[pNKV];\n"
         " mov.u32 %r6,%ctaid.x; mov.u32 %r7,%ctaid.y; mov.u32 %r8,%tid.x; shr.u32 %r9,%r8,5; and.b32 %r10,%r8,31;\n"
         " ld.global.u32 %r11,[%rd7]; add.u32 %r11,%r11,1;\n"
         " mov.u32 %r12,0; setp.ne.u32 %p1,%r5,0; @%p1 setp.gt.u32 %p1,%r11,%r5; @%p1 sub.u32 %r12,%r11,%r5;\n"
         " sub.u32 %r13,%r11,%r12; add.u32 %r13,%r13,%r4; sub.u32 %r13,%r13,1; div.u32 %r13,%r13,%r4;\n"
         " mad.lo.u32 %r14,%r7,%r13,%r12; add.u32 %r15,%r14,%r13; min.u32 %r15,%r15,%r11;\n"
         " mul.f32 %f1,%f1,0f3FB8AA3B;\n";
    L(" mul.lo.u32 %%r16,%%r6,%d; mul.lo.u32 %%r17,%%r6,%%r1;", G);
    for (int t = 0; t < NT; t++) L(" add.u32 %%r%d,%%r10,%d; setp.lt.u32 %%ps%d,%%r%d,%%r1;", 20 + t, 32 * t, t, 20 + t);
    // attention sinks (gpt-oss): a per-head logit with no value. It starts every softmax (max = sink), and is counted
    // once in the sum: in warp 0 of split 0.
    s += " ld.param.u64 %rd29,[pSink]; setp.ne.u64 %p7,%rd29,0; setp.eq.u32 %p6,%r9,0; setp.eq.and.u32 %p6,%r7,0,%p6; and.pred %p6,%p6,%p7;\n";
    for (int g = 0; g < G; g++) {
        L(" add.u32 %%r18,%%r16,%d; mul.lo.u32 %%r18,%%r18,%%r1; mov.f32 %%m%d,0fF149F2CA; mov.f32 %%l%d,0f00000000;", g, g, g);
        L(" add.u32 %%r19,%%r16,%d; mul.wide.u32 %%rd30,%%r19,4; add.u64 %%rd30,%%rd29,%%rd30; @%%p7 ld.global.nc.f32 %%m%d,[%%rd30]; @%%p7 mul.f32 %%m%d,%%m%d,0f3FB8AA3B; @%%p6 mov.f32 %%l%d,0f3F800000;", g, g, g, g, g);
        for (int t = 0; t < NT; t++)
            L(" add.u32 %%r19,%%r18,%%r%d; mul.wide.u32 %%rd8,%%r19,4; add.u64 %%rd8,%%rd1,%%rd8; mov.f32 %%q%d,0f00000000; @%%ps%d ld.global.f32 %%q%d,[%%rd8]; mul.f32 %%q%d,%%q%d,%%f1; mov.f32 %%o%d,0f00000000;",
              20 + t, g * NT + t, t, g * NT + t, g * NT + t, g * NT + t, g * NT + t);
    }
    s += " add.u32 %r30,%r14,%r9;\n"
         "LJ:\n setp.ge.u32 %p3,%r30,%r15; @%p3 bra EJ;\n"
         " mad.lo.u32 %r31,%r30,%r2,%r17;\n";
    if (q8) {   // int8 values; the per-(token, kv head) scales are applied to the dot product and to p
        s += " mad.lo.u32 %r51,%r30,%r50,%r6; mul.wide.u32 %rd26,%r51,4; add.u64 %rd27,%rd24,%rd26; ld.global.nc.f32 %f16,[%rd27]; add.u64 %rd28,%rd25,%rd26; ld.global.nc.f32 %f17,[%rd28];\n";
        for (int t = 0; t < NT; t++)
            L(" add.u32 %%r32,%%r31,%%r%d; cvt.u64.u32 %%rd9,%%r32; add.u64 %%rd10,%%rd2,%%rd9; add.u64 %%rd11,%%rd3,%%rd9; mov.u32 %%r52,0; mov.u32 %%r53,0;"
              " @%%ps%d ld.global.nc.s8 %%r52,[%%rd10]; @%%ps%d ld.global.nc.s8 %%r53,[%%rd11]; cvt.rn.f32.s32 %%k%d,%%r52; cvt.rn.f32.s32 %%v%d,%%r53;",
              20 + t, t, t, t, t);
    } else
    for (int t = 0; t < NT; t++)
        L(" add.u32 %%r32,%%r31,%%r%d; mul.wide.u32 %%rd9,%%r32,2; add.u64 %%rd10,%%rd2,%%rd9; add.u64 %%rd11,%%rd3,%%rd9;"
          " mov.f32 %%k%d,0f00000000; mov.f32 %%v%d,0f00000000; @%%ps%d ld.global.nc.b16 %%h,[%%rd10]; @%%ps%d cvt.f32.f16 %%k%d,%%h; @%%ps%d ld.global.nc.b16 %%h,[%%rd11]; @%%ps%d cvt.f32.f16 %%v%d,%%h;",
          20 + t, t, t, t, t, t, t, t, t);
    for (int g = 0; g < G; g++) {
        L(" mul.f32 %%f2,%%q%d,%%k0;", g * NT);
        for (int t = 1; t < NT; t++) L(" fma.rn.f32 %%f2,%%q%d,%%k%d,%%f2;", g * NT + t, t);
        for (int o = 16; o >= 1; o /= 2) L(" shfl.sync.bfly.b32 %%f3,%%f2,%d,31,0xffffffff; add.f32 %%f2,%%f2,%%f3;", o);
        if (q8) s += " mul.f32 %f2,%f2,%f16;\n";
        s += " @%pcap mul.f32 %cp3,%f2,%cp1; @%pcap ex2.approx.f32 %cp3,%cp3; @%pcap add.f32 %cp3,%cp3,0f3F800000; @%pcap rcp.approx.f32 %cp3,%cp3; @%pcap fma.rn.f32 %cp3,%cp3,0fC0000000,0f3F800000; @%pcap mul.f32 %f2,%cp3,%cp2;\n";
        L(" max.f32 %%f4,%%m%d,%%f2; sub.f32 %%f5,%%m%d,%%f4; ex2.approx.f32 %%f5,%%f5; sub.f32 %%f6,%%f2,%%f4; ex2.approx.f32 %%f6,%%f6;", g, g);
        L(" fma.rn.f32 %%l%d,%%l%d,%%f5,%%f6; mov.f32 %%m%d,%%f4;", g, g, g);
        if (q8) s += " mul.f32 %f6,%f6,%f17;\n";
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

// k_ssm_scan: block = 256 / pDs state rows (row = head * hdim + p) x pDs state entries, one entry per thread, in a
// register for all N tokens: S = exp(dt A) S + dt x B; y = sum(S C) + D x. dt = softplus(raw + bias).
// X: conv output [N][pC] (x at 0, B at pBo + group * pDs, C at pCo + group * pDs); dt raw at ZX[t * pLd + pDto + h].
// Tokens go in chunks of 16. Per chunk: dt and exp(dt A) once per (row, token) into shared memory; all loads issued
// up front; the 16 S*C sums reduced across the warp together (16 shuffles, not 5 per token); one barrier pair.
std::string gen_ssm_scan() {
    const int TC = 16;
    std::string s;
    s += ".visible .entry k_ssm_scan(.param .u64 pX, .param .u64 pZX, .param .u64 pDtb, .param .u64 pA, .param .u64 pD, .param .u64 pS, .param .u64 pY,"
         " .param .u32 pN, .param .u32 pC, .param .u32 pLd, .param .u32 pDto, .param .u32 pDi, .param .u32 pDs, .param .u32 pHdim, .param .u32 pHpg, .param .u32 pBo, .param .u32 pCo)\n{\n"
         " .reg .pred %p<12>; .reg .pred %pq<16>; .reg .b32 %r<64>; .reg .f32 %f<32>; .reg .f32 %qx<16>; .reg .f32 %qb<16>; .reg .f32 %qc<16>; .reg .b64 %rd<32>;\n"
         " .shared .align 4 .f32 red[128]; .shared .align 4 .f32 sdt[256]; .shared .align 4 .f32 sda[256];\n"
         " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pZX]; ld.param.u64 %rd3,[pDtb]; ld.param.u64 %rd4,[pA]; ld.param.u64 %rd5,[pD]; ld.param.u64 %rd6,[pS]; ld.param.u64 %rd7,[pY];\n"
         " ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pC]; ld.param.u32 %r3,[pLd]; ld.param.u32 %r4,[pDto]; ld.param.u32 %r5,[pDi]; ld.param.u32 %r6,[pDs]; ld.param.u32 %r7,[pHdim]; ld.param.u32 %r8,[pHpg];\n"
         " ld.param.u32 %r30,[pBo]; ld.param.u32 %r31,[pCo];\n"
         " mov.u32 %r9,%tid.x; rem.u32 %r10,%r9,%r6; div.u32 %r11,%r9,%r6;\n"                 // n, row in block
         " div.u32 %r12,256,%r6; mov.u32 %r13,%ctaid.x; mad.lo.u32 %r14,%r13,%r12,%r11;\n"    // rows per block, row
         " div.u32 %r15,%r14,%r7; div.u32 %r16,%r15,%r8; mul.lo.u32 %r16,%r16,%r6; add.u32 %r16,%r16,%r10;\n"   // h; group * ds + n
         " add.u32 %r32,%r30,%r16; add.u32 %r33,%r31,%r16;\n"                                   // B, C columns
         " mad.lo.u32 %r17,%r14,%r6,%r10; mul.wide.u32 %rd8,%r17,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f1,[%rd8];\n"   // state
         " shr.u32 %r18,%r9,5; and.b32 %r19,%r9,31; mov.u32 %r22,red; shr.u32 %r20,%r6,5;\n"   // warp, lane, warps per row
         " and.b32 %r25,%r19,1; setp.eq.u32 %p2,%r25,0; shr.u32 %r26,%r19,1; shl.b32 %r26,%r26,3; add.u32 %r26,%r26,%r18; shl.b32 %r26,%r26,2; add.u32 %r21,%r22,%r26;\n"   // red[(lane/2)*8 + warp]
         " and.b32 %r27,%r19,16; setp.ne.u32 %p8,%r27,0; and.b32 %r27,%r19,8; setp.ne.u32 %p9,%r27,0; and.b32 %r27,%r19,4; setp.ne.u32 %p10,%r27,0; and.b32 %r27,%r19,2; setp.ne.u32 %p11,%r27,0;\n"
         " mov.u32 %r28,sdt; mov.u32 %r29,sda; shl.b32 %r35,%r11,6; add.u32 %r36,%r28,%r35; add.u32 %r37,%r29,%r35;\n"   // this row's dt / dA
         " mul.wide.u32 %rd21,%r14,4; add.u64 %rd21,%rd1,%rd21; mul.wide.u32 %rd22,%r32,4; add.u64 %rd22,%rd1,%rd22; mul.wide.u32 %rd23,%r33,4; add.u64 %rd23,%rd1,%rd23;\n"   // x, B, C pointers
         " mul.wide.u32 %rd25,%r2,4;\n"
         " mul.lo.u32 %r41,%r12," + std::to_string(TC) + ";\n"                                   // dt / finisher threads
         // the dt / finisher thread's (row, token): rr = tid / 16, j = tid % 16
         " shr.u32 %r42,%r9,4; and.b32 %r43,%r9,15; mad.lo.u32 %r46,%r13,%r12,%r42; div.u32 %r47,%r46,%r7; setp.lt.u32 %p3,%r9,%r41;\n"
         " mov.f32 %f20,0f00000000; mov.f32 %f21,0f00000000; mov.f32 %f22,0f00000000;\n"
         " @%p3 mul.wide.u32 %rd26,%r47,4; @%p3 add.u64 %rd27,%rd3,%rd26; @%p3 ld.global.f32 %f20,[%rd27]; @%p3 add.u64 %rd27,%rd4,%rd26; @%p3 ld.global.f32 %f21,[%rd27];\n"
         " @%p3 add.u64 %rd27,%rd5,%rd26; @%p3 ld.global.f32 %f22,[%rd27]; mul.f32 %f21,%f21,0f3FB8AA3B;\n"   // dtb, A * log2(e), D of that row's head
         " shl.b32 %r38,%r9,2; add.u32 %r39,%r28,%r38; add.u32 %r48,%r29,%r38;\n"
         " mov.u32 %r24,0;\n"
         "LC:\n setp.ge.u32 %p1,%r24,%r1; @%p1 bra ET;\n"
         " add.u32 %r44,%r24,%r43; setp.lt.u32 %p6,%r44,%r1; and.pred %p7,%p6,%p3;\n"
         " @!%p3 bra DT; mov.f32 %f5,0f00000000; mov.f32 %f7,0f3F800000; @!%p7 bra DS;\n"
         " mad.lo.u32 %r23,%r44,%r3,%r4; add.u32 %r23,%r23,%r47; mul.wide.u32 %rd20,%r23,4; add.u64 %rd20,%rd2,%rd20; ld.global.f32 %f5,[%rd20]; add.f32 %f5,%f5,%f20;\n"
         " mul.f32 %f6,%f5,0f3FB8AA3B; ex2.approx.f32 %f6,%f6; add.f32 %f6,%f6,0f3F800000; lg2.approx.f32 %f6,%f6; mul.f32 %f6,%f6,0f3F317218; setp.gt.f32 %p4,%f5,0f41A00000; selp.f32 %f5,%f5,%f6,%p4;\n"   // softplus
         " mul.f32 %f7,%f5,%f21; ex2.approx.f32 %f7,%f7;\n"
         "DS:\n st.shared.f32 [%r39],%f5; st.shared.f32 [%r48],%f7;\n"
         "DT:\n bar.sync 0;\n";
    for (int j = 0; j < TC; j++) {
        std::string J = std::to_string(j);
        s += " add.u32 %r40,%r24," + J + "; setp.lt.u32 %pq" + J + ",%r40,%r1;"
             " mov.f32 %qx" + J + ",0f00000000; mov.f32 %qb" + J + ",0f00000000; mov.f32 %qc" + J + ",0f00000000;\n"
             " @%pq" + J + " ld.global.f32 %qx" + J + ",[%rd21]; @%pq" + J + " ld.global.f32 %qb" + J + ",[%rd22]; @%pq" + J + " ld.global.f32 %qc" + J + ",[%rd23];\n"
             " add.u64 %rd21,%rd21,%rd25; add.u64 %rd22,%rd22,%rd25; add.u64 %rd23,%rd23,%rd25;\n";
    }
    for (int j = 0; j < TC; j++) {
        std::string J = std::to_string(j), O = std::to_string(j * 4);
        s += " ld.shared.f32 %f5,[%r36+" + O + "]; ld.shared.f32 %f7,[%r37+" + O + "];"
             " mul.f32 %f11,%f5,%qx" + J + "; mul.f32 %f11,%f11,%qb" + J + "; fma.rn.f32 %f1,%f1,%f7,%f11; mul.f32 %qc" + J + ",%f1,%qc" + J + ";\n";
    }
    // 16 values per lane -> lane pair (2t, 2t+1) holds token t's sum: halve the set at each of offsets 16, 8, 4, 2
    int k = TC; const char* bp[4] = {"%p8", "%p9", "%p10", "%p11"};
    for (int st = 0; st < 4; st++) {
        k /= 2; std::string off = std::to_string(16 >> st);
        for (int i = 0; i < k; i++) {
            std::string lo = "%qc" + std::to_string(i), hi = "%qc" + std::to_string(i + k);
            s += std::string(" selp.f32 %f13,") + lo + "," + hi + "," + bp[st] + "; selp.f32 %f14," + hi + "," + lo + "," + bp[st] + ";"
                 " shfl.sync.bfly.b32 %f13,%f13," + off + ",31,0xffffffff; add.f32 " + lo + ",%f14,%f13;\n";
        }
    }
    s += " shfl.sync.bfly.b32 %f13,%qc0,1,31,0xffffffff; add.f32 %qc0,%qc0,%f13; @%p2 st.shared.f32 [%r21],%qc0;\n"
         " bar.sync 0;\n"
         // finisher: thread = (row in block) * 16 + token in chunk: sum the warps' partials, + D x, store
         " @!%p7 bra NF;\n"
         " shl.b32 %r45,%r43,3; mad.lo.u32 %r45,%r42,%r20,%r45; shl.b32 %r45,%r45,2; add.u32 %r45,%r22,%r45;\n"
         " mov.f32 %f14,0f00000000; mov.u32 %r34,0;\n"
         "LS:\n setp.ge.u32 %p5,%r34,%r20; @%p5 bra ES; ld.shared.f32 %f15,[%r45]; add.f32 %f14,%f14,%f15; add.u32 %r45,%r45,4; add.u32 %r34,%r34,1; bra LS;\nES:\n"
         " mad.lo.u32 %r49,%r44,%r2,%r46; mul.wide.u32 %rd28,%r49,4; add.u64 %rd28,%rd1,%rd28; ld.global.f32 %f17,[%rd28];\n"   // x
         " fma.rn.f32 %f14,%f22,%f17,%f14; mad.lo.u32 %r49,%r44,%r5,%r46; mul.wide.u32 %rd28,%r49,4; add.u64 %rd28,%rd7,%rd28; st.global.f32 [%rd28],%f14;\n"
         "NF:\n add.u32 %r24,%r24," + std::to_string(TC) + "; bra LC;\nET:\n"
         " st.global.f32 [%rd8],%f1;\n ret;\n}\n";
    return s;
}

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
    // gpt-oss: g = min(g, 7), u = clamp(u, -7, 7); g * sigmoid(1.702 g) * (u + 1) -> G
    s += ew("k_oai", ".param .u64 pG, .param .u64 pU",
            " ld.param.u64 %rd1,[pG]; ld.param.u64 %rd2,[pU];\n",
            " mul.wide.u32 %rd3,%r5,4; add.u64 %rd4,%rd1,%rd3; add.u64 %rd5,%rd2,%rd3;\n"
            " ld.global.f32 %f2,[%rd4]; ld.global.f32 %f3,[%rd5]; min.f32 %f2,%f2,0f40E00000; min.f32 %f3,%f3,0f40E00000; max.f32 %f3,%f3,0fC0E00000;\n"
            " mul.f32 %f4,%f2,0fC01D265F; ex2.approx.f32 %f4,%f4; add.f32 %f4,%f4,0f3F800000; div.approx.f32 %f4,%f2,%f4;\n"
            " add.f32 %f3,%f3,0f3F800000; mul.f32 %f4,%f4,%f3; st.global.f32 [%rd4],%f4;\n");
    // y[s*W+c] += b[ids[s]*W+c]: per-slot expert bias
    s += ew("k_xbias", ".param .u64 pY, .param .u64 pB, .param .u64 pI, .param .u32 pW",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pB]; ld.param.u64 %rd6,[pI]; ld.param.u32 %r7,[pW];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd3,%rd6,%rd3; ld.global.u32 %r10,[%rd3]; mad.lo.u32 %r10,%r10,%r7,%r9;\n"
            " mul.wide.u32 %rd4,%r10,4; add.u64 %rd4,%rd2,%rd4; ld.global.f32 %f2,[%rd4];\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd1,%rd5; ld.global.f32 %f3,[%rd5]; add.f32 %f3,%f3,%f2; st.global.f32 [%rd5],%f3;\n");
    // ---- Qwen3.5 Gated DeltaNet: block = value head, thread = value column j holding S[:, j] (128 key rows) in registers.
    // Per token: q, k L2-normed (q / sqrt(128)); S *= exp(g); delta = (v - S^T k) * beta; S += k delta^T; o = S^T q;
    // o RMS-normed over the head * w * silu(z). g = A * softplus(alpha + dt bias), beta = sigmoid(raw). ----
    s += ".visible .entry k_dnet(.param .u64 pX, .param .u64 pZ, .param .u64 pAl, .param .u64 pBe, .param .u64 pA, .param .u64 pDtb, .param .u64 pW, .param .u64 pS, .param .u64 pY, .param .u32 pN, .param .u32 pC, .param .u32 pNk, .param .u32 pHpk, .param .u32 pNh, .param .f32 pE)\n{\n";
    s += " .reg .pred %p<4>; .reg .b32 %r<32>; .reg .f32 %f<40>; .reg .b64 %rd<24>; .reg .f32 %s<128>; .shared .align 16 .f32 qs[128]; .shared .align 16 .f32 ks[128]; .shared .align 4 .f32 red[8];\n";
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pZ]; ld.param.u64 %rd3,[pAl]; ld.param.u64 %rd4,[pBe]; ld.param.u64 %rd5,[pA]; ld.param.u64 %rd6,[pDtb]; ld.param.u64 %rd7,[pW]; ld.param.u64 %rd8,[pS]; ld.param.u64 %rd9,[pY];\n";
    s += " ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pC]; ld.param.u32 %r3,[pNk]; ld.param.u32 %r4,[pHpk]; ld.param.u32 %r5,[pNh]; ld.param.f32 %f1,[pE];\n";
    s += " mov.u32 %r6,%ctaid.x; mov.u32 %r7,%tid.x; setp.eq.u32 %p3,%r4,0; @%p3 rem.u32 %r8,%r6,%r3; @!%p3 div.u32 %r8,%r6,%r4;\n";
    s += " shl.b32 %r9,%r6,14; shl.b32 %r10,%r7,0; add.u32 %r9,%r9,%r10;\n";
    s += " mad.lo.u32 %r11,0,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s0,[%rd10];\n";
    s += " mad.lo.u32 %r11,1,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s1,[%rd10];\n";
    s += " mad.lo.u32 %r11,2,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s2,[%rd10];\n";
    s += " mad.lo.u32 %r11,3,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s3,[%rd10];\n";
    s += " mad.lo.u32 %r11,4,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s4,[%rd10];\n";
    s += " mad.lo.u32 %r11,5,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s5,[%rd10];\n";
    s += " mad.lo.u32 %r11,6,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s6,[%rd10];\n";
    s += " mad.lo.u32 %r11,7,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s7,[%rd10];\n";
    s += " mad.lo.u32 %r11,8,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s8,[%rd10];\n";
    s += " mad.lo.u32 %r11,9,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s9,[%rd10];\n";
    s += " mad.lo.u32 %r11,10,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s10,[%rd10];\n";
    s += " mad.lo.u32 %r11,11,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s11,[%rd10];\n";
    s += " mad.lo.u32 %r11,12,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s12,[%rd10];\n";
    s += " mad.lo.u32 %r11,13,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s13,[%rd10];\n";
    s += " mad.lo.u32 %r11,14,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s14,[%rd10];\n";
    s += " mad.lo.u32 %r11,15,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s15,[%rd10];\n";
    s += " mad.lo.u32 %r11,16,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s16,[%rd10];\n";
    s += " mad.lo.u32 %r11,17,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s17,[%rd10];\n";
    s += " mad.lo.u32 %r11,18,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s18,[%rd10];\n";
    s += " mad.lo.u32 %r11,19,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s19,[%rd10];\n";
    s += " mad.lo.u32 %r11,20,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s20,[%rd10];\n";
    s += " mad.lo.u32 %r11,21,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s21,[%rd10];\n";
    s += " mad.lo.u32 %r11,22,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s22,[%rd10];\n";
    s += " mad.lo.u32 %r11,23,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s23,[%rd10];\n";
    s += " mad.lo.u32 %r11,24,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s24,[%rd10];\n";
    s += " mad.lo.u32 %r11,25,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s25,[%rd10];\n";
    s += " mad.lo.u32 %r11,26,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s26,[%rd10];\n";
    s += " mad.lo.u32 %r11,27,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s27,[%rd10];\n";
    s += " mad.lo.u32 %r11,28,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s28,[%rd10];\n";
    s += " mad.lo.u32 %r11,29,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s29,[%rd10];\n";
    s += " mad.lo.u32 %r11,30,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s30,[%rd10];\n";
    s += " mad.lo.u32 %r11,31,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s31,[%rd10];\n";
    s += " mad.lo.u32 %r11,32,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s32,[%rd10];\n";
    s += " mad.lo.u32 %r11,33,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s33,[%rd10];\n";
    s += " mad.lo.u32 %r11,34,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s34,[%rd10];\n";
    s += " mad.lo.u32 %r11,35,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s35,[%rd10];\n";
    s += " mad.lo.u32 %r11,36,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s36,[%rd10];\n";
    s += " mad.lo.u32 %r11,37,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s37,[%rd10];\n";
    s += " mad.lo.u32 %r11,38,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s38,[%rd10];\n";
    s += " mad.lo.u32 %r11,39,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s39,[%rd10];\n";
    s += " mad.lo.u32 %r11,40,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s40,[%rd10];\n";
    s += " mad.lo.u32 %r11,41,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s41,[%rd10];\n";
    s += " mad.lo.u32 %r11,42,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s42,[%rd10];\n";
    s += " mad.lo.u32 %r11,43,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s43,[%rd10];\n";
    s += " mad.lo.u32 %r11,44,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s44,[%rd10];\n";
    s += " mad.lo.u32 %r11,45,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s45,[%rd10];\n";
    s += " mad.lo.u32 %r11,46,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s46,[%rd10];\n";
    s += " mad.lo.u32 %r11,47,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s47,[%rd10];\n";
    s += " mad.lo.u32 %r11,48,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s48,[%rd10];\n";
    s += " mad.lo.u32 %r11,49,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s49,[%rd10];\n";
    s += " mad.lo.u32 %r11,50,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s50,[%rd10];\n";
    s += " mad.lo.u32 %r11,51,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s51,[%rd10];\n";
    s += " mad.lo.u32 %r11,52,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s52,[%rd10];\n";
    s += " mad.lo.u32 %r11,53,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s53,[%rd10];\n";
    s += " mad.lo.u32 %r11,54,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s54,[%rd10];\n";
    s += " mad.lo.u32 %r11,55,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s55,[%rd10];\n";
    s += " mad.lo.u32 %r11,56,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s56,[%rd10];\n";
    s += " mad.lo.u32 %r11,57,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s57,[%rd10];\n";
    s += " mad.lo.u32 %r11,58,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s58,[%rd10];\n";
    s += " mad.lo.u32 %r11,59,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s59,[%rd10];\n";
    s += " mad.lo.u32 %r11,60,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s60,[%rd10];\n";
    s += " mad.lo.u32 %r11,61,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s61,[%rd10];\n";
    s += " mad.lo.u32 %r11,62,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s62,[%rd10];\n";
    s += " mad.lo.u32 %r11,63,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s63,[%rd10];\n";
    s += " mad.lo.u32 %r11,64,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s64,[%rd10];\n";
    s += " mad.lo.u32 %r11,65,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s65,[%rd10];\n";
    s += " mad.lo.u32 %r11,66,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s66,[%rd10];\n";
    s += " mad.lo.u32 %r11,67,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s67,[%rd10];\n";
    s += " mad.lo.u32 %r11,68,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s68,[%rd10];\n";
    s += " mad.lo.u32 %r11,69,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s69,[%rd10];\n";
    s += " mad.lo.u32 %r11,70,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s70,[%rd10];\n";
    s += " mad.lo.u32 %r11,71,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s71,[%rd10];\n";
    s += " mad.lo.u32 %r11,72,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s72,[%rd10];\n";
    s += " mad.lo.u32 %r11,73,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s73,[%rd10];\n";
    s += " mad.lo.u32 %r11,74,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s74,[%rd10];\n";
    s += " mad.lo.u32 %r11,75,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s75,[%rd10];\n";
    s += " mad.lo.u32 %r11,76,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s76,[%rd10];\n";
    s += " mad.lo.u32 %r11,77,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s77,[%rd10];\n";
    s += " mad.lo.u32 %r11,78,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s78,[%rd10];\n";
    s += " mad.lo.u32 %r11,79,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s79,[%rd10];\n";
    s += " mad.lo.u32 %r11,80,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s80,[%rd10];\n";
    s += " mad.lo.u32 %r11,81,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s81,[%rd10];\n";
    s += " mad.lo.u32 %r11,82,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s82,[%rd10];\n";
    s += " mad.lo.u32 %r11,83,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s83,[%rd10];\n";
    s += " mad.lo.u32 %r11,84,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s84,[%rd10];\n";
    s += " mad.lo.u32 %r11,85,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s85,[%rd10];\n";
    s += " mad.lo.u32 %r11,86,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s86,[%rd10];\n";
    s += " mad.lo.u32 %r11,87,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s87,[%rd10];\n";
    s += " mad.lo.u32 %r11,88,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s88,[%rd10];\n";
    s += " mad.lo.u32 %r11,89,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s89,[%rd10];\n";
    s += " mad.lo.u32 %r11,90,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s90,[%rd10];\n";
    s += " mad.lo.u32 %r11,91,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s91,[%rd10];\n";
    s += " mad.lo.u32 %r11,92,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s92,[%rd10];\n";
    s += " mad.lo.u32 %r11,93,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s93,[%rd10];\n";
    s += " mad.lo.u32 %r11,94,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s94,[%rd10];\n";
    s += " mad.lo.u32 %r11,95,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s95,[%rd10];\n";
    s += " mad.lo.u32 %r11,96,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s96,[%rd10];\n";
    s += " mad.lo.u32 %r11,97,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s97,[%rd10];\n";
    s += " mad.lo.u32 %r11,98,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s98,[%rd10];\n";
    s += " mad.lo.u32 %r11,99,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s99,[%rd10];\n";
    s += " mad.lo.u32 %r11,100,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s100,[%rd10];\n";
    s += " mad.lo.u32 %r11,101,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s101,[%rd10];\n";
    s += " mad.lo.u32 %r11,102,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s102,[%rd10];\n";
    s += " mad.lo.u32 %r11,103,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s103,[%rd10];\n";
    s += " mad.lo.u32 %r11,104,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s104,[%rd10];\n";
    s += " mad.lo.u32 %r11,105,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s105,[%rd10];\n";
    s += " mad.lo.u32 %r11,106,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s106,[%rd10];\n";
    s += " mad.lo.u32 %r11,107,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s107,[%rd10];\n";
    s += " mad.lo.u32 %r11,108,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s108,[%rd10];\n";
    s += " mad.lo.u32 %r11,109,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s109,[%rd10];\n";
    s += " mad.lo.u32 %r11,110,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s110,[%rd10];\n";
    s += " mad.lo.u32 %r11,111,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s111,[%rd10];\n";
    s += " mad.lo.u32 %r11,112,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s112,[%rd10];\n";
    s += " mad.lo.u32 %r11,113,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s113,[%rd10];\n";
    s += " mad.lo.u32 %r11,114,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s114,[%rd10];\n";
    s += " mad.lo.u32 %r11,115,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s115,[%rd10];\n";
    s += " mad.lo.u32 %r11,116,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s116,[%rd10];\n";
    s += " mad.lo.u32 %r11,117,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s117,[%rd10];\n";
    s += " mad.lo.u32 %r11,118,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s118,[%rd10];\n";
    s += " mad.lo.u32 %r11,119,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s119,[%rd10];\n";
    s += " mad.lo.u32 %r11,120,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s120,[%rd10];\n";
    s += " mad.lo.u32 %r11,121,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s121,[%rd10];\n";
    s += " mad.lo.u32 %r11,122,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s122,[%rd10];\n";
    s += " mad.lo.u32 %r11,123,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s123,[%rd10];\n";
    s += " mad.lo.u32 %r11,124,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s124,[%rd10];\n";
    s += " mad.lo.u32 %r11,125,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s125,[%rd10];\n";
    s += " mad.lo.u32 %r11,126,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s126,[%rd10];\n";
    s += " mad.lo.u32 %r11,127,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; ld.global.f32 %s127,[%rd10];\n";
    s += " mul.wide.u32 %rd11,%r6,4; add.u64 %rd12,%rd5,%rd11; ld.global.f32 %f2,[%rd12]; add.u64 %rd13,%rd6,%rd11; ld.global.f32 %f3,[%rd13];\n";
    s += " mul.f32 %f2,%f2,0f3FB8AA3B; mul.wide.u32 %rd14,%r7,4; add.u64 %rd14,%rd7,%rd14; ld.global.f32 %f4,[%rd14];\n";
    s += " and.b32 %r12,%r7,31; setp.eq.u32 %p2,%r12,0; shr.u32 %r13,%r7,5; mov.u32 %r14,red; shl.b32 %r15,%r13,2; add.u32 %r15,%r14,%r15; mov.u32 %r16,qs; mov.u32 %r17,ks; shl.b32 %r18,%r7,2;\n";
    s += " shl.b32 %r19,%r3,7; shl.b32 %r20,%r8,7; add.u32 %r20,%r20,%r7; add.u32 %r21,%r20,%r19; shl.b32 %r22,%r19,1; shl.b32 %r23,%r6,7; add.u32 %r23,%r23,%r22; add.u32 %r23,%r23,%r7;\n";
    s += " mul.lo.u32 %r24,%r5,128; shl.b32 %r25,%r6,7; add.u32 %r25,%r25,%r7; mov.u32 %r26,0;\n";
    s += "LT:\n";
    s += " setp.ge.u32 %p1,%r26,%r1; @%p1 bra ET;\n";
    s += " mul.lo.u32 %r27,%r26,%r2; add.u32 %r28,%r27,%r20; mul.wide.u32 %rd15,%r28,4; add.u64 %rd15,%rd1,%rd15; ld.global.f32 %f5,[%rd15];\n";
    s += " add.u32 %r28,%r27,%r21; mul.wide.u32 %rd15,%r28,4; add.u64 %rd15,%rd1,%rd15; ld.global.f32 %f6,[%rd15];\n";
    s += " add.u32 %r28,%r27,%r23; mul.wide.u32 %rd15,%r28,4; add.u64 %rd15,%rd1,%rd15; ld.global.f32 %f7,[%rd15];\n";
    s += " mul.f32 %f8,%f5,%f5;\n";
    s += " shfl.sync.bfly.b32 %f30,%f8,16,31,0xffffffff; add.f32 %f8,%f8,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f8,8,31,0xffffffff; add.f32 %f8,%f8,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f8,4,31,0xffffffff; add.f32 %f8,%f8,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f8,2,31,0xffffffff; add.f32 %f8,%f8,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f8,1,31,0xffffffff; add.f32 %f8,%f8,%f30;\n";
    s += " @%p2 st.shared.f32 [%r15],%f8; bar.sync 0; ld.shared.f32 %f9,[%r14]; ld.shared.f32 %f30,[%r14+4]; add.f32 %f9,%f9,%f30; ld.shared.f32 %f30,[%r14+8]; add.f32 %f9,%f9,%f30; ld.shared.f32 %f30,[%r14+12]; add.f32 %f9,%f9,%f30; bar.sync 0;\n";
    s += " mul.f32 %f10,%f6,%f6;\n";
    s += " shfl.sync.bfly.b32 %f30,%f10,16,31,0xffffffff; add.f32 %f10,%f10,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f10,8,31,0xffffffff; add.f32 %f10,%f10,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f10,4,31,0xffffffff; add.f32 %f10,%f10,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f10,2,31,0xffffffff; add.f32 %f10,%f10,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f10,1,31,0xffffffff; add.f32 %f10,%f10,%f30;\n";
    s += " @%p2 st.shared.f32 [%r15],%f10; bar.sync 0; ld.shared.f32 %f11,[%r14]; ld.shared.f32 %f30,[%r14+4]; add.f32 %f11,%f11,%f30; ld.shared.f32 %f30,[%r14+8]; add.f32 %f11,%f11,%f30; ld.shared.f32 %f30,[%r14+12]; add.f32 %f11,%f11,%f30; bar.sync 0;\n";
    s += " add.f32 %f9,%f9,0f358637BD; rsqrt.approx.f32 %f9,%f9; mul.f32 %f5,%f5,%f9; mul.f32 %f5,%f5,0f3DB504F3;\n";
    s += " add.f32 %f11,%f11,0f358637BD; rsqrt.approx.f32 %f11,%f11; mul.f32 %f6,%f6,%f11;\n";
    s += " add.u32 %r29,%r16,%r18; st.shared.f32 [%r29],%f5; add.u32 %r29,%r17,%r18; st.shared.f32 [%r29],%f6; bar.sync 0;\n";
    s += " mad.lo.u32 %r28,%r26,%r5,%r6; mul.wide.u32 %rd16,%r28,4; add.u64 %rd17,%rd3,%rd16; ld.global.f32 %f12,[%rd17]; add.u64 %rd18,%rd4,%rd16; ld.global.f32 %f13,[%rd18];\n";
    s += " add.f32 %f12,%f12,%f3; mul.f32 %f14,%f12,0f3FB8AA3B; ex2.approx.f32 %f14,%f14; add.f32 %f14,%f14,0f3F800000; lg2.approx.f32 %f14,%f14; mul.f32 %f14,%f14,0f3F317218; setp.gt.f32 %p3,%f12,0f41A00000; selp.f32 %f12,%f12,%f14,%p3;\n";
    s += " mul.f32 %f12,%f12,%f2; ex2.approx.f32 %f12,%f12;\n";
    s += " mul.f32 %f13,%f13,0fBFB8AA3B; ex2.approx.f32 %f13,%f13; add.f32 %f13,%f13,0f3F800000; rcp.approx.f32 %f13,%f13;\n";
    s += " mov.f32 %f15,0f00000000;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+0];\n";
    s += " mul.f32 %s0,%s0,%f12; fma.rn.f32 %f15,%s0,%f20,%f15;\n";
    s += " mul.f32 %s1,%s1,%f12; fma.rn.f32 %f15,%s1,%f21,%f15;\n";
    s += " mul.f32 %s2,%s2,%f12; fma.rn.f32 %f15,%s2,%f22,%f15;\n";
    s += " mul.f32 %s3,%s3,%f12; fma.rn.f32 %f15,%s3,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+16];\n";
    s += " mul.f32 %s4,%s4,%f12; fma.rn.f32 %f15,%s4,%f20,%f15;\n";
    s += " mul.f32 %s5,%s5,%f12; fma.rn.f32 %f15,%s5,%f21,%f15;\n";
    s += " mul.f32 %s6,%s6,%f12; fma.rn.f32 %f15,%s6,%f22,%f15;\n";
    s += " mul.f32 %s7,%s7,%f12; fma.rn.f32 %f15,%s7,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+32];\n";
    s += " mul.f32 %s8,%s8,%f12; fma.rn.f32 %f15,%s8,%f20,%f15;\n";
    s += " mul.f32 %s9,%s9,%f12; fma.rn.f32 %f15,%s9,%f21,%f15;\n";
    s += " mul.f32 %s10,%s10,%f12; fma.rn.f32 %f15,%s10,%f22,%f15;\n";
    s += " mul.f32 %s11,%s11,%f12; fma.rn.f32 %f15,%s11,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+48];\n";
    s += " mul.f32 %s12,%s12,%f12; fma.rn.f32 %f15,%s12,%f20,%f15;\n";
    s += " mul.f32 %s13,%s13,%f12; fma.rn.f32 %f15,%s13,%f21,%f15;\n";
    s += " mul.f32 %s14,%s14,%f12; fma.rn.f32 %f15,%s14,%f22,%f15;\n";
    s += " mul.f32 %s15,%s15,%f12; fma.rn.f32 %f15,%s15,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+64];\n";
    s += " mul.f32 %s16,%s16,%f12; fma.rn.f32 %f15,%s16,%f20,%f15;\n";
    s += " mul.f32 %s17,%s17,%f12; fma.rn.f32 %f15,%s17,%f21,%f15;\n";
    s += " mul.f32 %s18,%s18,%f12; fma.rn.f32 %f15,%s18,%f22,%f15;\n";
    s += " mul.f32 %s19,%s19,%f12; fma.rn.f32 %f15,%s19,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+80];\n";
    s += " mul.f32 %s20,%s20,%f12; fma.rn.f32 %f15,%s20,%f20,%f15;\n";
    s += " mul.f32 %s21,%s21,%f12; fma.rn.f32 %f15,%s21,%f21,%f15;\n";
    s += " mul.f32 %s22,%s22,%f12; fma.rn.f32 %f15,%s22,%f22,%f15;\n";
    s += " mul.f32 %s23,%s23,%f12; fma.rn.f32 %f15,%s23,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+96];\n";
    s += " mul.f32 %s24,%s24,%f12; fma.rn.f32 %f15,%s24,%f20,%f15;\n";
    s += " mul.f32 %s25,%s25,%f12; fma.rn.f32 %f15,%s25,%f21,%f15;\n";
    s += " mul.f32 %s26,%s26,%f12; fma.rn.f32 %f15,%s26,%f22,%f15;\n";
    s += " mul.f32 %s27,%s27,%f12; fma.rn.f32 %f15,%s27,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+112];\n";
    s += " mul.f32 %s28,%s28,%f12; fma.rn.f32 %f15,%s28,%f20,%f15;\n";
    s += " mul.f32 %s29,%s29,%f12; fma.rn.f32 %f15,%s29,%f21,%f15;\n";
    s += " mul.f32 %s30,%s30,%f12; fma.rn.f32 %f15,%s30,%f22,%f15;\n";
    s += " mul.f32 %s31,%s31,%f12; fma.rn.f32 %f15,%s31,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+128];\n";
    s += " mul.f32 %s32,%s32,%f12; fma.rn.f32 %f15,%s32,%f20,%f15;\n";
    s += " mul.f32 %s33,%s33,%f12; fma.rn.f32 %f15,%s33,%f21,%f15;\n";
    s += " mul.f32 %s34,%s34,%f12; fma.rn.f32 %f15,%s34,%f22,%f15;\n";
    s += " mul.f32 %s35,%s35,%f12; fma.rn.f32 %f15,%s35,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+144];\n";
    s += " mul.f32 %s36,%s36,%f12; fma.rn.f32 %f15,%s36,%f20,%f15;\n";
    s += " mul.f32 %s37,%s37,%f12; fma.rn.f32 %f15,%s37,%f21,%f15;\n";
    s += " mul.f32 %s38,%s38,%f12; fma.rn.f32 %f15,%s38,%f22,%f15;\n";
    s += " mul.f32 %s39,%s39,%f12; fma.rn.f32 %f15,%s39,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+160];\n";
    s += " mul.f32 %s40,%s40,%f12; fma.rn.f32 %f15,%s40,%f20,%f15;\n";
    s += " mul.f32 %s41,%s41,%f12; fma.rn.f32 %f15,%s41,%f21,%f15;\n";
    s += " mul.f32 %s42,%s42,%f12; fma.rn.f32 %f15,%s42,%f22,%f15;\n";
    s += " mul.f32 %s43,%s43,%f12; fma.rn.f32 %f15,%s43,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+176];\n";
    s += " mul.f32 %s44,%s44,%f12; fma.rn.f32 %f15,%s44,%f20,%f15;\n";
    s += " mul.f32 %s45,%s45,%f12; fma.rn.f32 %f15,%s45,%f21,%f15;\n";
    s += " mul.f32 %s46,%s46,%f12; fma.rn.f32 %f15,%s46,%f22,%f15;\n";
    s += " mul.f32 %s47,%s47,%f12; fma.rn.f32 %f15,%s47,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+192];\n";
    s += " mul.f32 %s48,%s48,%f12; fma.rn.f32 %f15,%s48,%f20,%f15;\n";
    s += " mul.f32 %s49,%s49,%f12; fma.rn.f32 %f15,%s49,%f21,%f15;\n";
    s += " mul.f32 %s50,%s50,%f12; fma.rn.f32 %f15,%s50,%f22,%f15;\n";
    s += " mul.f32 %s51,%s51,%f12; fma.rn.f32 %f15,%s51,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+208];\n";
    s += " mul.f32 %s52,%s52,%f12; fma.rn.f32 %f15,%s52,%f20,%f15;\n";
    s += " mul.f32 %s53,%s53,%f12; fma.rn.f32 %f15,%s53,%f21,%f15;\n";
    s += " mul.f32 %s54,%s54,%f12; fma.rn.f32 %f15,%s54,%f22,%f15;\n";
    s += " mul.f32 %s55,%s55,%f12; fma.rn.f32 %f15,%s55,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+224];\n";
    s += " mul.f32 %s56,%s56,%f12; fma.rn.f32 %f15,%s56,%f20,%f15;\n";
    s += " mul.f32 %s57,%s57,%f12; fma.rn.f32 %f15,%s57,%f21,%f15;\n";
    s += " mul.f32 %s58,%s58,%f12; fma.rn.f32 %f15,%s58,%f22,%f15;\n";
    s += " mul.f32 %s59,%s59,%f12; fma.rn.f32 %f15,%s59,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+240];\n";
    s += " mul.f32 %s60,%s60,%f12; fma.rn.f32 %f15,%s60,%f20,%f15;\n";
    s += " mul.f32 %s61,%s61,%f12; fma.rn.f32 %f15,%s61,%f21,%f15;\n";
    s += " mul.f32 %s62,%s62,%f12; fma.rn.f32 %f15,%s62,%f22,%f15;\n";
    s += " mul.f32 %s63,%s63,%f12; fma.rn.f32 %f15,%s63,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+256];\n";
    s += " mul.f32 %s64,%s64,%f12; fma.rn.f32 %f15,%s64,%f20,%f15;\n";
    s += " mul.f32 %s65,%s65,%f12; fma.rn.f32 %f15,%s65,%f21,%f15;\n";
    s += " mul.f32 %s66,%s66,%f12; fma.rn.f32 %f15,%s66,%f22,%f15;\n";
    s += " mul.f32 %s67,%s67,%f12; fma.rn.f32 %f15,%s67,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+272];\n";
    s += " mul.f32 %s68,%s68,%f12; fma.rn.f32 %f15,%s68,%f20,%f15;\n";
    s += " mul.f32 %s69,%s69,%f12; fma.rn.f32 %f15,%s69,%f21,%f15;\n";
    s += " mul.f32 %s70,%s70,%f12; fma.rn.f32 %f15,%s70,%f22,%f15;\n";
    s += " mul.f32 %s71,%s71,%f12; fma.rn.f32 %f15,%s71,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+288];\n";
    s += " mul.f32 %s72,%s72,%f12; fma.rn.f32 %f15,%s72,%f20,%f15;\n";
    s += " mul.f32 %s73,%s73,%f12; fma.rn.f32 %f15,%s73,%f21,%f15;\n";
    s += " mul.f32 %s74,%s74,%f12; fma.rn.f32 %f15,%s74,%f22,%f15;\n";
    s += " mul.f32 %s75,%s75,%f12; fma.rn.f32 %f15,%s75,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+304];\n";
    s += " mul.f32 %s76,%s76,%f12; fma.rn.f32 %f15,%s76,%f20,%f15;\n";
    s += " mul.f32 %s77,%s77,%f12; fma.rn.f32 %f15,%s77,%f21,%f15;\n";
    s += " mul.f32 %s78,%s78,%f12; fma.rn.f32 %f15,%s78,%f22,%f15;\n";
    s += " mul.f32 %s79,%s79,%f12; fma.rn.f32 %f15,%s79,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+320];\n";
    s += " mul.f32 %s80,%s80,%f12; fma.rn.f32 %f15,%s80,%f20,%f15;\n";
    s += " mul.f32 %s81,%s81,%f12; fma.rn.f32 %f15,%s81,%f21,%f15;\n";
    s += " mul.f32 %s82,%s82,%f12; fma.rn.f32 %f15,%s82,%f22,%f15;\n";
    s += " mul.f32 %s83,%s83,%f12; fma.rn.f32 %f15,%s83,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+336];\n";
    s += " mul.f32 %s84,%s84,%f12; fma.rn.f32 %f15,%s84,%f20,%f15;\n";
    s += " mul.f32 %s85,%s85,%f12; fma.rn.f32 %f15,%s85,%f21,%f15;\n";
    s += " mul.f32 %s86,%s86,%f12; fma.rn.f32 %f15,%s86,%f22,%f15;\n";
    s += " mul.f32 %s87,%s87,%f12; fma.rn.f32 %f15,%s87,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+352];\n";
    s += " mul.f32 %s88,%s88,%f12; fma.rn.f32 %f15,%s88,%f20,%f15;\n";
    s += " mul.f32 %s89,%s89,%f12; fma.rn.f32 %f15,%s89,%f21,%f15;\n";
    s += " mul.f32 %s90,%s90,%f12; fma.rn.f32 %f15,%s90,%f22,%f15;\n";
    s += " mul.f32 %s91,%s91,%f12; fma.rn.f32 %f15,%s91,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+368];\n";
    s += " mul.f32 %s92,%s92,%f12; fma.rn.f32 %f15,%s92,%f20,%f15;\n";
    s += " mul.f32 %s93,%s93,%f12; fma.rn.f32 %f15,%s93,%f21,%f15;\n";
    s += " mul.f32 %s94,%s94,%f12; fma.rn.f32 %f15,%s94,%f22,%f15;\n";
    s += " mul.f32 %s95,%s95,%f12; fma.rn.f32 %f15,%s95,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+384];\n";
    s += " mul.f32 %s96,%s96,%f12; fma.rn.f32 %f15,%s96,%f20,%f15;\n";
    s += " mul.f32 %s97,%s97,%f12; fma.rn.f32 %f15,%s97,%f21,%f15;\n";
    s += " mul.f32 %s98,%s98,%f12; fma.rn.f32 %f15,%s98,%f22,%f15;\n";
    s += " mul.f32 %s99,%s99,%f12; fma.rn.f32 %f15,%s99,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+400];\n";
    s += " mul.f32 %s100,%s100,%f12; fma.rn.f32 %f15,%s100,%f20,%f15;\n";
    s += " mul.f32 %s101,%s101,%f12; fma.rn.f32 %f15,%s101,%f21,%f15;\n";
    s += " mul.f32 %s102,%s102,%f12; fma.rn.f32 %f15,%s102,%f22,%f15;\n";
    s += " mul.f32 %s103,%s103,%f12; fma.rn.f32 %f15,%s103,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+416];\n";
    s += " mul.f32 %s104,%s104,%f12; fma.rn.f32 %f15,%s104,%f20,%f15;\n";
    s += " mul.f32 %s105,%s105,%f12; fma.rn.f32 %f15,%s105,%f21,%f15;\n";
    s += " mul.f32 %s106,%s106,%f12; fma.rn.f32 %f15,%s106,%f22,%f15;\n";
    s += " mul.f32 %s107,%s107,%f12; fma.rn.f32 %f15,%s107,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+432];\n";
    s += " mul.f32 %s108,%s108,%f12; fma.rn.f32 %f15,%s108,%f20,%f15;\n";
    s += " mul.f32 %s109,%s109,%f12; fma.rn.f32 %f15,%s109,%f21,%f15;\n";
    s += " mul.f32 %s110,%s110,%f12; fma.rn.f32 %f15,%s110,%f22,%f15;\n";
    s += " mul.f32 %s111,%s111,%f12; fma.rn.f32 %f15,%s111,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+448];\n";
    s += " mul.f32 %s112,%s112,%f12; fma.rn.f32 %f15,%s112,%f20,%f15;\n";
    s += " mul.f32 %s113,%s113,%f12; fma.rn.f32 %f15,%s113,%f21,%f15;\n";
    s += " mul.f32 %s114,%s114,%f12; fma.rn.f32 %f15,%s114,%f22,%f15;\n";
    s += " mul.f32 %s115,%s115,%f12; fma.rn.f32 %f15,%s115,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+464];\n";
    s += " mul.f32 %s116,%s116,%f12; fma.rn.f32 %f15,%s116,%f20,%f15;\n";
    s += " mul.f32 %s117,%s117,%f12; fma.rn.f32 %f15,%s117,%f21,%f15;\n";
    s += " mul.f32 %s118,%s118,%f12; fma.rn.f32 %f15,%s118,%f22,%f15;\n";
    s += " mul.f32 %s119,%s119,%f12; fma.rn.f32 %f15,%s119,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+480];\n";
    s += " mul.f32 %s120,%s120,%f12; fma.rn.f32 %f15,%s120,%f20,%f15;\n";
    s += " mul.f32 %s121,%s121,%f12; fma.rn.f32 %f15,%s121,%f21,%f15;\n";
    s += " mul.f32 %s122,%s122,%f12; fma.rn.f32 %f15,%s122,%f22,%f15;\n";
    s += " mul.f32 %s123,%s123,%f12; fma.rn.f32 %f15,%s123,%f23,%f15;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+496];\n";
    s += " mul.f32 %s124,%s124,%f12; fma.rn.f32 %f15,%s124,%f20,%f15;\n";
    s += " mul.f32 %s125,%s125,%f12; fma.rn.f32 %f15,%s125,%f21,%f15;\n";
    s += " mul.f32 %s126,%s126,%f12; fma.rn.f32 %f15,%s126,%f22,%f15;\n";
    s += " mul.f32 %s127,%s127,%f12; fma.rn.f32 %f15,%s127,%f23,%f15;\n";
    s += " sub.f32 %f16,%f7,%f15; mul.f32 %f16,%f16,%f13; mov.f32 %f17,0f00000000;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+0]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+0];\n";
    s += " fma.rn.f32 %s0,%f20,%f16,%s0; fma.rn.f32 %f17,%s0,%f24,%f17;\n";
    s += " fma.rn.f32 %s1,%f21,%f16,%s1; fma.rn.f32 %f17,%s1,%f25,%f17;\n";
    s += " fma.rn.f32 %s2,%f22,%f16,%s2; fma.rn.f32 %f17,%s2,%f26,%f17;\n";
    s += " fma.rn.f32 %s3,%f23,%f16,%s3; fma.rn.f32 %f17,%s3,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+16]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+16];\n";
    s += " fma.rn.f32 %s4,%f20,%f16,%s4; fma.rn.f32 %f17,%s4,%f24,%f17;\n";
    s += " fma.rn.f32 %s5,%f21,%f16,%s5; fma.rn.f32 %f17,%s5,%f25,%f17;\n";
    s += " fma.rn.f32 %s6,%f22,%f16,%s6; fma.rn.f32 %f17,%s6,%f26,%f17;\n";
    s += " fma.rn.f32 %s7,%f23,%f16,%s7; fma.rn.f32 %f17,%s7,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+32]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+32];\n";
    s += " fma.rn.f32 %s8,%f20,%f16,%s8; fma.rn.f32 %f17,%s8,%f24,%f17;\n";
    s += " fma.rn.f32 %s9,%f21,%f16,%s9; fma.rn.f32 %f17,%s9,%f25,%f17;\n";
    s += " fma.rn.f32 %s10,%f22,%f16,%s10; fma.rn.f32 %f17,%s10,%f26,%f17;\n";
    s += " fma.rn.f32 %s11,%f23,%f16,%s11; fma.rn.f32 %f17,%s11,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+48]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+48];\n";
    s += " fma.rn.f32 %s12,%f20,%f16,%s12; fma.rn.f32 %f17,%s12,%f24,%f17;\n";
    s += " fma.rn.f32 %s13,%f21,%f16,%s13; fma.rn.f32 %f17,%s13,%f25,%f17;\n";
    s += " fma.rn.f32 %s14,%f22,%f16,%s14; fma.rn.f32 %f17,%s14,%f26,%f17;\n";
    s += " fma.rn.f32 %s15,%f23,%f16,%s15; fma.rn.f32 %f17,%s15,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+64]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+64];\n";
    s += " fma.rn.f32 %s16,%f20,%f16,%s16; fma.rn.f32 %f17,%s16,%f24,%f17;\n";
    s += " fma.rn.f32 %s17,%f21,%f16,%s17; fma.rn.f32 %f17,%s17,%f25,%f17;\n";
    s += " fma.rn.f32 %s18,%f22,%f16,%s18; fma.rn.f32 %f17,%s18,%f26,%f17;\n";
    s += " fma.rn.f32 %s19,%f23,%f16,%s19; fma.rn.f32 %f17,%s19,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+80]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+80];\n";
    s += " fma.rn.f32 %s20,%f20,%f16,%s20; fma.rn.f32 %f17,%s20,%f24,%f17;\n";
    s += " fma.rn.f32 %s21,%f21,%f16,%s21; fma.rn.f32 %f17,%s21,%f25,%f17;\n";
    s += " fma.rn.f32 %s22,%f22,%f16,%s22; fma.rn.f32 %f17,%s22,%f26,%f17;\n";
    s += " fma.rn.f32 %s23,%f23,%f16,%s23; fma.rn.f32 %f17,%s23,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+96]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+96];\n";
    s += " fma.rn.f32 %s24,%f20,%f16,%s24; fma.rn.f32 %f17,%s24,%f24,%f17;\n";
    s += " fma.rn.f32 %s25,%f21,%f16,%s25; fma.rn.f32 %f17,%s25,%f25,%f17;\n";
    s += " fma.rn.f32 %s26,%f22,%f16,%s26; fma.rn.f32 %f17,%s26,%f26,%f17;\n";
    s += " fma.rn.f32 %s27,%f23,%f16,%s27; fma.rn.f32 %f17,%s27,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+112]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+112];\n";
    s += " fma.rn.f32 %s28,%f20,%f16,%s28; fma.rn.f32 %f17,%s28,%f24,%f17;\n";
    s += " fma.rn.f32 %s29,%f21,%f16,%s29; fma.rn.f32 %f17,%s29,%f25,%f17;\n";
    s += " fma.rn.f32 %s30,%f22,%f16,%s30; fma.rn.f32 %f17,%s30,%f26,%f17;\n";
    s += " fma.rn.f32 %s31,%f23,%f16,%s31; fma.rn.f32 %f17,%s31,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+128]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+128];\n";
    s += " fma.rn.f32 %s32,%f20,%f16,%s32; fma.rn.f32 %f17,%s32,%f24,%f17;\n";
    s += " fma.rn.f32 %s33,%f21,%f16,%s33; fma.rn.f32 %f17,%s33,%f25,%f17;\n";
    s += " fma.rn.f32 %s34,%f22,%f16,%s34; fma.rn.f32 %f17,%s34,%f26,%f17;\n";
    s += " fma.rn.f32 %s35,%f23,%f16,%s35; fma.rn.f32 %f17,%s35,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+144]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+144];\n";
    s += " fma.rn.f32 %s36,%f20,%f16,%s36; fma.rn.f32 %f17,%s36,%f24,%f17;\n";
    s += " fma.rn.f32 %s37,%f21,%f16,%s37; fma.rn.f32 %f17,%s37,%f25,%f17;\n";
    s += " fma.rn.f32 %s38,%f22,%f16,%s38; fma.rn.f32 %f17,%s38,%f26,%f17;\n";
    s += " fma.rn.f32 %s39,%f23,%f16,%s39; fma.rn.f32 %f17,%s39,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+160]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+160];\n";
    s += " fma.rn.f32 %s40,%f20,%f16,%s40; fma.rn.f32 %f17,%s40,%f24,%f17;\n";
    s += " fma.rn.f32 %s41,%f21,%f16,%s41; fma.rn.f32 %f17,%s41,%f25,%f17;\n";
    s += " fma.rn.f32 %s42,%f22,%f16,%s42; fma.rn.f32 %f17,%s42,%f26,%f17;\n";
    s += " fma.rn.f32 %s43,%f23,%f16,%s43; fma.rn.f32 %f17,%s43,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+176]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+176];\n";
    s += " fma.rn.f32 %s44,%f20,%f16,%s44; fma.rn.f32 %f17,%s44,%f24,%f17;\n";
    s += " fma.rn.f32 %s45,%f21,%f16,%s45; fma.rn.f32 %f17,%s45,%f25,%f17;\n";
    s += " fma.rn.f32 %s46,%f22,%f16,%s46; fma.rn.f32 %f17,%s46,%f26,%f17;\n";
    s += " fma.rn.f32 %s47,%f23,%f16,%s47; fma.rn.f32 %f17,%s47,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+192]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+192];\n";
    s += " fma.rn.f32 %s48,%f20,%f16,%s48; fma.rn.f32 %f17,%s48,%f24,%f17;\n";
    s += " fma.rn.f32 %s49,%f21,%f16,%s49; fma.rn.f32 %f17,%s49,%f25,%f17;\n";
    s += " fma.rn.f32 %s50,%f22,%f16,%s50; fma.rn.f32 %f17,%s50,%f26,%f17;\n";
    s += " fma.rn.f32 %s51,%f23,%f16,%s51; fma.rn.f32 %f17,%s51,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+208]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+208];\n";
    s += " fma.rn.f32 %s52,%f20,%f16,%s52; fma.rn.f32 %f17,%s52,%f24,%f17;\n";
    s += " fma.rn.f32 %s53,%f21,%f16,%s53; fma.rn.f32 %f17,%s53,%f25,%f17;\n";
    s += " fma.rn.f32 %s54,%f22,%f16,%s54; fma.rn.f32 %f17,%s54,%f26,%f17;\n";
    s += " fma.rn.f32 %s55,%f23,%f16,%s55; fma.rn.f32 %f17,%s55,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+224]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+224];\n";
    s += " fma.rn.f32 %s56,%f20,%f16,%s56; fma.rn.f32 %f17,%s56,%f24,%f17;\n";
    s += " fma.rn.f32 %s57,%f21,%f16,%s57; fma.rn.f32 %f17,%s57,%f25,%f17;\n";
    s += " fma.rn.f32 %s58,%f22,%f16,%s58; fma.rn.f32 %f17,%s58,%f26,%f17;\n";
    s += " fma.rn.f32 %s59,%f23,%f16,%s59; fma.rn.f32 %f17,%s59,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+240]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+240];\n";
    s += " fma.rn.f32 %s60,%f20,%f16,%s60; fma.rn.f32 %f17,%s60,%f24,%f17;\n";
    s += " fma.rn.f32 %s61,%f21,%f16,%s61; fma.rn.f32 %f17,%s61,%f25,%f17;\n";
    s += " fma.rn.f32 %s62,%f22,%f16,%s62; fma.rn.f32 %f17,%s62,%f26,%f17;\n";
    s += " fma.rn.f32 %s63,%f23,%f16,%s63; fma.rn.f32 %f17,%s63,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+256]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+256];\n";
    s += " fma.rn.f32 %s64,%f20,%f16,%s64; fma.rn.f32 %f17,%s64,%f24,%f17;\n";
    s += " fma.rn.f32 %s65,%f21,%f16,%s65; fma.rn.f32 %f17,%s65,%f25,%f17;\n";
    s += " fma.rn.f32 %s66,%f22,%f16,%s66; fma.rn.f32 %f17,%s66,%f26,%f17;\n";
    s += " fma.rn.f32 %s67,%f23,%f16,%s67; fma.rn.f32 %f17,%s67,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+272]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+272];\n";
    s += " fma.rn.f32 %s68,%f20,%f16,%s68; fma.rn.f32 %f17,%s68,%f24,%f17;\n";
    s += " fma.rn.f32 %s69,%f21,%f16,%s69; fma.rn.f32 %f17,%s69,%f25,%f17;\n";
    s += " fma.rn.f32 %s70,%f22,%f16,%s70; fma.rn.f32 %f17,%s70,%f26,%f17;\n";
    s += " fma.rn.f32 %s71,%f23,%f16,%s71; fma.rn.f32 %f17,%s71,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+288]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+288];\n";
    s += " fma.rn.f32 %s72,%f20,%f16,%s72; fma.rn.f32 %f17,%s72,%f24,%f17;\n";
    s += " fma.rn.f32 %s73,%f21,%f16,%s73; fma.rn.f32 %f17,%s73,%f25,%f17;\n";
    s += " fma.rn.f32 %s74,%f22,%f16,%s74; fma.rn.f32 %f17,%s74,%f26,%f17;\n";
    s += " fma.rn.f32 %s75,%f23,%f16,%s75; fma.rn.f32 %f17,%s75,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+304]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+304];\n";
    s += " fma.rn.f32 %s76,%f20,%f16,%s76; fma.rn.f32 %f17,%s76,%f24,%f17;\n";
    s += " fma.rn.f32 %s77,%f21,%f16,%s77; fma.rn.f32 %f17,%s77,%f25,%f17;\n";
    s += " fma.rn.f32 %s78,%f22,%f16,%s78; fma.rn.f32 %f17,%s78,%f26,%f17;\n";
    s += " fma.rn.f32 %s79,%f23,%f16,%s79; fma.rn.f32 %f17,%s79,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+320]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+320];\n";
    s += " fma.rn.f32 %s80,%f20,%f16,%s80; fma.rn.f32 %f17,%s80,%f24,%f17;\n";
    s += " fma.rn.f32 %s81,%f21,%f16,%s81; fma.rn.f32 %f17,%s81,%f25,%f17;\n";
    s += " fma.rn.f32 %s82,%f22,%f16,%s82; fma.rn.f32 %f17,%s82,%f26,%f17;\n";
    s += " fma.rn.f32 %s83,%f23,%f16,%s83; fma.rn.f32 %f17,%s83,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+336]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+336];\n";
    s += " fma.rn.f32 %s84,%f20,%f16,%s84; fma.rn.f32 %f17,%s84,%f24,%f17;\n";
    s += " fma.rn.f32 %s85,%f21,%f16,%s85; fma.rn.f32 %f17,%s85,%f25,%f17;\n";
    s += " fma.rn.f32 %s86,%f22,%f16,%s86; fma.rn.f32 %f17,%s86,%f26,%f17;\n";
    s += " fma.rn.f32 %s87,%f23,%f16,%s87; fma.rn.f32 %f17,%s87,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+352]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+352];\n";
    s += " fma.rn.f32 %s88,%f20,%f16,%s88; fma.rn.f32 %f17,%s88,%f24,%f17;\n";
    s += " fma.rn.f32 %s89,%f21,%f16,%s89; fma.rn.f32 %f17,%s89,%f25,%f17;\n";
    s += " fma.rn.f32 %s90,%f22,%f16,%s90; fma.rn.f32 %f17,%s90,%f26,%f17;\n";
    s += " fma.rn.f32 %s91,%f23,%f16,%s91; fma.rn.f32 %f17,%s91,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+368]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+368];\n";
    s += " fma.rn.f32 %s92,%f20,%f16,%s92; fma.rn.f32 %f17,%s92,%f24,%f17;\n";
    s += " fma.rn.f32 %s93,%f21,%f16,%s93; fma.rn.f32 %f17,%s93,%f25,%f17;\n";
    s += " fma.rn.f32 %s94,%f22,%f16,%s94; fma.rn.f32 %f17,%s94,%f26,%f17;\n";
    s += " fma.rn.f32 %s95,%f23,%f16,%s95; fma.rn.f32 %f17,%s95,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+384]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+384];\n";
    s += " fma.rn.f32 %s96,%f20,%f16,%s96; fma.rn.f32 %f17,%s96,%f24,%f17;\n";
    s += " fma.rn.f32 %s97,%f21,%f16,%s97; fma.rn.f32 %f17,%s97,%f25,%f17;\n";
    s += " fma.rn.f32 %s98,%f22,%f16,%s98; fma.rn.f32 %f17,%s98,%f26,%f17;\n";
    s += " fma.rn.f32 %s99,%f23,%f16,%s99; fma.rn.f32 %f17,%s99,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+400]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+400];\n";
    s += " fma.rn.f32 %s100,%f20,%f16,%s100; fma.rn.f32 %f17,%s100,%f24,%f17;\n";
    s += " fma.rn.f32 %s101,%f21,%f16,%s101; fma.rn.f32 %f17,%s101,%f25,%f17;\n";
    s += " fma.rn.f32 %s102,%f22,%f16,%s102; fma.rn.f32 %f17,%s102,%f26,%f17;\n";
    s += " fma.rn.f32 %s103,%f23,%f16,%s103; fma.rn.f32 %f17,%s103,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+416]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+416];\n";
    s += " fma.rn.f32 %s104,%f20,%f16,%s104; fma.rn.f32 %f17,%s104,%f24,%f17;\n";
    s += " fma.rn.f32 %s105,%f21,%f16,%s105; fma.rn.f32 %f17,%s105,%f25,%f17;\n";
    s += " fma.rn.f32 %s106,%f22,%f16,%s106; fma.rn.f32 %f17,%s106,%f26,%f17;\n";
    s += " fma.rn.f32 %s107,%f23,%f16,%s107; fma.rn.f32 %f17,%s107,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+432]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+432];\n";
    s += " fma.rn.f32 %s108,%f20,%f16,%s108; fma.rn.f32 %f17,%s108,%f24,%f17;\n";
    s += " fma.rn.f32 %s109,%f21,%f16,%s109; fma.rn.f32 %f17,%s109,%f25,%f17;\n";
    s += " fma.rn.f32 %s110,%f22,%f16,%s110; fma.rn.f32 %f17,%s110,%f26,%f17;\n";
    s += " fma.rn.f32 %s111,%f23,%f16,%s111; fma.rn.f32 %f17,%s111,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+448]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+448];\n";
    s += " fma.rn.f32 %s112,%f20,%f16,%s112; fma.rn.f32 %f17,%s112,%f24,%f17;\n";
    s += " fma.rn.f32 %s113,%f21,%f16,%s113; fma.rn.f32 %f17,%s113,%f25,%f17;\n";
    s += " fma.rn.f32 %s114,%f22,%f16,%s114; fma.rn.f32 %f17,%s114,%f26,%f17;\n";
    s += " fma.rn.f32 %s115,%f23,%f16,%s115; fma.rn.f32 %f17,%s115,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+464]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+464];\n";
    s += " fma.rn.f32 %s116,%f20,%f16,%s116; fma.rn.f32 %f17,%s116,%f24,%f17;\n";
    s += " fma.rn.f32 %s117,%f21,%f16,%s117; fma.rn.f32 %f17,%s117,%f25,%f17;\n";
    s += " fma.rn.f32 %s118,%f22,%f16,%s118; fma.rn.f32 %f17,%s118,%f26,%f17;\n";
    s += " fma.rn.f32 %s119,%f23,%f16,%s119; fma.rn.f32 %f17,%s119,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+480]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+480];\n";
    s += " fma.rn.f32 %s120,%f20,%f16,%s120; fma.rn.f32 %f17,%s120,%f24,%f17;\n";
    s += " fma.rn.f32 %s121,%f21,%f16,%s121; fma.rn.f32 %f17,%s121,%f25,%f17;\n";
    s += " fma.rn.f32 %s122,%f22,%f16,%s122; fma.rn.f32 %f17,%s122,%f26,%f17;\n";
    s += " fma.rn.f32 %s123,%f23,%f16,%s123; fma.rn.f32 %f17,%s123,%f27,%f17;\n";
    s += " ld.shared.v4.f32 {%f20,%f21,%f22,%f23},[ks+496]; ld.shared.v4.f32 {%f24,%f25,%f26,%f27},[qs+496];\n";
    s += " fma.rn.f32 %s124,%f20,%f16,%s124; fma.rn.f32 %f17,%s124,%f24,%f17;\n";
    s += " fma.rn.f32 %s125,%f21,%f16,%s125; fma.rn.f32 %f17,%s125,%f25,%f17;\n";
    s += " fma.rn.f32 %s126,%f22,%f16,%s126; fma.rn.f32 %f17,%s126,%f26,%f17;\n";
    s += " fma.rn.f32 %s127,%f23,%f16,%s127; fma.rn.f32 %f17,%s127,%f27,%f17;\n";
    s += " mul.f32 %f18,%f17,%f17;\n";
    s += " shfl.sync.bfly.b32 %f30,%f18,16,31,0xffffffff; add.f32 %f18,%f18,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f18,8,31,0xffffffff; add.f32 %f18,%f18,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f18,4,31,0xffffffff; add.f32 %f18,%f18,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f18,2,31,0xffffffff; add.f32 %f18,%f18,%f30;\n";
    s += " shfl.sync.bfly.b32 %f30,%f18,1,31,0xffffffff; add.f32 %f18,%f18,%f30;\n";
    s += " @%p2 st.shared.f32 [%r15],%f18; bar.sync 0; ld.shared.f32 %f19,[%r14]; ld.shared.f32 %f30,[%r14+4]; add.f32 %f19,%f19,%f30; ld.shared.f32 %f30,[%r14+8]; add.f32 %f19,%f19,%f30; ld.shared.f32 %f30,[%r14+12]; add.f32 %f19,%f19,%f30; bar.sync 0;\n";
    s += " mul.f32 %f19,%f19,0f3C000000; add.f32 %f19,%f19,%f1; rsqrt.approx.f32 %f19,%f19; mul.f32 %f17,%f17,%f19; mul.f32 %f17,%f17,%f4;\n";
    s += " mad.lo.u32 %r28,%r26,%r24,%r25; mul.wide.u32 %rd19,%r28,4; add.u64 %rd20,%rd2,%rd19; ld.global.f32 %f28,[%rd20];\n";
    s += " mul.f32 %f29,%f28,0fBFB8AA3B; ex2.approx.f32 %f29,%f29; add.f32 %f29,%f29,0f3F800000; div.approx.f32 %f28,%f28,%f29; mul.f32 %f17,%f17,%f28;\n";
    s += " add.u64 %rd21,%rd9,%rd19; st.global.f32 [%rd21],%f17; add.u32 %r26,%r26,1; bra LT;\n";
    s += "ET:\n";
    s += " mad.lo.u32 %r11,0,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s0;\n";
    s += " mad.lo.u32 %r11,1,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s1;\n";
    s += " mad.lo.u32 %r11,2,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s2;\n";
    s += " mad.lo.u32 %r11,3,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s3;\n";
    s += " mad.lo.u32 %r11,4,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s4;\n";
    s += " mad.lo.u32 %r11,5,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s5;\n";
    s += " mad.lo.u32 %r11,6,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s6;\n";
    s += " mad.lo.u32 %r11,7,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s7;\n";
    s += " mad.lo.u32 %r11,8,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s8;\n";
    s += " mad.lo.u32 %r11,9,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s9;\n";
    s += " mad.lo.u32 %r11,10,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s10;\n";
    s += " mad.lo.u32 %r11,11,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s11;\n";
    s += " mad.lo.u32 %r11,12,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s12;\n";
    s += " mad.lo.u32 %r11,13,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s13;\n";
    s += " mad.lo.u32 %r11,14,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s14;\n";
    s += " mad.lo.u32 %r11,15,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s15;\n";
    s += " mad.lo.u32 %r11,16,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s16;\n";
    s += " mad.lo.u32 %r11,17,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s17;\n";
    s += " mad.lo.u32 %r11,18,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s18;\n";
    s += " mad.lo.u32 %r11,19,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s19;\n";
    s += " mad.lo.u32 %r11,20,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s20;\n";
    s += " mad.lo.u32 %r11,21,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s21;\n";
    s += " mad.lo.u32 %r11,22,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s22;\n";
    s += " mad.lo.u32 %r11,23,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s23;\n";
    s += " mad.lo.u32 %r11,24,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s24;\n";
    s += " mad.lo.u32 %r11,25,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s25;\n";
    s += " mad.lo.u32 %r11,26,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s26;\n";
    s += " mad.lo.u32 %r11,27,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s27;\n";
    s += " mad.lo.u32 %r11,28,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s28;\n";
    s += " mad.lo.u32 %r11,29,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s29;\n";
    s += " mad.lo.u32 %r11,30,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s30;\n";
    s += " mad.lo.u32 %r11,31,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s31;\n";
    s += " mad.lo.u32 %r11,32,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s32;\n";
    s += " mad.lo.u32 %r11,33,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s33;\n";
    s += " mad.lo.u32 %r11,34,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s34;\n";
    s += " mad.lo.u32 %r11,35,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s35;\n";
    s += " mad.lo.u32 %r11,36,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s36;\n";
    s += " mad.lo.u32 %r11,37,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s37;\n";
    s += " mad.lo.u32 %r11,38,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s38;\n";
    s += " mad.lo.u32 %r11,39,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s39;\n";
    s += " mad.lo.u32 %r11,40,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s40;\n";
    s += " mad.lo.u32 %r11,41,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s41;\n";
    s += " mad.lo.u32 %r11,42,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s42;\n";
    s += " mad.lo.u32 %r11,43,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s43;\n";
    s += " mad.lo.u32 %r11,44,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s44;\n";
    s += " mad.lo.u32 %r11,45,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s45;\n";
    s += " mad.lo.u32 %r11,46,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s46;\n";
    s += " mad.lo.u32 %r11,47,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s47;\n";
    s += " mad.lo.u32 %r11,48,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s48;\n";
    s += " mad.lo.u32 %r11,49,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s49;\n";
    s += " mad.lo.u32 %r11,50,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s50;\n";
    s += " mad.lo.u32 %r11,51,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s51;\n";
    s += " mad.lo.u32 %r11,52,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s52;\n";
    s += " mad.lo.u32 %r11,53,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s53;\n";
    s += " mad.lo.u32 %r11,54,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s54;\n";
    s += " mad.lo.u32 %r11,55,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s55;\n";
    s += " mad.lo.u32 %r11,56,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s56;\n";
    s += " mad.lo.u32 %r11,57,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s57;\n";
    s += " mad.lo.u32 %r11,58,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s58;\n";
    s += " mad.lo.u32 %r11,59,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s59;\n";
    s += " mad.lo.u32 %r11,60,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s60;\n";
    s += " mad.lo.u32 %r11,61,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s61;\n";
    s += " mad.lo.u32 %r11,62,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s62;\n";
    s += " mad.lo.u32 %r11,63,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s63;\n";
    s += " mad.lo.u32 %r11,64,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s64;\n";
    s += " mad.lo.u32 %r11,65,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s65;\n";
    s += " mad.lo.u32 %r11,66,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s66;\n";
    s += " mad.lo.u32 %r11,67,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s67;\n";
    s += " mad.lo.u32 %r11,68,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s68;\n";
    s += " mad.lo.u32 %r11,69,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s69;\n";
    s += " mad.lo.u32 %r11,70,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s70;\n";
    s += " mad.lo.u32 %r11,71,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s71;\n";
    s += " mad.lo.u32 %r11,72,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s72;\n";
    s += " mad.lo.u32 %r11,73,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s73;\n";
    s += " mad.lo.u32 %r11,74,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s74;\n";
    s += " mad.lo.u32 %r11,75,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s75;\n";
    s += " mad.lo.u32 %r11,76,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s76;\n";
    s += " mad.lo.u32 %r11,77,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s77;\n";
    s += " mad.lo.u32 %r11,78,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s78;\n";
    s += " mad.lo.u32 %r11,79,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s79;\n";
    s += " mad.lo.u32 %r11,80,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s80;\n";
    s += " mad.lo.u32 %r11,81,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s81;\n";
    s += " mad.lo.u32 %r11,82,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s82;\n";
    s += " mad.lo.u32 %r11,83,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s83;\n";
    s += " mad.lo.u32 %r11,84,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s84;\n";
    s += " mad.lo.u32 %r11,85,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s85;\n";
    s += " mad.lo.u32 %r11,86,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s86;\n";
    s += " mad.lo.u32 %r11,87,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s87;\n";
    s += " mad.lo.u32 %r11,88,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s88;\n";
    s += " mad.lo.u32 %r11,89,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s89;\n";
    s += " mad.lo.u32 %r11,90,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s90;\n";
    s += " mad.lo.u32 %r11,91,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s91;\n";
    s += " mad.lo.u32 %r11,92,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s92;\n";
    s += " mad.lo.u32 %r11,93,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s93;\n";
    s += " mad.lo.u32 %r11,94,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s94;\n";
    s += " mad.lo.u32 %r11,95,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s95;\n";
    s += " mad.lo.u32 %r11,96,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s96;\n";
    s += " mad.lo.u32 %r11,97,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s97;\n";
    s += " mad.lo.u32 %r11,98,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s98;\n";
    s += " mad.lo.u32 %r11,99,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s99;\n";
    s += " mad.lo.u32 %r11,100,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s100;\n";
    s += " mad.lo.u32 %r11,101,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s101;\n";
    s += " mad.lo.u32 %r11,102,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s102;\n";
    s += " mad.lo.u32 %r11,103,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s103;\n";
    s += " mad.lo.u32 %r11,104,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s104;\n";
    s += " mad.lo.u32 %r11,105,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s105;\n";
    s += " mad.lo.u32 %r11,106,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s106;\n";
    s += " mad.lo.u32 %r11,107,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s107;\n";
    s += " mad.lo.u32 %r11,108,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s108;\n";
    s += " mad.lo.u32 %r11,109,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s109;\n";
    s += " mad.lo.u32 %r11,110,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s110;\n";
    s += " mad.lo.u32 %r11,111,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s111;\n";
    s += " mad.lo.u32 %r11,112,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s112;\n";
    s += " mad.lo.u32 %r11,113,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s113;\n";
    s += " mad.lo.u32 %r11,114,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s114;\n";
    s += " mad.lo.u32 %r11,115,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s115;\n";
    s += " mad.lo.u32 %r11,116,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s116;\n";
    s += " mad.lo.u32 %r11,117,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s117;\n";
    s += " mad.lo.u32 %r11,118,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s118;\n";
    s += " mad.lo.u32 %r11,119,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s119;\n";
    s += " mad.lo.u32 %r11,120,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s120;\n";
    s += " mad.lo.u32 %r11,121,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s121;\n";
    s += " mad.lo.u32 %r11,122,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s122;\n";
    s += " mad.lo.u32 %r11,123,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s123;\n";
    s += " mad.lo.u32 %r11,124,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s124;\n";
    s += " mad.lo.u32 %r11,125,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s125;\n";
    s += " mad.lo.u32 %r11,126,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s126;\n";
    s += " mad.lo.u32 %r11,127,128,%r9; mul.wide.u32 %rd10,%r11,4; add.u64 %rd10,%rd8,%rd10; st.global.f32 [%rd10],%s127;\n";
    s += " ret;\n}\n";
    // ---- LFM2 short convolution: one thread per channel, tokens in order. BCX rows [b | c | x] of 3 pD;
    // v = b * x, conv over the last 3 v (2 kept as state), y = c * conv -> Y [N][pD] ----
    s += ".visible .entry k_sconv(.param .u64 pBCX, .param .u64 pW, .param .u64 pS, .param .u64 pY, .param .u32 pN, .param .u32 pD)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pBCX]; ld.param.u64 %rd2,[pW]; ld.param.u64 %rd3,[pS]; ld.param.u64 %rd4,[pY]; ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pD];\n"
         " mov.u32 %r3,%ctaid.x; mov.u32 %r4,%tid.x; mad.lo.u32 %r3,%r3,256,%r4; setp.ge.u32 %p1,%r3,%r2; @%p1 bra DONE;\n"
         " mul.lo.u32 %r5,%r3,3; mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd2,%rd5; ld.global.f32 %f1,[%rd5]; ld.global.f32 %f2,[%rd5+4]; ld.global.f32 %f3,[%rd5+8];\n"
         " mul.wide.u32 %rd6,%r3,4; add.u64 %rd6,%rd3,%rd6; mul.wide.u32 %rd7,%r2,4; ld.global.f32 %f4,[%rd6]; add.u64 %rd8,%rd6,%rd7; ld.global.f32 %f5,[%rd8];\n"
         " mul.lo.u32 %r6,%r2,3; mov.u32 %r7,0;\n"
         "LT:\n setp.ge.u32 %p2,%r7,%r1; @%p2 bra ET;\n"
         " mad.lo.u32 %r8,%r7,%r6,%r3; mul.wide.u32 %rd9,%r8,4; add.u64 %rd9,%rd1,%rd9; ld.global.f32 %f6,[%rd9]; add.u64 %rd10,%rd9,%rd7; ld.global.f32 %f7,[%rd10]; add.u64 %rd11,%rd10,%rd7; ld.global.f32 %f8,[%rd11];\n"
         " mul.f32 %f9,%f6,%f8; mul.f32 %f10,%f1,%f4; fma.rn.f32 %f10,%f2,%f5,%f10; fma.rn.f32 %f10,%f3,%f9,%f10; mov.f32 %f4,%f5; mov.f32 %f5,%f9; mul.f32 %f10,%f10,%f7;\n"
         " mad.lo.u32 %r9,%r7,%r2,%r3; mul.wide.u32 %rd12,%r9,4; add.u64 %rd12,%rd4,%rd12; st.global.f32 [%rd12],%f10;\n"
         " add.u32 %r7,%r7,1; bra LT;\nET:\n st.global.f32 [%rd6],%f4; st.global.f32 [%rd8],%f5;\n"
         "DONE:\n ret;\n}\n";
    // ---- Mamba-2 (Granite 4.0 H) ----
    // k_ssm_conv: one thread per xBC channel, tokens in order. Causal depthwise conv (kernel 4) with the last 3 inputs as
    // state, + bias, SiLU. In: ZX rows of pLd floats, xBC at column pOff. Out: [N][C]. State: [3][C], updated.
    s += ".visible .entry k_ssm_conv(.param .u64 pZX, .param .u64 pW, .param .u64 pB, .param .u64 pS, .param .u64 pO, .param .u32 pN, .param .u32 pC, .param .u32 pLd, .param .u32 pOff)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<20>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pZX]; ld.param.u64 %rd2,[pW]; ld.param.u64 %rd3,[pB]; ld.param.u64 %rd4,[pS]; ld.param.u64 %rd5,[pO];\n"
         " ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pC]; ld.param.u32 %r3,[pLd]; ld.param.u32 %r4,[pOff];\n"
         " mov.u32 %r5,%ctaid.x; mov.u32 %r6,%tid.x; mad.lo.u32 %r5,%r5,256,%r6; setp.ge.u32 %p1,%r5,%r2; @%p1 bra DONE;\n"
         " mul.wide.u32 %rd6,%r5,16; add.u64 %rd6,%rd2,%rd6; ld.global.v4.f32 {%f1,%f2,%f3,%f4},[%rd6];\n"
         " mul.wide.u32 %rd7,%r5,4; add.u64 %rd8,%rd3,%rd7; ld.global.f32 %f5,[%rd8];\n"
         " add.u64 %rd9,%rd4,%rd7; mul.wide.u32 %rd10,%r2,4; ld.global.f32 %f6,[%rd9]; add.u64 %rd11,%rd9,%rd10; ld.global.f32 %f7,[%rd11]; add.u64 %rd12,%rd11,%rd10; ld.global.f32 %f8,[%rd12];\n"
         " mov.u32 %r7,0;\n"
         "LT:\n setp.ge.u32 %p2,%r7,%r1; @%p2 bra ET;\n"
         " mad.lo.u32 %r8,%r7,%r3,%r4; add.u32 %r8,%r8,%r5; mul.wide.u32 %rd13,%r8,4; add.u64 %rd13,%rd1,%rd13; ld.global.f32 %f9,[%rd13];\n"
         " fma.rn.f32 %f10,%f1,%f6,%f5; fma.rn.f32 %f10,%f2,%f7,%f10; fma.rn.f32 %f10,%f3,%f8,%f10; fma.rn.f32 %f10,%f4,%f9,%f10;\n"
         " mov.f32 %f6,%f7; mov.f32 %f7,%f8; mov.f32 %f8,%f9;\n"
         " mul.f32 %f11,%f10,0fBFB8AA3B; ex2.approx.f32 %f11,%f11; add.f32 %f11,%f11,0f3F800000; div.approx.f32 %f10,%f10,%f11;\n"
         " mad.lo.u32 %r9,%r7,%r2,%r5; mul.wide.u32 %rd14,%r9,4; add.u64 %rd14,%rd5,%rd14; st.global.f32 [%rd14],%f10;\n"
         " add.u32 %r7,%r7,1; bra LT;\nET:\n"
         " st.global.f32 [%rd9],%f6; st.global.f32 [%rd11],%f7; st.global.f32 [%rd12],%f8;\n"
         "DONE:\n ret;\n}\n";
    s += gen_ssm_scan();
    // k_ssm_scan1 (decode, one token): block = 256 / pDs state rows (row = head * hdim + p) x pDs state entries, one entry per thread, in a
    // register for all N tokens: S = exp(dt A) S + dt x B; y = sum(S C) + D x. dt = softplus(raw + bias).
    // X: conv output [N][pC] (x at 0, B at pBo + group * pDs, C at pCo + group * pDs); dt raw at ZX[t * pLd + pDto + h].
    s += ".visible .entry k_ssm_scan1(.param .u64 pX, .param .u64 pZX, .param .u64 pDtb, .param .u64 pA, .param .u64 pD, .param .u64 pS, .param .u64 pY,"
         " .param .u32 pN, .param .u32 pC, .param .u32 pLd, .param .u32 pDto, .param .u32 pDi, .param .u32 pDs, .param .u32 pHdim, .param .u32 pHpg, .param .u32 pBo, .param .u32 pCo)\n{\n"
         " .reg .pred %p<6>; .reg .b32 %r<40>; .reg .f32 %f<24>; .reg .b64 %rd<24>; .shared .align 4 .f32 red[8];\n"
         " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pZX]; ld.param.u64 %rd3,[pDtb]; ld.param.u64 %rd4,[pA]; ld.param.u64 %rd5,[pD]; ld.param.u64 %rd6,[pS]; ld.param.u64 %rd7,[pY];\n"
         " ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pC]; ld.param.u32 %r3,[pLd]; ld.param.u32 %r4,[pDto]; ld.param.u32 %r5,[pDi]; ld.param.u32 %r6,[pDs]; ld.param.u32 %r7,[pHdim]; ld.param.u32 %r8,[pHpg];\n"
         " ld.param.u32 %r30,[pBo]; ld.param.u32 %r31,[pCo];\n"
         " mov.u32 %r9,%tid.x; rem.u32 %r10,%r9,%r6; div.u32 %r11,%r9,%r6;\n"                 // n, row in block
         " div.u32 %r12,256,%r6; mov.u32 %r13,%ctaid.x; mad.lo.u32 %r14,%r13,%r12,%r11;\n"    // row
         " div.u32 %r15,%r14,%r7; div.u32 %r16,%r15,%r8; mul.lo.u32 %r16,%r16,%r6; add.u32 %r16,%r16,%r10;\n"   // h; group * ds + n
         " add.u32 %r32,%r30,%r16; add.u32 %r33,%r31,%r16;\n"                                   // B, C columns
         " mad.lo.u32 %r17,%r14,%r6,%r10; mul.wide.u32 %rd8,%r17,4; add.u64 %rd8,%rd6,%rd8; ld.global.f32 %f1,[%rd8];\n"   // state
         " mul.wide.u32 %rd9,%r15,4; add.u64 %rd10,%rd3,%rd9; ld.global.f32 %f2,[%rd10]; add.u64 %rd11,%rd4,%rd9; ld.global.f32 %f3,[%rd11]; add.u64 %rd12,%rd5,%rd9; ld.global.f32 %f4,[%rd12];\n"   // dtb, A, D
         " mul.f32 %f3,%f3,0f3FB8AA3B;\n"                                                       // A * log2(e)
         " and.b32 %r18,%r9,31; setp.eq.u32 %p2,%r18,0; shr.u32 %r19,%r10,5; shr.u32 %r20,%r6,5; mad.lo.u32 %r21,%r11,%r20,%r19; mov.u32 %r22,red; shl.b32 %r21,%r21,2; add.u32 %r21,%r22,%r21;\n"
         " mul.lo.u32 %r23,%r11,%r20; shl.b32 %r23,%r23,2; add.u32 %r23,%r22,%r23; setp.eq.u32 %p3,%r10,0;\n"
         " mov.u32 %r24,0;\n"
         "LT:\n setp.ge.u32 %p1,%r24,%r1; @%p1 bra ET;\n"
         " mad.lo.u32 %r25,%r24,%r3,%r4; add.u32 %r25,%r25,%r15; mul.wide.u32 %rd13,%r25,4; add.u64 %rd13,%rd2,%rd13; ld.global.f32 %f5,[%rd13]; add.f32 %f5,%f5,%f2;\n"
         " mul.f32 %f6,%f5,0f3FB8AA3B; ex2.approx.f32 %f6,%f6; add.f32 %f6,%f6,0f3F800000; lg2.approx.f32 %f6,%f6; mul.f32 %f6,%f6,0f3F317218; setp.gt.f32 %p4,%f5,0f41A00000; selp.f32 %f5,%f5,%f6,%p4;\n"   // softplus
         " mul.f32 %f7,%f5,%f3; ex2.approx.f32 %f7,%f7;\n"                                      // dA
         " mul.lo.u32 %r26,%r24,%r2; add.u32 %r27,%r26,%r14; mul.wide.u32 %rd14,%r27,4; add.u64 %rd14,%rd1,%rd14; ld.global.f32 %f8,[%rd14];\n"   // x
         " add.u32 %r28,%r26,%r32; mul.wide.u32 %rd15,%r28,4; add.u64 %rd15,%rd1,%rd15; ld.global.f32 %f9,[%rd15];\n"   // B
         " add.u32 %r29,%r26,%r33; mul.wide.u32 %rd16,%r29,4; add.u64 %rd16,%rd1,%rd16; ld.global.f32 %f10,[%rd16];\n" // C
         " mul.f32 %f11,%f5,%f8; mul.f32 %f11,%f11,%f9; fma.rn.f32 %f1,%f1,%f7,%f11; mul.f32 %f12,%f1,%f10;\n"
         " shfl.sync.bfly.b32 %f13,%f12,16,31,0xffffffff; add.f32 %f12,%f12,%f13; shfl.sync.bfly.b32 %f13,%f12,8,31,0xffffffff; add.f32 %f12,%f12,%f13;\n"
         " shfl.sync.bfly.b32 %f13,%f12,4,31,0xffffffff; add.f32 %f12,%f12,%f13; shfl.sync.bfly.b32 %f13,%f12,2,31,0xffffffff; add.f32 %f12,%f12,%f13;\n"
         " shfl.sync.bfly.b32 %f13,%f12,1,31,0xffffffff; add.f32 %f12,%f12,%f13;\n"
         " @%p2 st.shared.f32 [%r21],%f12; bar.sync 0;\n"
         " @!%p3 bra NW; mov.f32 %f14,0f00000000; mov.u32 %r34,0;\n"
         "LS:\n setp.ge.u32 %p5,%r34,%r20; @%p5 bra ES; shl.b32 %r35,%r34,2; add.u32 %r35,%r23,%r35; ld.shared.f32 %f15,[%r35]; add.f32 %f14,%f14,%f15; add.u32 %r34,%r34,1; bra LS;\nES:\n"
         " fma.rn.f32 %f14,%f4,%f8,%f14; mad.lo.u32 %r36,%r24,%r5,%r14; mul.wide.u32 %rd17,%r36,4; add.u64 %rd17,%rd7,%rd17; st.global.f32 [%rd17],%f14;\n"
         "NW:\n bar.sync 0; add.u32 %r24,%r24,1; bra LT;\nET:\n"
         " st.global.f32 [%rd8],%f1;\n ret;\n}\n";
    // k_ssm_gnorm: grid (groups, N), 256 threads: y *= silu(z), then RMS over the group * weight.
    s += ".visible .entry k_ssm_gnorm(.param .u64 pY, .param .u64 pZX, .param .u64 pW, .param .u32 pDi, .param .u32 pLd, .param .u32 pGs, .param .f32 pE)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<24>; .reg .f32 %f<16>; .reg .b64 %rd<16>; .shared .align 4 .f32 red[8];\n"
         " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pZX]; ld.param.u64 %rd3,[pW]; ld.param.u32 %r1,[pDi]; ld.param.u32 %r2,[pLd]; ld.param.u32 %r3,[pGs]; ld.param.f32 %f1,[pE];\n"
         " mov.u32 %r4,%ctaid.x; mov.u32 %r5,%ctaid.y; mov.u32 %r6,%tid.x; mul.lo.u32 %r7,%r4,%r3;\n"   // group start
         " mul.lo.u32 %r8,%r5,%r1; mul.lo.u32 %r9,%r5,%r2; mov.f32 %f2,0f00000000; mov.u32 %r10,%r6;\n"
         "L1:\n setp.ge.u32 %p1,%r10,%r3; @%p1 bra E1;\n"
         " add.u32 %r11,%r7,%r10; add.u32 %r12,%r8,%r11; mul.wide.u32 %rd4,%r12,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f3,[%rd4];\n"
         " add.u32 %r13,%r9,%r11; mul.wide.u32 %rd5,%r13,4; add.u64 %rd5,%rd2,%rd5; ld.global.f32 %f4,[%rd5];\n"
         " mul.f32 %f5,%f4,0fBFB8AA3B; ex2.approx.f32 %f5,%f5; add.f32 %f5,%f5,0f3F800000; div.approx.f32 %f4,%f4,%f5; mul.f32 %f3,%f3,%f4; st.global.f32 [%rd4],%f3;\n"
         " fma.rn.f32 %f2,%f3,%f3,%f2; add.u32 %r10,%r10,256; bra L1;\nE1:\n"
         " shfl.sync.bfly.b32 %f6,%f2,16,31,0xffffffff; add.f32 %f2,%f2,%f6; shfl.sync.bfly.b32 %f6,%f2,8,31,0xffffffff; add.f32 %f2,%f2,%f6;\n"
         " shfl.sync.bfly.b32 %f6,%f2,4,31,0xffffffff; add.f32 %f2,%f2,%f6; shfl.sync.bfly.b32 %f6,%f2,2,31,0xffffffff; add.f32 %f2,%f2,%f6; shfl.sync.bfly.b32 %f6,%f2,1,31,0xffffffff; add.f32 %f2,%f2,%f6;\n"
         " and.b32 %r14,%r6,31; setp.eq.u32 %p2,%r14,0; shr.u32 %r15,%r6,5; mov.u32 %r16,red; shl.b32 %r17,%r15,2; add.u32 %r17,%r16,%r17; @%p2 st.shared.f32 [%r17],%f2; bar.sync 0;\n"
         " ld.shared.f32 %f2,[%r16]; ld.shared.f32 %f7,[%r16+4]; add.f32 %f2,%f2,%f7; ld.shared.f32 %f7,[%r16+8]; add.f32 %f2,%f2,%f7; ld.shared.f32 %f7,[%r16+12]; add.f32 %f2,%f2,%f7;\n"
         " ld.shared.f32 %f7,[%r16+16]; add.f32 %f2,%f2,%f7; ld.shared.f32 %f7,[%r16+20]; add.f32 %f2,%f2,%f7; ld.shared.f32 %f7,[%r16+24]; add.f32 %f2,%f2,%f7; ld.shared.f32 %f7,[%r16+28]; add.f32 %f2,%f2,%f7;\n"
         " cvt.rn.f32.u32 %f8,%r3; div.rn.f32 %f2,%f2,%f8; add.f32 %f2,%f2,%f1; rsqrt.approx.f32 %f2,%f2; mov.u32 %r10,%r6;\n"
         "L2:\n setp.ge.u32 %p1,%r10,%r3; @%p1 bra E2;\n"
         " add.u32 %r11,%r7,%r10; add.u32 %r12,%r8,%r11; mul.wide.u32 %rd4,%r12,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f3,[%rd4];\n"
         " mul.wide.u32 %rd6,%r11,4; add.u64 %rd6,%rd3,%rd6; ld.global.f32 %f9,[%rd6]; mul.f32 %f3,%f3,%f2; mul.f32 %f3,%f3,%f9; st.global.f32 [%rd4],%f3;\n"
         " add.u32 %r10,%r10,256; bra L2;\nE2:\n ret;\n}\n";
    // DeepSeek MLA: padded heads from the projections. i over (token, head, c < pHd):
    // Q = [q_rope | q_nope | 0], K = [shared k_rope | k_nope | 0], V = [v | 0].
    s += ".visible .entry k_mla(.param .u64 pQr, .param .u64 pKa, .param .u64 pKV, .param .u64 pQ, .param .u64 pK, .param .u64 pV, .param .u32 pNh, .param .u32 pHd,"
         " .param .u32 pNope, .param .u32 pNr, .param .u32 pVd, .param .u32 pRank, .param .u32 pN)\n{\n"
         " .reg .pred %p<8>; .reg .b32 %r<32>; .reg .f32 %f<8>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pQr]; ld.param.u64 %rd2,[pKa]; ld.param.u64 %rd3,[pKV]; ld.param.u64 %rd4,[pQ]; ld.param.u64 %rd5,[pK]; ld.param.u64 %rd6,[pV];\n"
         " ld.param.u32 %r1,[pN]; ld.param.u32 %r2,[pNh]; ld.param.u32 %r3,[pHd]; ld.param.u32 %r4,[pNope]; ld.param.u32 %r5,[pNr]; ld.param.u32 %r6,[pVd]; ld.param.u32 %r7,[pRank];\n"
         " mov.u32 %r8,%ctaid.x; mov.u32 %r9,%ntid.x; mov.u32 %r10,%tid.x; mad.lo.u32 %r11,%r8,%r9,%r10; setp.ge.u32 %p1,%r11,%r1; @%p1 bra DONE;\n"
         " rem.u32 %r12,%r11,%r3; div.u32 %r13,%r11,%r3; div.u32 %r14,%r13,%r2;\n"                     // c, token * nh + head, token
         " add.u32 %r15,%r4,%r5; add.u32 %r16,%r4,%r6; add.u32 %r17,%r7,%r5;\n"                       // qk, nope + vd, rank + nr
         " setp.lt.u32 %p2,%r12,%r5; setp.lt.u32 %p3,%r12,%r15; setp.ge.u32 %p5,%r12,%r5; and.pred %p4,%p3,%p5; setp.lt.u32 %p6,%r12,%r6;\n"
         " sub.u32 %r21,%r12,%r5;\n"                                                                   // c - nr
         " mul.lo.u32 %r19,%r13,%r15; add.u32 %r20,%r19,%r4; add.u32 %r20,%r20,%r12; add.u32 %r22,%r19,%r21; selp.b32 %r22,%r20,%r22,%p2;\n"
         " mov.f32 %f1,0f00000000; @%p3 mul.wide.u32 %rd7,%r22,4; @%p3 add.u64 %rd7,%rd1,%rd7; @%p3 ld.global.f32 %f1,[%rd7];\n"
         " mad.lo.u32 %r23,%r14,%r17,%r7; add.u32 %r23,%r23,%r12; mul.lo.u32 %r24,%r13,%r16; add.u32 %r25,%r24,%r21;\n"
         " mov.f32 %f2,0f00000000; @%p2 mul.wide.u32 %rd8,%r23,4; @%p2 add.u64 %rd8,%rd2,%rd8; @%p2 ld.global.f32 %f2,[%rd8];\n"
         " @%p4 mul.wide.u32 %rd9,%r25,4; @%p4 add.u64 %rd9,%rd3,%rd9; @%p4 ld.global.f32 %f2,[%rd9];\n"
         " add.u32 %r27,%r24,%r4; add.u32 %r27,%r27,%r12; mov.f32 %f3,0f00000000; @%p6 mul.wide.u32 %rd10,%r27,4; @%p6 add.u64 %rd10,%rd3,%rd10; @%p6 ld.global.f32 %f3,[%rd10];\n"
         " mul.wide.u32 %rd11,%r11,4; add.u64 %rd12,%rd4,%rd11; st.global.f32 [%rd12],%f1; add.u64 %rd13,%rd5,%rd11; st.global.f32 [%rd13],%f2; add.u64 %rd14,%rd6,%rd11; st.global.f32 [%rd14],%f3;\n"
         "DONE:\n ret;\n}\n";
    // Qwen2-MoE shared expert: grid = tokens, 256 threads: X += pS * sigmoid(XN . W) * T, per token.
    s += ".visible .entry k_gadd(.param .u64 pX, .param .u64 pT, .param .u64 pXN, .param .u64 pW, .param .u32 pD, .param .f32 pS)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<16>; .reg .f32 %f<16>; .reg .b64 %rd<16>;\n";
    s += kRedPrologue;
    s += " ld.param.u64 %rd1,[pX]; ld.param.u64 %rd2,[pT]; ld.param.u64 %rd3,[pXN]; ld.param.u64 %rd4,[pW]; ld.param.u32 %r1,[pD]; ld.param.f32 %f1,[pS];\n"
         " mov.u32 %r2,%ctaid.x; mul.lo.u32 %r3,%r2,%r1; mov.u32 %r4,%tid.x; mov.f32 %f2,0f00000000; mov.u32 %r5,%r4;\n"
         "L1:\n setp.ge.u32 %p1,%r5,%r1; @%p1 bra E1; add.u32 %r6,%r3,%r5; mul.wide.u32 %rd5,%r6,4; add.u64 %rd6,%rd3,%rd5; ld.global.f32 %f3,[%rd6];\n"
         " mul.wide.u32 %rd7,%r5,4; add.u64 %rd7,%rd4,%rd7; ld.global.f32 %f4,[%rd7]; fma.rn.f32 %f2,%f3,%f4,%f2; add.u32 %r5,%r5,256; bra L1;\nE1:\n";
    s += bred("%f2", "add.f32");
    s += " mul.f32 %f5,%f2,0fBFB8AA3B; ex2.approx.f32 %f5,%f5; add.f32 %f5,%f5,0f3F800000; div.approx.f32 %f6,%f1,%f5; mov.u32 %r5,%r4;\n"
         "L2:\n setp.ge.u32 %p1,%r5,%r1; @%p1 bra E2; add.u32 %r6,%r3,%r5; mul.wide.u32 %rd5,%r6,4; add.u64 %rd8,%rd1,%rd5; add.u64 %rd9,%rd2,%rd5;\n"
         " ld.global.f32 %f7,[%rd8]; ld.global.f32 %f8,[%rd9]; fma.rn.f32 %f7,%f6,%f8,%f7; st.global.f32 [%rd8],%f7; add.u32 %r5,%r5,256; bra L2;\nE2:\n ret;\n}\n";
    // y[r*W+c] *= sigmoid(g[r*GS+c]): Qwen3.5 attention output gate
    s += ew("k_sigmul", ".param .u64 pY, .param .u64 pG, .param .u32 pW, .param .u32 pGS",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pG]; ld.param.u32 %r7,[pW]; ld.param.u32 %r8,[pGS];\n",
            " div.u32 %r9,%r5,%r7; mul.lo.u32 %r10,%r9,%r7; sub.u32 %r10,%r5,%r10; mad.lo.u32 %r11,%r9,%r8,%r10;\n"
            " mul.wide.u32 %rd3,%r11,4; add.u64 %rd3,%rd2,%rd3; ld.global.f32 %f2,[%rd3]; mul.f32 %f2,%f2,0fBFB8AA3B; ex2.approx.f32 %f2,%f2; add.f32 %f2,%f2,0f3F800000;\n"
            " mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f3,[%rd4]; div.approx.f32 %f3,%f3,%f2; st.global.f32 [%rd4],%f3;\n");
    // dst[r] = the expert of grouped row r: the last e with off[e] <= r (off: E + 1 row offsets)
    s += ew("k_rowexp", ".param .u64 pD, .param .u64 pOff, .param .u32 pE",
            " ld.param.u64 %rd1,[pD]; ld.param.u64 %rd2,[pOff]; ld.param.u32 %r7,[pE];\n",
            " mov.u32 %r8,0; mov.u32 %r9,1;\n"
            "RE:\n setp.ge.u32 %p2,%r9,%r7; @%p2 bra RD;\n"
            " mul.wide.u32 %rd3,%r9,4; add.u64 %rd3,%rd2,%rd3; ld.global.u32 %r10,[%rd3]; setp.le.u32 %p3,%r10,%r5; @%p3 mov.u32 %r8,%r9; add.u32 %r9,%r9,1; bra RE;\n"
            "RD:\n mul.wide.u32 %rd4,%r5,4; add.u64 %rd4,%rd1,%rd4; st.global.u32 [%rd4],%r8;\n");
    // dst[r*D+c] = src[idx[r]*D+c]
    s += ew("k_gather", ".param .u64 pS, .param .u64 pI, .param .u64 pD, .param .u32 pDim",
            " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pI]; ld.param.u64 %rd6,[pD]; ld.param.u32 %r7,[pDim];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd3,%rd2,%rd3; ld.global.u32 %r10,[%rd3];\n"
            " mad.lo.u32 %r11,%r10,%r7,%r9; mul.wide.u32 %rd4,%r11,4; add.u64 %rd4,%rd1,%rd4; ld.global.f32 %f2,[%rd4];\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd6,%rd5; st.global.f32 [%rd5],%f2;\n");
    // dst[idx[r]*D+c] += w[r] * src[r*D+c], atomically (one launch may hold several rows per token)
    s += ew("k_scatter", ".param .u64 pY, .param .u64 pS, .param .u64 pI, .param .u64 pW, .param .u32 pDim",
            " ld.param.u64 %rd1,[pY]; ld.param.u64 %rd2,[pS]; ld.param.u64 %rd6,[pI]; ld.param.u64 %rd7,[pW]; ld.param.u32 %r7,[pDim];\n",
            " div.u32 %r8,%r5,%r7; mul.lo.u32 %r9,%r8,%r7; sub.u32 %r9,%r5,%r9;\n"
            " mul.wide.u32 %rd3,%r8,4; add.u64 %rd4,%rd6,%rd3; ld.global.u32 %r10,[%rd4]; add.u64 %rd8,%rd7,%rd3; ld.global.f32 %f3,[%rd8];\n"
            " mul.wide.u32 %rd5,%r5,4; add.u64 %rd5,%rd2,%rd5; ld.global.f32 %f2,[%rd5];\n"
            " mad.lo.u32 %r11,%r10,%r7,%r9; mul.wide.u32 %rd9,%r11,4; add.u64 %rd9,%rd1,%rd9;\n"
            " mul.f32 %f4,%f2,%f3; red.global.add.f32 [%rd9],%f4;\n");

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
    // int8 KV store: grid = n_tok * n_kv, 32 threads. dst[(pos+t)*kvd + h*hd + i] = round(x / s), S[(pos+t)*nkv + h] = max|x| / 127
    s += ".visible .entry k_kvstore_q8(.param .u64 pS, .param .u64 pD, .param .u64 pSc, .param .u64 pPos, .param .u32 pHD, .param .u32 pNKV)\n{\n"
         " .reg .pred %p<4>; .reg .b32 %r<24>; .reg .f32 %f<24>; .reg .b64 %rd<16>;\n"
         " ld.param.u64 %rd1,[pS]; ld.param.u64 %rd2,[pD]; ld.param.u64 %rd3,[pSc]; ld.param.u64 %rd4,[pPos]; ld.param.u32 %r1,[pHD]; ld.param.u32 %r2,[pNKV];\n"
         " ld.global.u32 %r3,[%rd4]; mov.u32 %r4,%ctaid.x; mov.u32 %r5,%tid.x;\n"
         " div.u32 %r6,%r4,%r2; rem.u32 %r7,%r4,%r2; mul.lo.u32 %r8,%r2,%r1;\n"                 // token, head, kvd
         " mul.lo.u32 %r9,%r6,%r8; mad.lo.u32 %r9,%r7,%r1,%r9;\n"                                // src offset (token row)
         " add.u32 %r10,%r3,%r6; mul.lo.u32 %r11,%r10,%r8; mad.lo.u32 %r11,%r7,%r1,%r11;\n"     // dst offset
         " mov.f32 %f1,0f00000000;\n";
    for (int t = 0; t < 8; t++) {
        char b[256];
        snprintf(b, sizeof b, " add.u32 %%r12,%%r5,%d; setp.lt.u32 %%p1,%%r12,%%r1; add.u32 %%r13,%%r9,%%r12; mul.wide.u32 %%rd5,%%r13,4; add.u64 %%rd5,%%rd1,%%rd5;"
                 " mov.f32 %%f%d,0f00000000; @%%p1 ld.global.f32 %%f%d,[%%rd5]; abs.f32 %%f20,%%f%d; max.f32 %%f1,%%f1,%%f20;\n", 32 * t, 2 + t, 2 + t, 2 + t);
        s += b;
    }
    for (int o = 16; o >= 1; o /= 2) { char b[128]; snprintf(b, sizeof b, " shfl.sync.bfly.b32 %%f20,%%f1,%d,31,0xffffffff; max.f32 %%f1,%%f1,%%f20;\n", o); s += b; }
    s += " mul.f32 %f12,%f1,0f3C010204; setp.gt.f32 %p2,%f1,0f00000000; div.rn.f32 %f13,0f42FE0000,%f1; selp.f32 %f13,%f13,0f00000000,%p2;\n";
    for (int t = 0; t < 8; t++) {
        char b[256];
        snprintf(b, sizeof b, " add.u32 %%r12,%%r5,%d; setp.lt.u32 %%p1,%%r12,%%r1; add.u32 %%r14,%%r11,%%r12; cvt.u64.u32 %%rd6,%%r14; add.u64 %%rd6,%%rd2,%%rd6;"
                 " mul.f32 %%f21,%%f%d,%%f13; cvt.rni.s32.f32 %%r15,%%f21; @%%p1 st.global.s8 [%%rd6],%%r15;\n", 32 * t, 2 + t);
        s += b;
    }
    s += " setp.ne.u32 %p3,%r5,0; @%p3 bra KQD; mad.lo.u32 %r16,%r10,%r2,%r7; mul.wide.u32 %rd7,%r16,4; add.u64 %rd7,%rd3,%rd7; st.global.f32 [%rd7],%f12;\n"
         "KQD:\n ret;\n}\n";
    s += gen_attn();
    return s;
}

// Per-model decode attention module (see gen_attn_gqa).
std::string gqa_attn_ptx(int G, int NT, bool q8) { return ".version 6.0\n.target sm_61\n.address_size 64\n" + gen_attn_gqa(G, NT, q8); }
