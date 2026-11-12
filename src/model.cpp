#include "model.h"
#include "gx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

CudaTier& Model::cuda() {
    static CudaTier c;
    static std::string info;
    static bool inited = (c.init(info) || (fprintf(stderr, "[gpu] %s\n", info.c_str()), true), true);
    (void)inited;
    return c;
}

static std::string gpu_init_info() {
    Model::cuda();
    return Model::cuda().ok() ? "GPU: " + Model::cuda().name : "CPU only (no usable NVIDIA GPU)";
}

std::vector<float> Model::vec(const std::string& name) {
    const GTensor* t = g_.get(name);
    if (!t) return {};
    int64_t n = t->ne[0] * t->ne[1];
    std::vector<float> v(n);
    dequant_row(t->type, t->data, v.data(), n);
    return v;
}

bool Model::bind(Lin& l, const std::string& name, std::string& err, bool required) {
    const GTensor* t = g_.get(name);
    if (!t) { if (required) err = "missing tensor " + name; return !required; }
    if (!gtype_supported(t->type)) { err = std::string("unsupported quant type ") + gtype_name(t->type) + " in " + name; return false; }
    l.t = t; l.cols = (int)t->ne[0]; l.rows = (int)(t->ne[1] * t->ne[2]);   // 3-D expert stacks: rows of all experts
    return true;
}

static size_t gpu_cost(const Lin& l) {
    if (!l.t) return 0;
    if (CudaTier::can_load(l.t->type)) return CudaTier::gpu_bytes(l.t->type, l.rows, l.cols);
    return (size_t)l.rows * (l.cols / 32) * 36 + 256;  // CPU-repacked Q8 layout
}

int iq_index(int gtype);
static const int kExpChunk = 8;   // IQ experts converted to Q8_0 this many at a time for prompts
bool Model::load(const std::wstring& path, const LoadOpts& o, std::string& err) {
    unload();
    double t0 = now_sec();
    if (!g_.open(path, err)) return false;
    path_utf8 = w_to_utf8(path);
    hp = HParams();
    moe_ = false; kvq8_ = false;
    hp.arch = g_.gets("general.architecture");
    hp.name = g_.gets("general.name", "model");
    const std::string& a = hp.arch;
    static const char* ok_arch[] = {"llama", "qwen2", "qwen2vl", "qwen3", "phi3", "granite", "mistral", "gemma3", "granitemoe", "qwen3moe", "gpt-oss", "granitehybrid", "ernie4_5", "smollm3", "olmo2", "olmoe", "gemma2", "lfm2", "qwen35", "qwen2moe", "deepseek2"};
    if (std::find_if(std::begin(ok_arch), std::end(ok_arch), [&](const char* s) { return a == s; }) == std::end(ok_arch)) {
        err = "architecture '" + a + "' is not supported yet (supported: llama, mistral, mixtral, qwen2, qwen2.5-vl text, qwen3, qwen3-moe, phi3/phi4, granite, granite-moe, gemma3, gpt-oss)";
        g_.close(); return false;
    }
    std::string P = a + ".";
    hp.n_embd = (int)g_.geti(P + "embedding_length", 0);
    hp.n_layer = (int)g_.geti(P + "block_count", 0) - (int)g_.geti(P + "nextn_predict_layers", 0);   // multi-token-prediction layers are not run
    hp.n_head = (int)g_.geti(P + "attention.head_count", 0);
    {
        const GValue* kvv = g_.find(P + "attention.head_count_kv");
        hp.n_kv = kvv ? (kvv->arr_n.empty() ? (int)kvv->i : (int)*std::max_element(kvv->arr_n.begin(), kvv->arr_n.end())) : hp.n_head;   // per-layer list: the attention layers' count
    }
    hp.n_ff = (int)g_.geti(P + "feed_forward_length", 0);
    hp.eps = (float)g_.getf(P + "attention.layer_norm_rms_epsilon", 1e-5);
    hp.rope_base = (float)g_.getf(P + "rope.freq_base", 10000.0);
    hp.hd = (int)g_.geti(P + "attention.key_length", hp.n_head ? hp.n_embd / hp.n_head : 0);
    hp.n_rot = (int)g_.geti(P + "rope.dimension_count", hp.hd);
    hp.ctx_train = (int)g_.geti(P + "context_length", 4096);
    hp.neox = a == "qwen2" || a == "qwen2vl" || a == "qwen3" || a == "phi3" || a == "qwen3moe" || a == "gemma3" || a == "gpt-oss" || a == "gemma2" || a == "olmo2" || a == "olmoe" || a == "lfm2" || a == "qwen35" || a == "qwen2moe";
    hp.n_expert = (int)g_.geti(P + "expert_count", 0);
    hp.n_expert_used = (int)g_.geti(P + "expert_used_count", 0);
    { const GValue* en = g_.find(P + "expert_weights_norm"); hp.expert_norm = en ? en->i != 0 : true; }
    if (a == "gemma3") {
        // Gemma 3: scaled embeddings, GELU, sandwich norms, 5 sliding-window layers per global one
        hp.gelu = true; hp.emb_scale = std::sqrt((float)hp.n_embd);
        hp.swa_window = (int)g_.geti(P + "attention.sliding_window", 0); if (getenv("ITHACA_NO_SWA")) hp.swa_window = 0;
        hp.rope_base_swa = (float)g_.getf(P + "rope.local.freq_base", 10000.0);
        if (hp.n_layer == 62) hp.attn_scale = 1.f / std::sqrt((float)hp.n_embd / hp.n_head);   // 27B uses query_pre_attn_scalar
    }
    if (a == "gemma2") {   // Gemma 3's layout (minus q/k norms) with soft-capping; every other layer has a 4096 window
        hp.gelu = true; hp.emb_scale = std::sqrt((float)hp.n_embd);
        hp.swa_window = (int)g_.geti(P + "attention.sliding_window", 0); if (getenv("ITHACA_NO_SWA")) hp.swa_window = 0;
        hp.rope_base_swa = hp.rope_base;
        hp.attn_softcap = (float)g_.getf(P + "attn_logit_softcapping", 0.0); hp.final_softcap = (float)g_.getf(P + "final_logit_softcapping", 0.0);
    }
    if (a == "smollm3") hp.nope_step = 4;
    if (a == "deepseek2") {
        // MLA: q / k heads are [nope | rope] (key_length), values value_length. Ithaca stores both padded to one
        // 256-wide head laid out [rope | nope | 0], so the usual attention kernels and KV cache run it unchanged.
        hp.mla_rank = (int)g_.geti(P + "attention.kv_lora_rank", 0); hp.mla_qlora = (int)g_.geti(P + "attention.q_lora_rank", 0);
        hp.mla_v = (int)g_.geti(P + "attention.value_length", 0); hp.mla_nope = hp.hd - hp.n_rot;
        if (hp.mla_rank <= 0 || hp.mla_v <= 0 || hp.hd > 256 || hp.mla_v > 256) { err = "unexpected DeepSeek attention shapes"; g_.close(); return false; }
        hp.hd = 256; hp.n_kv = hp.n_head;
        hp.exp_scale = (float)g_.getf(P + "expert_weights_scale", 1.0);
        if (!g_.find(P + "expert_weights_norm")) hp.expert_norm = false;   // V2: top-k weights are not renormalized
        if (g_.geti(P + "expert_gating_func", 1) == 2) { err = "DeepSeek V3-style sigmoid routing isn't supported yet"; g_.close(); return false; }
    }
    if (a == "qwen35") {   // Gated DeltaNet + gated attention (3:1)
        hp.ssm_inner = (int)g_.geti(P + "ssm.inner_size", 0); hp.ssm_state = (int)g_.geti(P + "ssm.state_size", 0);
        hp.ssm_groups = (int)g_.geti(P + "ssm.group_count", 1); hp.ssm_heads = (int)g_.geti(P + "ssm.time_step_rank", 0); hp.ssm_conv = (int)g_.geti(P + "ssm.conv_kernel", 4);
    }
    if (a == "olmo2") hp.no_prenorm = true;
    if ((a == "olmoe" || a == "qwen2moe") && !g_.find(P + "expert_weights_norm")) hp.expert_norm = false;   // top-k weights are not renormalized   // top-k weights are not renormalized
    if (a == "gpt-oss") {   // every other layer (from the first) has a 128-token window; same RoPE on both kinds
        hp.oai = true;
        hp.swa_window = (int)g_.geti(P + "attention.sliding_window", 0); if (getenv("ITHACA_NO_SWA")) hp.swa_window = 0;
        hp.rope_base_swa = hp.rope_base;
    }
    if (a == "granitehybrid") {
        hp.nope = !(g_.find(P + "rope.scaling.finetuned") && g_.find(P + "rope.scaling.finetuned")->i);   // the converter stores "uses RoPE" here
        hp.ssm_inner = (int)g_.geti(P + "ssm.inner_size", 0); hp.ssm_state = (int)g_.geti(P + "ssm.state_size", 0);
        hp.ssm_groups = (int)g_.geti(P + "ssm.group_count", 1); hp.ssm_heads = (int)g_.geti(P + "ssm.time_step_rank", 0); hp.ssm_conv = (int)g_.geti(P + "ssm.conv_kernel", 4);
    }
    if (a == "granite" || a == "granitemoe" || a == "granitehybrid") {
        hp.emb_scale = (float)g_.getf(P + "embedding_scale", 1.0);
        hp.res_scale = (float)g_.getf(P + "residual_scale", 1.0);
        hp.attn_scale = (float)g_.getf(P + "attention.scale", 0.0);
        hp.logit_scale = (float)g_.getf(P + "logit_scale", 1.0);
    }
    if (!hp.n_embd || !hp.n_layer || !hp.n_head || !hp.hd) { err = "incomplete model metadata"; g_.close(); return false; }
    chat_template = g_.gets("tokenizer.chat_template");
    if (!tok.load(g_, err)) { g_.close(); return false; }

    embd_ = g_.get("token_embd.weight");
    if (!embd_ || !gtype_supported(embd_->type)) { err = "missing/unsupported token_embd.weight"; g_.close(); return false; }
    hp.n_vocab = (int)embd_->ne[1];
    if (!bind(out_, "output.weight", err, false) || (!out_.t && !bind(out_, "token_embd.weight", err))) { g_.close(); return false; }
    out_norm_ = vec("output_norm.weight");
    if (out_norm_.empty()) out_norm_ = vec("token_embd_norm.weight");   // LFM2 keeps its final norm under this name

    layers_ = std::vector<Layer>(hp.n_layer);
    int ff_max = 0;   // widest dense FFN (layers may differ: DeepSeek's dense first layer, shared experts)
    for (int i = 0; i < hp.n_layer; i++) {
        Layer& L = layers_[i];
        std::string b = "blk." + std::to_string(i) + ".";
        L.attn_norm = vec(b + "attn_norm.weight");
        L.ffn_norm = vec(b + "ffn_norm.weight");
        L.q_norm = vec(b + "attn_q_norm.weight");
        L.k_norm = vec(b + "attn_k_norm.weight");
        L.bq = vec(b + "attn_q.bias"); L.bk = vec(b + "attn_k.bias"); L.bv = vec(b + "attn_v.bias");
        L.post_attn_norm = vec(b + "post_attention_norm.weight"); L.post_ffn_norm = vec(b + "post_ffw_norm.weight");
        L.swa = hp.swa_window > 0 && (hp.oai || hp.arch == "gemma2" ? i % 2 == 0 : (i + 1) % 6 != 0);
        L.nope = hp.nope || (hp.nope_step > 0 && (i + 1) % hp.nope_step == 0);
        if (hp.oai) {   // gpt-oss names the FFN input norm post_attention_norm
            L.ffn_norm.swap(L.post_attn_norm);
            L.bo = vec(b + "attn_output.bias"); L.sinks = vec(b + "attn_sinks.weight"); L.b_inp = vec(b + "ffn_gate_inp.bias");
            L.b_gate = vec(b + "ffn_gate_exps.bias"); L.b_up = vec(b + "ffn_up_exps.bias"); L.b_down = vec(b + "ffn_down_exps.bias");
        }
        bool ok = true;
        if (hp.arch == "qwen35") L.ffn_norm.swap(L.post_attn_norm);   // post_attention_norm is the FFN input norm
        if (hp.arch == "qwen35" && g_.get(b + "ssm_a")) {   // Gated DeltaNet layer
            ok = bind(L.ssm_in, b + "attn_qkv.weight", err) && bind(L.ssm_out, b + "ssm_out.weight", err) && bind(L.dn_z, b + "attn_gate.weight", err) &&
                 bind(L.dn_a, b + "ssm_alpha.weight", err) && bind(L.dn_b, b + "ssm_beta.weight", err);
            L.conv_w = vec(b + "ssm_conv1d.weight"); L.ssm_a = vec(b + "ssm_a"); L.ssm_dtb = vec(b + "ssm_dt.bias"); L.ssm_norm = vec(b + "ssm_norm.weight");
            const size_t C = (size_t)2 * hp.ssm_groups * hp.ssm_state + hp.ssm_inner;
            if (L.conv_w.size() != C * hp.ssm_conv || L.ssm_a.size() != (size_t)hp.ssm_heads || hp.ssm_inner != hp.ssm_heads * hp.ssm_state) { err = "unexpected Gated DeltaNet shapes"; ok = false; }
            L.conv_b.assign(C, 0.f); L.conv_st.assign((size_t)(hp.ssm_conv - 1) * C, 0.f); L.ssm_st.assign((size_t)hp.ssm_heads * hp.ssm_state * hp.ssm_state, 0.f);
            L.dnet = true; recurrent_ = true;
        }
        else if (g_.get(b + "shortconv.in_proj.weight")) {   // LFM2 short-convolution layer instead of attention
            ok = bind(L.ssm_in, b + "shortconv.in_proj.weight", err) && bind(L.ssm_out, b + "shortconv.out_proj.weight", err);
            L.conv_w = vec(b + "shortconv.conv.weight"); L.sconv = true;
            if (L.conv_w.size() != (size_t)hp.n_embd * 3) { err = "unexpected short-conv shapes (kernel 3 expected)"; ok = false; }
            L.conv_st.assign((size_t)2 * hp.n_embd, 0.f);
            recurrent_ = true;
        }
        else if (g_.get(b + "ssm_in.weight")) {   // Mamba-2 layer instead of attention
            ok = bind(L.ssm_in, b + "ssm_in.weight", err) && bind(L.ssm_out, b + "ssm_out.weight", err);
            L.conv_w = vec(b + "ssm_conv1d.weight"); L.conv_b = vec(b + "ssm_conv1d.bias"); L.ssm_a = vec(b + "ssm_a"); L.ssm_d = vec(b + "ssm_d");
            L.ssm_dtb = vec(b + "ssm_dt.bias"); L.ssm_norm = vec(b + "ssm_norm.weight");
            const size_t xbc = hp.ssm_inner + 2 * hp.ssm_groups * hp.ssm_state;
            if (!hp.ssm_heads || L.conv_w.size() != xbc * hp.ssm_conv || L.ssm_a.size() != (size_t)hp.ssm_heads) { err = "unexpected Mamba-2 layer shapes"; ok = false; }
            L.conv_st.assign((hp.ssm_conv - 1) * xbc, 0.f); L.ssm_st.assign((size_t)hp.ssm_inner * hp.ssm_state, 0.f);
            recurrent_ = true;
        }
        else if (g_.get(b + "attn_kv_a_mqa.weight")) {   // DeepSeek V2 multi-head latent attention
            if (!g_.get(b + "attn_kv_b.weight")) { err = "this DeepSeek file splits attn_kv_b (newer converter); not supported yet"; ok = false; }
            ok = ok && bind(L.kva, b + "attn_kv_a_mqa.weight", err) && bind(L.kvb, b + "attn_kv_b.weight", err) &&
                 (g_.get(b + "attn_q_a.weight") ? bind(L.qa, b + "attn_q_a.weight", err) && bind(L.qb, b + "attn_q_b.weight", err) : bind(L.q, b + "attn_q.weight", err));
            L.kva_norm = vec(b + "attn_kv_a_norm.weight"); L.qa_norm = vec(b + "attn_q_a_norm.weight");
            const int nh = hp.n_head, qk = hp.mla_nope + hp.n_rot;
            if (ok && (L.kva.rows != hp.mla_rank + hp.n_rot || L.kvb.cols != hp.mla_rank || L.kvb.rows != nh * (hp.mla_nope + hp.mla_v) ||
                       (L.q.t ? L.q.rows : L.qb.rows) != nh * qk || (int)L.kva_norm.size() != hp.mla_rank || (L.qa.t && (int)L.qa_norm.size() != L.qa.rows))) {
                err = "unexpected DeepSeek attention shapes"; ok = false;
            }
        }
        else if (g_.get(b + "attn_qkv.weight")) ok &= bind(L.qkv, b + "attn_qkv.weight", err);
        else ok &= bind(L.q, b + "attn_q.weight", err) && bind(L.k, b + "attn_k.weight", err) && bind(L.v, b + "attn_v.weight", err);
        if (!L.ssm_in.t) ok = ok && bind(L.o, b + "attn_output.weight", err);
        L.qgate = L.q.t && L.q.rows == 2 * hp.n_head * hp.hd;
        if (ok && g_.get(b + "ffn_gate_exps.weight")) {
            if (hp.n_expert < 1 || hp.n_expert_used < 1) { err = "MoE model without expert counts"; ok = false; }
            ok = ok && bind(L.gate_inp, b + "ffn_gate_inp.weight", err) && bind(L.gate_exps, b + "ffn_gate_exps.weight", err) &&
                 bind(L.up_exps, b + "ffn_up_exps.weight", err) && bind(L.down_exps, b + "ffn_down_exps.weight", err);
            if (ok && g_.get(b + "ffn_gate_shexp.weight")) {   // shared experts: a dense FFN beside the routed ones (Qwen2-MoE gates it)
                ok = bind(L.gate, b + "ffn_gate_shexp.weight", err) && bind(L.up, b + "ffn_up_shexp.weight", err) && bind(L.down, b + "ffn_down_shexp.weight", err);
                L.shg = vec(b + "ffn_gate_inp_shexp.weight"); L.shexp = true;
                if (ok && !L.shg.empty() && (int)L.shg.size() != hp.n_embd) { err = "unexpected shared-expert gate"; ok = false; }
            }
        } else {
            ok = ok && bind(L.down, b + "ffn_down.weight", err);
            if (ok && g_.get(b + "ffn_gate.weight")) ok = bind(L.gate, b + "ffn_gate.weight", err) && bind(L.up, b + "ffn_up.weight", err);
            else if (ok) ok = bind(L.gateup, b + "ffn_up.weight", err);
        }
        if (hp.no_prenorm) { L.attn_norm.assign(hp.n_embd, 1.f); L.ffn_norm.assign(hp.n_embd, 1.f); }   // placeholders: skipped below
        if (!ok || L.attn_norm.empty() || L.ffn_norm.empty()) {
            if (err.empty()) err = "layer " + std::to_string(i) + " is incomplete (MoE / hybrid models aren't supported yet)";
            layers_.clear(); g_.close(); return false;
        }
        int lshard = -1;   // split GGUF: the range covers the file holding the layer's first matrix (a range can't span two mappings)
        for (Lin* l : L.lins()) {
            if (!l->t) continue;
            if (lshard < 0) lshard = l->t->shard;
            if (l->t->shard != lshard) continue;
            if (!L.lo || l->t->data < L.lo) L.lo = l->t->data;
            if (l->t->data + l->t->nbytes > L.hi) L.hi = l->t->data + l->t->nbytes;
        }
        if (L.gate_exps.t) { hp.n_ff_exp = L.gate_exps.rows / hp.n_expert; moe_ = true; if (L.gate.t) ff_max = std::max(ff_max, L.gate.rows); }
        else if (L.gateup.t) ff_max = std::max(ff_max, L.gateup.rows / 2);
        else ff_max = std::max(ff_max, L.gate.rows);
    }

    hp.n_ff = std::max(ff_max > 0 ? ff_max : hp.n_ff, hp.n_ff_exp);
    inv_freq_.resize(hp.n_rot / 2); inv_freq_swa_.resize(hp.n_rot / 2);
    std::vector<float> factors = vec("rope_freqs.weight");
    float rscale = g_.gets(P + "rope.scaling.type") == "linear" ? (float)g_.getf(P + "rope.scaling.factor", 1.0) : 1.f;
    if (rscale <= 0) rscale = 1.f;
    for (int i = 0; i < hp.n_rot / 2; i++) {
        inv_freq_[i] = std::pow(hp.rope_base, -2.0f * i / hp.n_rot) / rscale;   // global layers carry the context scaling
        if (i < (int)factors.size() && factors[i] > 0) inv_freq_[i] /= factors[i];
        inv_freq_swa_[i] = std::pow(hp.rope_base_swa, -2.0f * i / hp.n_rot);
    }
    if (g_.gets(P + "rope.scaling.type") == "yarn") {
        // YaRN: low frequencies are interpolated by the factor, high ones kept, a linear ramp between; cos and sin are
        // scaled by 1 + 0.1 ln(factor), which here goes into the attention scale (it multiplies both q and k).
        const double f = g_.getf(P + "rope.scaling.factor", 1.0), orig = g_.getf(P + "rope.scaling.original_context_length", 4096.0);
        const double bf = g_.getf(P + "rope.scaling.yarn_beta_fast", 32.0), bs = g_.getf(P + "rope.scaling.yarn_beta_slow", 1.0), dim = hp.n_rot;
        auto corr = [&](double b) { return dim * std::log(orig / (b * 2 * 3.14159265358979)) / (2 * std::log((double)hp.rope_base)); };
        double lo = std::max(0.0, std::floor(corr(bf))), hi = std::min(dim - 1, std::ceil(corr(bs))); if (hi <= lo) hi = lo + 0.001;
        for (int i = 0; i < hp.n_rot / 2; i++) {
            const double fr = std::pow((double)hp.rope_base, -2.0 * i / dim), ramp = std::min(1.0, std::max(0.0, (i - lo) / (hi - lo)));
            inv_freq_[i] = inv_freq_swa_[i] = (float)(fr / f * ramp + fr * (1 - ramp));
        }
        const double ms = f > 1 ? 1 + 0.1 * std::log(f) : 1;
        hp.attn_scale = (float)(ms * ms / std::sqrt((double)hp.hd));
    }
    if (hp.mla_rank) {   // llama.cpp: the score scale is mscale^2 / sqrt(qk head size), mscale = 1 + log_mul ln(factor)
        const double f = g_.gets(P + "rope.scaling.type") == "yarn" ? g_.getf(P + "rope.scaling.factor", 1.0) : 1.0;
        const double lm = g_.getf(P + "rope.scaling.yarn_log_multiplier", 0.0);
        const double ms = f > 1 ? 1 + lm * std::log(f) : 1;
        hp.attn_scale = (float)(ms * ms / std::sqrt((double)(hp.mla_nope + hp.n_rot)));
    }

    ctx = std::max(256, std::min(o.ctx, hp.ctx_train > 0 ? hp.ctx_train : o.ctx));
    size_t kvd = (size_t)hp.n_kv * hp.hd;
    kc_.assign((size_t)hp.n_layer * ctx * kvd, 0);
    vc_.assign((size_t)hp.n_layer * ctx * kvd, 0);
    n_past = 0;

    int nt = o.threads > 0 ? o.threads : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    pool_.start(nt);
    int maxc = std::max({hp.n_embd, hp.n_ff, hp.n_head * hp.hd, ctx}) + 64;
    scratch_.assign(nt, std::vector<float>(maxc));
    bufn_ = 0;

    // ---- Tier plan: VRAM first (output head, then layers from the bottom), then pinned RAM, then disk streaming.
    gpu_info = gpu_init_info();
    int64_t vram_budget = 0;
    size_t vfree = 0, vtotal = 0;
    if (o.vram_mb != 0 && cuda().ok() && cuda().mem_info(vfree, vtotal)) {
        int64_t reserve = 768ll << 20;  // leave headroom for other apps (Blender, the desktop)
        vram_budget = o.vram_mb < 0 ? (int64_t)vfree - reserve : std::min<int64_t>(o.vram_mb << 20, (int64_t)vfree - (256ll << 20));
        // With tensor-core flash attention, prompt batches never materialize a score matrix, so the
        // score buffer only needs one token's worth (decode) and batches can stay large.
        std::string ge;
        const bool flash = !getenv("ITHACA_NO_FLASH") && !getenv("ITHACA_NO_TC_PREFILL") && (gx().ok() || gx().init(ge)) && gx().tensor_cores() &&
                           (hp.hd == 64 || hp.hd == 80 || hp.hd == 96 || hp.hd == 128 || hp.hd == 256);
        if ((hp.oai || hp.attn_softcap > 0) && !flash) vram_budget = 0;   // sinks and soft-capping are only in the tensor-core kernels
        if (recurrent_ && (getenv("ITHACA_SSM_CPU") || (hp.ssm_heads && (hp.arch == "qwen35" ? hp.ssm_state != 128 || hp.ssm_inner / hp.ssm_heads != 128 : !ssm_gpu_ok())))) vram_budget = 0;   // state kernels need conv 4 and a state size dividing 256   // attention sinks are only in the tensor-core kernels: CPU elsewhere
        gpu_batch_ = flash ? 512 : (int)std::max<int64_t>(1, std::min<int64_t>(256, (128ll << 20) / ((int64_t)hp.n_head * ctx * 4)));
        sc_rows_ = flash ? 1 : gpu_batch_;
        {   // int8 KV needs the flash prompt kernels and the generated decode kernel (they read int8)
            const int G = hp.n_head / std::max(1, hp.n_kv), NT = (hp.hd + 31) / 32;
            kvq8_ = flash && G >= 1 && G <= 8 && NT <= 8 && G * NT * 32 <= 1024 && !getenv("ITHACA_KV16") && !getenv("ITHACA_NO_GQA");
        }
        int64_t act = (int64_t)gpu_batch_ * 4 * ((int64_t)hp.n_embd * 2 + hp.n_head * hp.hd * 2 + kvd * 2 + hp.n_ff * 3 +
                      std::max<int64_t>({hp.n_embd, hp.n_head * hp.hd + 2 * (int64_t)kvd, 2ll * hp.n_ff}) + hp.n_rot) +
                      (int64_t)sc_rows_ * hp.n_head * ctx * 4 + hp.n_vocab * 4ll;
        w16_plan_ = 0;
        for (auto& L : layers_)   // IQ / MXFP4 experts: prompts convert one stacked matrix at a time to Q8_0 (scratch)
            for (Lin* l : {&L.gate_exps, &L.up_exps, &L.down_exps})
                if (l->t && iq_index(l->t->type) >= 0) w16_plan_ = std::max<int64_t>(w16_plan_, (int64_t)l->rows / hp.n_expert * kExpChunk * (l->cols / 32) * 34 + 256);
        vram_budget -= act + w16_plan_ + (64ll << 20);
        if (o.vram_mb < 0) {   // a model that only misses by the headroom runs all on the GPU with a smaller one (a layer on the CPU halves the speed)
            int64_t need = gpu_cost(out_);
            for (auto& L : layers_) { for (Lin* l : L.lins()) need += gpu_cost(*l); need += (int64_t)ctx * kv_pos_bytes(); }
            if (need > vram_budget && need <= vram_budget + (512ll << 20)) vram_budget = need;
        }
    }
    MEMORYSTATUSEX ms{sizeof(ms)}; GlobalMemoryStatusEx(&ms);
    int64_t kv_bytes = (int64_t)(kc_.size() + vc_.size()) * 2;
    int64_t ram_budget = o.ram_mb < 0 ? (int64_t)ms.ullAvailPhys - (2560ll << 20) - kv_bytes : (o.ram_mb << 20);
    if (ram_budget < 0) ram_budget = 0;

    int64_t vused = 0, rused = 0;
    out_plan_ = TIER_DISK;
    if (vram_budget > 0 && (int64_t)gpu_cost(out_) <= vram_budget) { out_plan_ = TIER_GPU; vused += gpu_cost(out_); }
    bool gpu_open = vram_budget > 0;
    for (auto& L : layers_) {
        size_t c = 0;
        for (Lin* l : L.lins()) c += gpu_cost(*l);
        c += (size_t)ctx * kv_pos_bytes();  // K+V cache in VRAM
        if (gpu_open && vused + (int64_t)c <= vram_budget) { L.plan = TIER_GPU; vused += c; continue; }
        gpu_open = false;
        if (rused + (int64_t)L.bytes() <= ram_budget) { L.plan = TIER_RAM; rused += L.bytes(); }
        else L.plan = TIER_DISK;
    }
    if (rused > 0) {
        SIZE_T mn = (SIZE_T)(rused + (512ll << 20)), mx = (SIZE_T)(rused + kv_bytes + (2048ll << 20));
        SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn, mx, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    }
    stop_ = false;
    migrating_ = true;
    tier_th_ = std::thread([this] { tier_thread(); });
    pre_th_ = std::thread([this] { prefetch_thread(); });
    loaded_ = true;
    load_sec_ = now_sec() - t0;
    return true;
}

struct MemRange { PVOID va; SIZE_T bytes; };
typedef BOOL(WINAPI* PrefetchFn)(HANDLE, ULONG_PTR, MemRange*, ULONG);
// Ask the OS to read a mapped range with large async I/O instead of 4 KB page faults.
static void prefetch_range(const void* p, size_t n) {
    static PrefetchFn pf = (PrefetchFn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    if (!pf || !p || !n) return;
    MemRange e{(PVOID)p, n};
    pf(GetCurrentProcess(), 1, &e, 0);
}

void Model::tier_thread() {
    cuda().bind_thread();
    // Separate read handle: big positioned reads into pinned memory beat page-faulting the mapping.
    // One per file of a split GGUF.
    std::vector<HANDLE> fhs;
    for (auto& p : g_.paths) fhs.push_back(CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0));   // no SEQUENTIAL_SCAN: it evicts the file from the cache right behind the reader
    struct FH { std::vector<HANDLE>& v; ~FH() { for (HANDLE h : v) if (h != INVALID_HANDLE_VALUE) CloseHandle(h); } } fh_guard{fhs};
    auto upload = [&](Lin& l) -> bool {
        if (!l.t) return true;
        HANDLE fh = l.t->shard < (int)fhs.size() ? fhs[l.t->shard] : INVALID_HANDLE_VALUE;
        if (fh != INVALID_HANDLE_VALUE && CudaTier::can_load(l.t->type) && !getenv("ITHACA_SLOW_LOAD")) {
            int kind = 0; uint64_t qoff = 0;
            uint64_t p = cuda().load_weight(fh, l.t->foff, l.t->nbytes, l.t->type, l.rows, l.cols, kind, qoff);
            if (p) {
                l.gkind = kind; l.gqoff = qoff;
                l.dptr.store(p, std::memory_order_release);
                gpu_bytes_ += CudaTier::gpu_bytes(l.t->type, l.rows, l.cols);
                return true;
            }
        }
        // Repack into the GPU layout: [f16 scales][pad to 256][quants]. Q4_0 stays 4-bit,
        // everything else becomes 8-bit. Done on a few side threads, not the inference pool.
        const bool q4 = l.t->type == GT_Q4_0;
        const size_t nb = (size_t)l.cols / 32, nblk = nb * l.rows, qb = q4 ? 16 : 32;
        const size_t qoff = (nblk * 2 + 255) / 256 * 256;
        std::vector<uint8_t> buf(qoff + nblk * qb);
        uint16_t* S = (uint16_t*)buf.data(); uint8_t* Q = buf.data() + qoff;
        const size_t rb_in = gtype_row_bytes(l.t->type, l.cols);
        int nth = std::max(2, (int)std::thread::hardware_concurrency() / 2);
        std::vector<std::thread> th;
        for (int w = 0; w < nth; w++) th.emplace_back([&, w] {
            std::vector<float> row(l.cols);
            for (int r = w; r < l.rows && !stop_; r += nth) {
                const uint8_t* src = l.t->data + r * rb_in;
                for (size_t b = 0; b < nb; b++) {
                    size_t i = r * nb + b;
                    if (q4) { memcpy(&S[i], src + b * 18, 2); memcpy(Q + i * 16, src + b * 18 + 2, 16); }
                    else if (l.t->type == GT_Q8_0) { memcpy(&S[i], src + b * 34, 2); memcpy(Q + i * 32, src + b * 34 + 2, 32); }
                }
                if (q4 || l.t->type == GT_Q8_0) continue;
                dequant_row(l.t->type, src, row.data(), l.cols);
                for (size_t b = 0; b < nb; b++) {
                    const float* x = row.data() + b * 32; float amax = 0;
                    for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(x[j]));
                    float d = amax / 127.f, id = d ? 1.f / d : 0.f;
                    size_t i = r * nb + b; S[i] = f2h(d);
                    for (int j = 0; j < 32; j++) Q[i * 32 + j] = (uint8_t)(int8_t)std::lrintf(x[j] * id);
                }
            }
        });
        for (auto& t : th) t.join();
        if (stop_) return false;
        size_t bytes = buf.size();
        uint64_t p = cuda().alloc_upload(buf.data(), bytes);
        if (!p) return false;
        l.gkind = q4 ? CudaTier::K_Q4_0 : CudaTier::K_Q8_0;
        l.gqoff = qoff;
        l.dptr.store(p, std::memory_order_release);
        gpu_bytes_ += bytes;
        return true;
    };
    double tt0 = now_sec();
    struct Done { std::atomic<bool>& f; double t0; Model* m; ~Done() { f = false; fprintf(stderr, "[tier] placed in %.2f s (%lld MB VRAM, %lld MB RAM)\n", now_sec() - t0, (long long)(m->gpu_bytes_ >> 20), (long long)(m->ram_bytes_ >> 20)); } } done{migrating_, tt0, this};
    bool any_gpu = out_plan_ == TIER_GPU;
    for (auto& L : layers_) any_gpu |= L.plan == TIER_GPU;
    if (any_gpu) dev_ok_ = alloc_dev_buffers();
    f_gqa_ = nullptr;
    {   // decode attention sized for this model's head layout (query heads per kv head, head dim)
        const int G = hp.n_head / std::max(1, hp.n_kv), NT = (hp.hd + 31) / 32;
        if (dev_ok_ && G >= 1 && G <= 8 && NT <= 8 && G * NT * 32 <= 1024 && !getenv("ITHACA_NO_GQA")) {
            extern std::string gqa_attn_ptx(int, int, bool);
            std::string log;
            if (void* mod = cuda().load_module(gqa_attn_ptx(G, NT, kvq8_), log)) f_gqa_ = cuda().function(mod, "k_attn_gqa");
            else fprintf(stderr, "[gpu] decode attention JIT failed: %s\n", log.c_str());
        }
        if (!f_gqa_) kvq8_ = false;
    }
    if (dev_ok_ && !getenv("ITHACA_NO_TC_PREFILL")) { std::string e; tc_prefill_ = (gx().ok() || gx().init(e)) && gx().tensor_cores(); }
    const size_t kvd = (size_t)hp.n_kv * hp.hd;
    for (size_t li = 0; li < layers_.size(); li++) {
        Layer& L = layers_[li];
        if (stop_) return;
        if (L.plan != TIER_GPU) continue;
        bool ok = true;
        for (Lin* l : L.lins()) ok = ok && upload(*l);
        if (ok && dev_ok_) {
            L.d_an = dev_vec(L.attn_norm); L.d_fn = dev_vec(L.ffn_norm);
            L.d_qn = dev_vec(L.q_norm); L.d_kn = dev_vec(L.k_norm);
            L.d_qan = dev_vec(L.qa_norm); L.d_kvan = dev_vec(L.kva_norm); L.d_shg = dev_vec(L.shg);
            L.d_bq = dev_vec(L.bq); L.d_bk = dev_vec(L.bk); L.d_bv = dev_vec(L.bv);
            L.d_pan = dev_vec(L.post_attn_norm); L.d_pfn = dev_vec(L.post_ffn_norm);
            L.d_convw = dev_vec(L.conv_w); L.d_convb = dev_vec(L.conv_b); L.d_ssma = dev_vec(L.ssm_a); L.d_ssmd = dev_vec(L.ssm_d);
            L.d_dtb = dev_vec(L.ssm_dtb); L.d_ssmnorm = dev_vec(L.ssm_norm); L.d_convst = dev_vec(L.conv_st); L.d_ssmst = dev_vec(L.ssm_st);
            L.d_bo = dev_vec(L.bo); L.d_sinks = dev_vec(L.sinks); L.d_binp = dev_vec(L.b_inp); L.d_bgate = dev_vec(L.b_gate); L.d_bup = dev_vec(L.b_up); L.d_bdown = dev_vec(L.b_down);
            {   // the KV cache starts small and grows with the conversation (ensure_kv); the tier plan
                // already budgets the full context, so growing never pushes this model out of VRAM
                std::lock_guard<std::mutex> kl(kv_m_);
                int cap = std::min(ctx, (std::max({1024, kv_want_, n_past + gpu_batch_}) + 511) / 512 * 512);   // the request in flight, or one more batch
                const size_t eb = kvq8_ ? 1 : 2;
                L.d_kc = cuda().alloc((size_t)cap * kvd * eb); L.d_vc = cuda().alloc((size_t)cap * kvd * eb);
                if (kvq8_) { L.d_ks = cuda().alloc((size_t)cap * hp.n_kv * 4); L.d_vs = cuda().alloc((size_t)cap * hp.n_kv * 4); }
                L.kv_cap = cap;
            }
            ok = L.d_an && L.d_fn && L.d_kc && L.d_vc && (!kvq8_ || (L.d_ks && L.d_vs));
            L.kv_dev = 0;
        }
        if (ok) L.state = TIER_GPU; else L.plan = TIER_RAM;  // VRAM ran out: fall back
    }
    if (out_plan_ == TIER_GPU && !stop_ && upload(out_)) { d_onorm_ = dev_vec(out_norm_); out_state_ = TIER_GPU; }
    for (auto& L : layers_) {
        if (stop_) return;
        if (L.plan != TIER_RAM) continue;
        prefetch_range(L.lo, L.bytes());
        if (!VirtualLock((void*)L.lo, L.bytes())) {
            volatile uint8_t sink = 0;  // could not lock: at least fault it into the page cache
            for (const uint8_t* p = L.lo; p < L.hi && !stop_; p += 4096) sink ^= *p;
        }
        ram_bytes_ += L.bytes();
        L.state = TIER_RAM;
    }
}

void Model::prefetch_thread() {
    PrefetchFn pf = (PrefetchFn)(void*)GetProcAddress(GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    for (;;) {
        int want;
        {
            std::unique_lock<std::mutex> l(pre_m_);
            pre_cv_.wait(l, [&] { return stop_ || pre_want_ >= 0; });
            if (stop_) return;
            want = pre_want_; pre_want_ = -1;
        }
        // Read ahead the next layers that still live on disk so the SSD works while the CPU computes.
        for (int i = want; i < want + 3 && i < (int)layers_.size(); i++) {
            Layer& L = layers_[i];
            if (L.state != TIER_DISK || !L.bytes()) continue;
            if (pf) { MemRange e{(PVOID)L.lo, L.bytes()}; pf(GetCurrentProcess(), 1, &e, 0); }
        }
    }
}

void Model::unload() {
    if (!loaded_ && !g_.base) return;
    for (auto& p : parked_) cuda().host_free(p.host);
    parked_.clear(); parked_bytes_ = 0;
    stop_ = true;
    pre_cv_.notify_all();
    if (tier_th_.joinable()) tier_th_.join();
    if (pre_th_.joinable()) pre_th_.join();
    pool_.stop();
    auto fr = [](Lin& l) { uint64_t p = l.dptr.exchange(0); if (p) cuda().free(p); };
    fr(out_);
    dev_ok_ = false;
    cuda().graph_destroy(graph_); graph_ = nullptr;
    for (auto& kv : seg_graphs_) cuda().graph_destroy(kv.second);
    seg_graphs_.clear();
    for (uint64_t p : dev_allocs_) cuda().free(p);
    dev_allocs_.clear();
    dX_ = dXN_ = dQ_ = dK_ = dV_ = dATT_ = dF1_ = dF2_ = dT_ = dLOG_ = dSC_ = dCS_ = d_onorm_ = 0;
    for (auto& L : layers_) {
        for (Lin* l : L.lins()) fr(*l);
        for (uint64_t* p : {&L.d_kc, &L.d_vc, &L.d_ks, &L.d_vs}) { if (*p) cuda().free(*p); *p = 0; }
        L.kv_cap = 0;
    }
    layers_.clear();
    out_.t = nullptr; out_state_ = TIER_DISK;
    if (dLOGA_) { cuda().free(dLOGA_); dLOGA_ = 0; } loga_rows_ = 0;
    if (dW16_) { cuda().free(dW16_); dW16_ = 0; } w16_cap_ = 0;
    kc_.clear(); kc_.shrink_to_fit(); vc_.clear(); vc_.shrink_to_fit();
    x_.clear(); xn_.clear(); q_.clear(); k_.clear(); v_.clear(); att_.clear(); ff1_.clear(); ff2_.clear(); tmp_.clear();
    bufn_ = 0; gpu_bytes_ = 0; ram_bytes_ = 0; n_past = 0;
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1, 0);
    g_.close();
    loaded_ = false;
}

bool Model::park() {
    if (!loaded_ || !parked_.empty() || !fully_on_gpu()) return false;
    for (double t0 = now_sec(); migrating_ && now_sec() - t0 < 30;) Sleep(20);
    if (migrating_) return false;
    std::vector<Lin*> ls;
    for (auto& L : layers_) for (Lin* l : L.lins()) if (l->dptr.load()) ls.push_back(l);
    if (out_.dptr.load()) ls.push_back(&out_);
    size_t need = 0;
    for (Lin* l : ls) { size_t n = cuda().alloc_size(l->dptr.load()); if (!n) return false; need += n; }
    MEMORYSTATUSEX ms{sizeof(ms)}; GlobalMemoryStatusEx(&ms);
    if (ms.ullAvailPhys < need + (2ull << 30)) return false;   // keep 2 GB of RAM for everything else
    cuda().sync();
    for (Lin* l : ls) {
        uint64_t d = l->dptr.load(); size_t n = cuda().alloc_size(d);
        void* h = cuda().host_alloc(n);
        if (!h || !cuda().dtoh(h, d, n)) { cuda().host_free(h); wake(); return false; }
        parked_.push_back({l, h, n}); parked_bytes_ += (int64_t)n;
        l->dptr.store(0); cuda().free(d);
    }
    // Captured graphs hold the old addresses. The KV cache shrinks to 256 positions (a null cache would mean
    // "this layer is not on the GPU"); it grows again as the next conversation needs.
    cuda().graph_destroy(graph_); graph_ = nullptr;
    for (auto& kv : seg_graphs_) cuda().graph_destroy(kv.second);
    seg_graphs_.clear();
    {
        std::lock_guard<std::mutex> kl(kv_m_);
        const size_t kvd = (size_t)hp.n_kv * hp.hd, eb = kvq8_ ? 1 : 2, per[4] = {kvd * eb, kvd * eb, (size_t)hp.n_kv * 4, (size_t)hp.n_kv * 4};
        const int nbuf = kvq8_ ? 4 : 2, cap = std::min(ctx, 256);
        for (auto& L : layers_) {
            if (!L.d_kc || L.kv_cap <= cap) continue;
            uint64_t* cur[4] = {&L.d_kc, &L.d_vc, &L.d_ks, &L.d_vs};
            uint64_t nb[4] = {}; bool all = true;
            for (int i = 0; i < nbuf && all; i++) all = (nb[i] = cuda().alloc((size_t)cap * per[i])) != 0;
            if (!all) { for (int i = 0; i < nbuf; i++) cuda().free(nb[i]); continue; }   // keep the big cache
            for (int i = 0; i < nbuf; i++) { cuda().free(*cur[i]); *cur[i] = nb[i]; }
            L.kv_cap = cap;
        }
        kv_want_ = 0;
    }
    n_past = 0;
    return true;
}
bool Model::wake() {
    while (!parked_.empty()) {
        Parked& p = parked_.back();
        uint64_t d = cuda().alloc(p.bytes);
        if (!d) return false;
        if (!cuda().htod(d, p.host, p.bytes)) { cuda().free(d); return false; }
        p.l->dptr.store(d);
        cuda().host_free(p.host); parked_bytes_ -= (int64_t)p.bytes;
        parked_.pop_back();
    }
    return true;
}

void Model::ensure_buffers(int n) {
    if (n <= bufn_) return;
    bufn_ = n;
    size_t d = hp.n_embd, qd = (size_t)hp.n_head * hp.hd, kvd = (size_t)hp.n_kv * hp.hd, ff = hp.n_ff;
    x_.resize(n * d); xn_.resize(n * d); q_.resize(n * qd); k_.resize(n * kvd); v_.resize(n * kvd);
    att_.resize(n * qd); ff1_.resize(n * ff * 2); ff2_.resize(n * ff);
    moe_x_.resize(n * d); router_.resize((size_t)n * std::max(1, hp.n_expert));
    tmp_.resize(n * std::max({d, qd + 2 * kvd, 2 * ff}));
    logits_.resize(hp.n_vocab);
    qbuf_.resize(n * (std::max({d, qd, ff}) / 32 + 1));
    if (hp.mla_rank) {
        const size_t nh = hp.n_head;
        ml1_.resize(n * nh * std::max(hp.mla_nope + hp.n_rot, hp.mla_v)); ml2_.resize(n * (size_t)(hp.mla_rank + hp.n_rot));
        ml3_.resize(n * (size_t)std::max(hp.mla_rank, hp.mla_qlora)); ml4_.resize(n * nh * (hp.mla_nope + hp.mla_v));
    }
}

void Model::matmul(Lin& w, const float* X, int n, float* Y) {
    uint64_t dp = w.dptr.load(std::memory_order_acquire);
    if (dp && cuda().matmul(dp, w.gqoff, w.gkind, X, n, w.cols, w.rows, Y)) return;
    matmul_raw(w.t->type, w.t->data, w.rows, w.cols, X, n, Y);
}

void Model::matmul_raw(int type, const uint8_t* base, int rows, int cols, const float* X, int n, float* Y) {
    const size_t rb = gtype_row_bytes(type, cols);
    if (type == GT_Q4_0 || type == GT_Q8_0) {
        const int nb = cols / 32;
        for (int t = 0; t < n; t++) quantize_act(X + (size_t)t * cols, qbuf_.data() + (size_t)t * nb, cols);
        const BQ8* qb = qbuf_.data();
        pool_.run(rows, 16, [&](int64_t b, int64_t e, int) {
            for (int64_t r = b; r < e; r++) {
                const uint8_t* wr = base + r * rb;
                for (int t = 0; t < n; t++)
                    Y[(size_t)t * rows + r] = type == GT_Q4_0 ? dot_q4_0(wr, qb + (size_t)t * nb, nb) : dot_q8_0(wr, qb + (size_t)t * nb, nb);
            }
        });
        return;
    }
    pool_.run(rows, 16, [&](int64_t b, int64_t e, int tid) {
        float* buf = scratch_[tid].data();
        for (int64_t r = b; r < e; r++) {
            const float* wr;
            if (type == GT_F32) wr = (const float*)(base + r * rb);
            else { dequant_row(type, base + r * rb, buf, cols); wr = buf; }
            for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = dot_f32(wr, X + (size_t)t * cols, cols);
        }
    });
}

void Model::rmsnorm(const float* x, const float* w, float* y, int n, int dim) {
    for (int t = 0; t < n; t++) {
        const float* a = x + (size_t)t * dim; float* o = y + (size_t)t * dim;
        float ss = dot_f32(a, a, dim);
        float s = 1.0f / std::sqrt(ss / dim + hp.eps);
        for (int i = 0; i < dim; i++) o[i] = a[i] * s * w[i];
    }
}

void Model::rope(float* x, int nheads, int pos, bool swa) {
    int half = hp.n_rot / 2;
    float cs[512], sn[512];
    const std::vector<float>& f = swa ? inv_freq_swa_ : inv_freq_;
    for (int i = 0; i < half; i++) { float th = pos * f[i]; cs[i] = std::cos(th); sn[i] = std::sin(th); }
    for (int h = 0; h < nheads; h++) {
        float* v = x + (size_t)h * hp.hd;
        for (int i = 0; i < half; i++) {
            int i0 = hp.neox ? i : 2 * i, i1 = hp.neox ? i + half : 2 * i + 1;
            float a = v[i0], b = v[i1];
            v[i0] = a * cs[i] - b * sn[i];
            v[i1] = a * sn[i] + b * cs[i];
        }
    }
}


uint64_t Model::dev_vec(const std::vector<float>& v) {
    if (v.empty()) return 0;
    uint64_t p = cuda().alloc_upload(v.data(), v.size() * 4);
    if (p) dev_allocs_.push_back(p);
    return p;
}

bool Model::alloc_dev_buffers() {
    size_t B = gpu_batch_, d = hp.n_embd, qd = (size_t)hp.n_head * hp.hd, kvd = (size_t)hp.n_kv * hp.hd, ff = hp.n_ff;
    size_t tmax = std::max({d, qd + 2 * kvd, 2 * ff});
    struct { uint64_t* p; size_t n; } bufs[] = {
        {&dX_, B * d}, {&dXN_, B * d}, {&dQ_, B * qd}, {&dK_, B * kvd}, {&dV_, B * kvd}, {&dATT_, B * qd},
        {&dF1_, B * ff}, {&dF2_, B * ff}, {&dT_, B * tmax}, {&dLOG_, (size_t)hp.n_vocab},
        {&dSC_, (size_t)sc_rows_ * hp.n_head * (size_t)ctx}, {&dCS_, B * (size_t)hp.n_rot}, {&dCS2_, B * (size_t)hp.n_rot}, {&dPOS_, 4},
        {&dR_, B * (size_t)std::max(1, hp.n_expert)}, {&dG_, B * d}, {&dIdx_, B * (size_t)std::max(1, hp.n_expert_used)}, {&dWt_, B * (size_t)std::max(1, hp.n_expert_used)},
        {&dPM_, (size_t)hp.n_head * 16}, {&dPL_, (size_t)hp.n_head * 16}, {&dPO_, (size_t)hp.n_head * 16 * hp.hd}};
    if (hp.mla_rank) {
        const size_t nh = hp.n_head;
        struct { uint64_t* p; size_t n; } mb[] = {{&dML1_, B * nh * std::max(hp.mla_nope + hp.n_rot, hp.mla_v)}, {&dML2_, B * (hp.mla_rank + hp.n_rot)}, {&dML3_, B * std::max(hp.mla_rank, hp.mla_qlora)}};
        for (auto& b : mb) { *b.p = cuda().alloc(b.n * 4); if (!*b.p) return false; dev_allocs_.push_back(*b.p); }
    }
    if (recurrent_) {
        const size_t xbc = hp.ssm_inner + 2 * hp.ssm_groups * hp.ssm_state, tot = std::max<size_t>(hp.ssm_inner + xbc + hp.ssm_heads, 3 * d);
        struct { uint64_t* p; size_t n; } sb[] = {{&dZX_, B * tot}, {&dXC_, B * std::max<size_t>(xbc, 1)}, {&dSY_, B * std::max<size_t>(hp.ssm_inner, d)}};
        for (auto& b : sb) { *b.p = cuda().alloc(b.n * 4); if (!*b.p) return false; dev_allocs_.push_back(*b.p); }
    }
    if (hp.n_expert > 0) {   // grouped expert matmuls for prompts: every (token, expert) row at once
        const size_t R = B * (size_t)std::max(1, hp.n_expert_used), fe = hp.n_ff_exp;
        struct { uint64_t* p; size_t n; } moe[] = {{&dMG_, R * d}, {&dMF1_, R * fe}, {&dMF2_, R * fe}, {&dMT_, R * d}, {&dOff_, (size_t)hp.n_expert + 1}, {&dSI_, R}, {&dSW_, R}};
        for (auto& b : moe) { *b.p = cuda().alloc(b.n * 4); if (*b.p) dev_allocs_.push_back(*b.p); }
        if (!dMG_ || !dMF1_ || !dMF2_ || !dMT_ || !dOff_ || !dSI_ || !dSW_) dMG_ = 0;   // the per-expert loop still works
    }
    for (auto& b : bufs) {
        *b.p = cuda().alloc(b.n * 4);
        if (!*b.p) return false;
        dev_allocs_.push_back(*b.p);
    }
    return cuda().reserve_act(B * std::max({d, qd, ff}));
}

bool Model::all_gpu() const {
    if (out_state_ != TIER_GPU) return false;
    for (auto& L : layers_) if (L.state != TIER_GPU) return false;
    return true;
}

void Model::fill_cs(int n) {
    int half = hp.n_rot / 2;
    cs_.resize((size_t)n * half * 2);
    cs2_.resize(hp.swa_window ? cs_.size() : 0);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < half; i++) {
            float th = (n_past + t) * inv_freq_[i];
            cs_[((size_t)t * half + i) * 2] = std::cos(th);
            cs_[((size_t)t * half + i) * 2 + 1] = std::sin(th);
            if (!cs2_.empty()) {
                float t2 = (n_past + t) * inv_freq_swa_[i];
                cs2_[((size_t)t * half + i) * 2] = std::cos(t2);
                cs2_[((size_t)t * half + i) * 2 + 1] = std::sin(t2);
            }
        }
}

// Top-k routing: softmax over all experts, keep the k best (renormalized when the model says so).
void Model::route(const float* lg, int n, std::vector<std::vector<std::pair<int, float>>>& pe) {
    const int E = hp.n_expert, K = std::min(hp.n_expert_used, hp.n_expert);
    pe.assign(E, {});
    std::vector<std::pair<float, int>> p(E);
    for (int t = 0; t < n; t++) {
        const float* l = lg + (size_t)t * E;
        float mx = *std::max_element(l, l + E), s = 0;
        for (int e = 0; e < E; e++) { p[e] = {std::exp(l[e] - mx), e}; s += p[e].first; }
        std::partial_sort(p.begin(), p.begin() + K, p.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        float ws = 0;
        for (int k = 0; k < K; k++) ws += p[k].first;
        const float norm = hp.expert_norm ? ws : s;
        for (int k = 0; k < K; k++) pe[p[k].second].push_back({t, p[k].first / norm});
    }
}

static inline float act_silu(float x) { return x / (1.0f + std::exp(-x)); }
void Model::post_logits(float* v, size_t n) const {
    if (hp.logit_scale != 1.f) for (size_t i = 0; i < n; i++) v[i] /= hp.logit_scale;
    if (hp.final_softcap > 0) for (size_t i = 0; i < n; i++) v[i] = hp.final_softcap * std::tanh(v[i] / hp.final_softcap);
}

// Mamba-2 (Granite 4.0 H) on the CPU, token by token: in-projection to (z, xBC, dt); causal depthwise conv over xBC
// with the last conv_kernel - 1 inputs kept as state, SiLU; per head the scan S = exp(dt A) S + dt x B^T, y = S C + D x;
// y * silu(z), RMS-normed per group, out-projection. Follows llama.cpp's build_mamba2_layer (MIT, the ggml authors).
// Qwen3.5 Gated DeltaNet on the CPU (same math as k_dnet): conv over [q | k | v] (no bias) + SiLU; per value head h
// (key head h % groups): S *= exp(A softplus(alpha + dt bias)); delta = (v - S^T k) sigmoid(beta); S += k delta^T;
// o = S^T q (q, k L2-normed, q / sqrt(dk)); o RMS-normed * w * silu(z); out-projection. After the HF Qwen3-Next reference.
void Model::dnet_cpu(Layer& L, int n) {
    const int dk = hp.ssm_state, nk = hp.ssm_groups, nh = hp.ssm_heads, dv = hp.ssm_inner / nh, ck = hp.ssm_conv, C = 2 * nk * dk + hp.ssm_inner;
    std::vector<float> qkv((size_t)n * C), z((size_t)n * hp.ssm_inner), al((size_t)n * nh), be((size_t)n * nh), y((size_t)n * hp.ssm_inner), xc(C);
    matmul(L.ssm_in, xn_.data(), n, qkv.data()); matmul(L.dn_z, xn_.data(), n, z.data());
    matmul(L.dn_a, xn_.data(), n, al.data()); matmul(L.dn_b, xn_.data(), n, be.data());
    for (int t = 0; t < n; t++) {
        const float* xb = &qkv[(size_t)t * C];
        for (int c = 0; c < C; c++) {
            const float* w = &L.conv_w[(size_t)c * ck];
            float s = w[ck - 1] * xb[c];
            for (int k = 0; k < ck - 1; k++) s += w[k] * L.conv_st[(size_t)k * C + c];
            for (int k = 0; k < ck - 2; k++) L.conv_st[(size_t)k * C + c] = L.conv_st[(size_t)(k + 1) * C + c];
            L.conv_st[(size_t)(ck - 2) * C + c] = xb[c];
            xc[c] = act_silu(s);
        }
        for (int h = 0; h < 2 * nk; h++) {   // L2 norm of every q and k head
            float* v = &xc[(size_t)h * dk]; float ss = 0; for (int i = 0; i < dk; i++) ss += v[i] * v[i];
            const float r = 1.f / std::sqrt(ss + 1e-6f) * (h < nk ? 1.f / std::sqrt((float)dk) : 1.f);
            for (int i = 0; i < dk; i++) v[i] *= r;
        }
        float* yt = &y[(size_t)t * hp.ssm_inner];
        pool_.run(nh, 1, [&](int64_t h0, int64_t h1, int) {
            for (int64_t h = h0; h < h1; h++) {
                const int kh = (int)(h % nk);   // the GGUF orders value heads so they tile over the key heads
                const float* q = &xc[(size_t)kh * dk]; const float* k = &xc[(size_t)(nk + kh) * dk]; const float* v = &xc[(size_t)2 * nk * dk + h * dv];
                float a = al[(size_t)t * nh + h] + L.ssm_dtb[h]; a = a > 20.f ? a : std::log1p(std::exp(a));
                const float eg = std::exp(L.ssm_a[h] * a), beta = 1.f / (1.f + std::exp(-be[(size_t)t * nh + h]));
                float* S = &L.ssm_st[(size_t)h * dk * dv];   // [i key][j value]
                float* o = yt + h * dv; float ss = 0;
                float kv[512], dl[512];   // row-wise over S so the inner loops are contiguous (dv <= 512)
                for (int j = 0; j < dv; j++) { kv[j] = 0; o[j] = 0; }
                for (int i = 0; i < dk; i++) { float* Si = S + (size_t)i * dv; const float ki = k[i]; for (int j = 0; j < dv; j++) { Si[j] *= eg; kv[j] += Si[j] * ki; } }
                for (int j = 0; j < dv; j++) dl[j] = (v[j] - kv[j]) * beta;
                for (int i = 0; i < dk; i++) { float* Si = S + (size_t)i * dv; const float ki = k[i], qi = q[i]; for (int j = 0; j < dv; j++) { Si[j] += ki * dl[j]; o[j] += Si[j] * qi; } }
                for (int j = 0; j < dv; j++) ss += o[j] * o[j];
                const float r = 1.f / std::sqrt(ss / dv + hp.eps);
                for (int j = 0; j < dv; j++) o[j] *= r * L.ssm_norm[j] * act_silu(z[(size_t)t * hp.ssm_inner + h * dv + j]);
            }
        });
    }
    matmul(L.ssm_out, y.data(), n, tmp_.data());
}

void Model::dnet_gpu(Layer& L, int n, bool same_x) {
    CudaTier& cu = cuda();
    unsigned un = n, dk = hp.ssm_state, nk = hp.ssm_groups, nh = hp.ssm_heads, C = 2 * nk * dk + hp.ssm_inner, hpk = 0, zero = 0;   // hpk 0: value head h uses key head h % nk
    float eps = hp.eps;
    uint64_t dAl = dF2_, dBe = dF2_ + (uint64_t)gpu_batch_ * nh * 4;
    mmq(L.ssm_in.dptr.load(), L.ssm_in.gqoff, L.ssm_in.gkind, dXN_, n, L.ssm_in.cols, L.ssm_in.rows, dZX_, same_x);
    mmq(L.dn_z.dptr.load(), L.dn_z.gqoff, L.dn_z.gkind, dXN_, n, L.dn_z.cols, L.dn_z.rows, dF1_, true);
    cu.matmul_dev(L.dn_a.dptr.load(), L.dn_a.gqoff, L.dn_a.gkind, dXN_, n, L.dn_a.cols, L.dn_a.rows, dAl);
    cu.matmul_dev(L.dn_b.dptr.load(), L.dn_b.gqoff, L.dn_b.gkind, dXN_, n, L.dn_b.cols, L.dn_b.rows, dBe);
    cu.launch(CudaTier::SSM_CONV, (C + 255) / 256, 1, 256, {&dZX_, &L.d_convw, &L.d_convb, &L.d_convst, &dXC_, &un, &C, &C, &zero});
    cu.launch(CudaTier::DNET, nh, 1, 128, {&dXC_, &dF1_, &dAl, &dBe, &L.d_ssma, &L.d_dtb, &L.d_ssmnorm, &L.d_ssmst, &dSY_, &un, &C, &nk, &hpk, &nh, &eps});
    mmq(L.ssm_out.dptr.load(), L.ssm_out.gqoff, L.ssm_out.gkind, dSY_, n, L.ssm_out.cols, L.ssm_out.rows, dT_);
}

// LFM2 short convolution: (b, c, x) = in_proj(x); v = b * x; y = c * conv3(v) over the last 3 tokens; out_proj(y).
void Model::sconv_cpu(Layer& L, int n) {
    const int d = hp.n_embd;
    std::vector<float> bcx((size_t)n * 3 * d), y((size_t)n * d);
    matmul(L.ssm_in, xn_.data(), n, bcx.data());
    for (int t = 0; t < n; t++) {
        const float* r = &bcx[(size_t)t * 3 * d];
        for (int c = 0; c < d; c++) {
            const float v = r[c] * r[2 * d + c], *w = &L.conv_w[(size_t)c * 3];
            const float o = w[0] * L.conv_st[c] + w[1] * L.conv_st[d + c] + w[2] * v;
            L.conv_st[c] = L.conv_st[d + c]; L.conv_st[d + c] = v;
            y[(size_t)t * d + c] = r[d + c] * o;
        }
    }
    matmul(L.ssm_out, y.data(), n, tmp_.data());
}

void Model::mamba_cpu(Layer& L, int n) {
    const int di = hp.ssm_inner, ds = hp.ssm_state, ng = hp.ssm_groups, nh = hp.ssm_heads, hdim = di / nh, ck = hp.ssm_conv;
    const int xbc = di + 2 * ng * ds, tot = di + xbc + nh, d = hp.n_embd;
    std::vector<float> zx((size_t)n * tot), y((size_t)n * di), xc(xbc);
    matmul(L.ssm_in, xn_.data(), n, zx.data());
    for (int t = 0; t < n; t++) {
        const float* z = &zx[(size_t)t * tot]; const float* xb = z + di; const float* dt = xb + xbc;
        for (int c = 0; c < xbc; c++) {
            const float* w = &L.conv_w[(size_t)c * ck];
            float s = L.conv_b[c] + w[ck - 1] * xb[c];
            for (int k = 0; k < ck - 1; k++) s += w[k] * L.conv_st[(size_t)k * xbc + c];
            for (int k = 0; k < ck - 2; k++) L.conv_st[(size_t)k * xbc + c] = L.conv_st[(size_t)(k + 1) * xbc + c];
            L.conv_st[(size_t)(ck - 2) * xbc + c] = xb[c];
            xc[c] = act_silu(s);
        }
        const float* xs = xc.data(); const float* B = xs + di; const float* C = B + ng * ds;
        float* yt = &y[(size_t)t * di];
        pool_.run(nh, 1, [&](int64_t h0, int64_t h1, int) {
            for (int64_t h = h0; h < h1; h++) {
                float v = dt[h] + L.ssm_dtb[h]; v = v > 20.f ? v : std::log1p(std::exp(v));
                const float dA = std::exp(v * L.ssm_a[h]); const int g = (int)(h / (nh / ng));
                const float* Bg = B + g * ds; const float* Cg = C + g * ds;
                for (int p = 0; p < hdim; p++) {
                    const int i = (int)h * hdim + p; const float xv = xs[i]; float* S = &L.ssm_st[(size_t)i * ds]; float acc = 0;
                    for (int k = 0; k < ds; k++) { S[k] = S[k] * dA + v * xv * Bg[k]; acc += S[k] * Cg[k]; }
                    yt[i] = acc + L.ssm_d[h] * xv;
                }
            }
        });
        for (int i = 0; i < di; i++) yt[i] *= act_silu(z[i]);
        const int gs = di / ng;
        for (int g = 0; g < ng; g++) {
            float ss = 0; for (int i = 0; i < gs; i++) ss += yt[g * gs + i] * yt[g * gs + i];
            const float r = 1.f / std::sqrt(ss / gs + hp.eps);
            for (int i = 0; i < gs; i++) yt[g * gs + i] *= r * L.ssm_norm[g * gs + i];
        }
    }
    (void)d;
    matmul(L.ssm_out, y.data(), n, tmp_.data());
}
// gpt-oss: gate clamped to <= 7, up to [-7, 7]; gate * sigmoid(1.702 gate) * (up + 1)
static inline float act_oai(float g, float u) { g = std::min(g, 7.f); u = std::min(7.f, std::max(-7.f, u)); return g / (1.f + std::exp(-1.702f * g)) * (u + 1.f); }
static inline float act_gelu(float x) { return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// Mixture-of-experts FFN on the CPU: tokens are grouped per expert so each expert's weights are read once.
void Model::moe_cpu(Layer& L, int n) {
    const int d = hp.n_embd, ff = hp.n_ff_exp;
    matmul(L.gate_inp, xn_.data(), n, router_.data());
    if (!L.b_inp.empty()) for (int t = 0; t < n; t++) for (int e = 0; e < hp.n_expert; e++) router_[(size_t)t * hp.n_expert + e] += L.b_inp[e];
    std::vector<std::vector<std::pair<int, float>>> pe;
    route(router_.data(), n, pe);
    auto bias = [&](const std::vector<float>& b, int e, float* y, int c, int w) { if (!b.empty()) for (int i = 0; i < c; i++) for (int j = 0; j < w; j++) y[(size_t)i * w + j] += b[(size_t)e * w + j]; };
    const size_t eg = gtype_row_bytes(L.gate_exps.t->type, d) * ff, eu = gtype_row_bytes(L.up_exps.t->type, d) * ff,
                 ed = gtype_row_bytes(L.down_exps.t->type, ff) * d;
    for (int e = 0; e < hp.n_expert; e++) {
        const int c = (int)pe[e].size();
        if (!c) continue;
        for (int i = 0; i < c; i++) memcpy(&moe_x_[(size_t)i * d], &xn_[(size_t)pe[e][i].first * d], d * 4);
        matmul_raw(L.gate_exps.t->type, L.gate_exps.t->data + e * eg, ff, d, moe_x_.data(), c, ff1_.data());
        matmul_raw(L.up_exps.t->type, L.up_exps.t->data + e * eu, ff, d, moe_x_.data(), c, ff2_.data());
        bias(L.b_gate, e, ff1_.data(), c, ff); bias(L.b_up, e, ff2_.data(), c, ff);
        if (hp.oai) for (size_t i = 0; i < (size_t)c * ff; i++) ff1_[i] = act_oai(ff1_[i], ff2_[i]);
        else for (size_t i = 0; i < (size_t)c * ff; i++) ff1_[i] = (hp.gelu ? act_gelu(ff1_[i]) : act_silu(ff1_[i])) * ff2_[i];
        matmul_raw(L.down_exps.t->type, L.down_exps.t->data + e * ed, d, ff, ff1_.data(), c, tmp_.data());
        bias(L.b_down, e, tmp_.data(), c, d);
        for (int i = 0; i < c; i++) {
            float w = pe[e][i].second * hp.res_scale * hp.exp_scale;
            float* xr = &x_[(size_t)pe[e][i].first * d]; const float* yr = &tmp_[(size_t)i * d];
            for (int j = 0; j < d; j++) xr[j] += w * yr[j];
        }
    }
}

// Expert slice of a stacked GPU weight: rows [e*R, (e+1)*R).
static void expert_slice(const Lin& l, int e, int ne, uint64_t& W, uint64_t& qoff) {
    const size_t nb = l.cols / 32, R = l.rows / ne, boff = (size_t)e * R * nb;
    if (l.gkind >= CudaTier::K_IQ) { W = l.dptr.load() + (uint64_t)e * R * gtype_row_bytes(l.t->type, l.cols); qoff = 0; return; }
    const size_t sbytes = CudaTier::scale_bytes(l.gkind, boff), qb = l.gkind == CudaTier::K_Q8_0 ? 32 : l.gkind == CudaTier::K_Q6 ? 24 : 16;
    W = l.dptr.load() + sbytes;
    qoff = l.gqoff + boff * qb - sbytes;
}

// Mixture-of-experts FFN on the GPU. Routing is read back to the host (one small sync per layer),
// then each used expert runs on its gathered tokens and scatters the weighted result into x.
void Model::moe_gpu(Layer& L, int n) {
    CudaTier& cu = cuda();
    const int E = hp.n_expert;
    unsigned d = hp.n_embd, ff = hp.n_ff_exp;
    cu.matmul_dev(L.gate_inp.dptr.load(), L.gate_inp.gqoff, L.gate_inp.gkind, dXN_, n, L.gate_inp.cols, L.gate_inp.rows, dR_);
    if (L.d_binp) { unsigned ue = E; cu.ew(CudaTier::BIAS, (unsigned)n * E, {&dR_, &L.d_binp, &ue}); }
    auto act = [&](uint64_t a, uint64_t b, unsigned cnt) { cu.ew(hp.oai ? CudaTier::OAI : hp.gelu ? CudaTier::GELU : CudaTier::SILU, cnt, {&a, &b}); };
    if (n == 1) {
        // Decode: route on the device and run the K experts as slots of one launch each (graph-capturable).
        const int K = std::min(hp.n_expert_used, E);
        cu.topk(dR_, E, K, hp.expert_norm, hp.res_scale * hp.exp_scale, dIdx_, dWt_);
        cu.matmul_experts(L.gate_exps.dptr.load(), L.gate_exps.gqoff, L.gate_exps.gkind, dIdx_, K, dXN_, 1, 0, d, ff, dF1_);
        cu.matmul_experts(L.up_exps.dptr.load(), L.up_exps.gqoff, L.up_exps.gkind, dIdx_, K, dXN_, 1, 0, d, ff, dF2_);
        if (L.d_bgate) { cu.ew(CudaTier::XBIAS, K * ff, {&dF1_, &L.d_bgate, &dIdx_, &ff}); cu.ew(CudaTier::XBIAS, K * ff, {&dF2_, &L.d_bup, &dIdx_, &ff}); }
        act(dF1_, dF2_, K * ff);
        cu.matmul_experts(L.down_exps.dptr.load(), L.down_exps.gqoff, L.down_exps.gkind, dIdx_, K, dF1_, K, 1, ff, d, dT_);
        if (L.d_bdown) cu.ew(CudaTier::XBIAS, K * d, {&dT_, &L.d_bdown, &dIdx_, &d});
        cu.wsum(dX_, dT_, dWt_, K, d);
        return;
    }
    if (tc_prefill_ && dMG_ && moe_grouped(L, n)) return;
    cu.dtoh(router_.data(), dR_, (size_t)n * E * 4);
    std::vector<std::vector<std::pair<int, float>>> pe;
    route(router_.data(), n, pe);
    std::vector<uint32_t> idx; std::vector<float> w; std::vector<size_t> off(E + 1);
    for (int e = 0; e < E; e++) {
        off[e] = idx.size();
        for (auto& p : pe[e]) { idx.push_back(p.first); w.push_back(p.second * hp.res_scale * hp.exp_scale); }
    }
    off[E] = idx.size();
    if (idx.empty()) return;
    cu.htod_async(dIdx_, idx.data(), idx.size() * 4);
    cu.htod_async(dWt_, w.data(), w.size() * 4);
    for (int e = 0; e < E; e++) {
        unsigned c = (unsigned)(off[e + 1] - off[e]);
        if (!c) continue;
        uint64_t pI = dIdx_ + off[e] * 4, pW = dWt_ + off[e] * 4, W, q;
        cu.ew(CudaTier::GATHER, c * d, {&dXN_, &pI, &dG_, &d});
        expert_slice(L.gate_exps, e, E, W, q); mmq(W, q, L.gate_exps.gkind, dG_, c, d, ff, dF1_);
        expert_slice(L.up_exps, e, E, W, q); mmq(W, q, L.up_exps.gkind, dG_, c, d, ff, dF2_, true);
        if (L.d_bgate) {
            uint64_t bg = L.d_bgate + (uint64_t)e * ff * 4, bu = L.d_bup + (uint64_t)e * ff * 4;
            cu.ew(CudaTier::BIAS, c * ff, {&dF1_, &bg, &ff}); cu.ew(CudaTier::BIAS, c * ff, {&dF2_, &bu, &ff});
        }
        act(dF1_, dF2_, c * ff);
        expert_slice(L.down_exps, e, E, W, q); mmq(W, q, L.down_exps.gkind, dF1_, c, ff, d, dT_);
        if (L.d_bdown) { uint64_t bd = L.d_bdown + (uint64_t)e * d * 4; cu.ew(CudaTier::BIAS, c * d, {&dT_, &bd, &d}); }
        cu.ew(CudaTier::SCATTER, c * d, {&dX_, &dT_, &pI, &pW, &d});
    }
}

// Prompt MoE in one launch per matrix: routing and grouping on the GPU (no host round trip), every expert's rows
// gathered at once, grouped int8 matmuls (grid z = expert), one atomic scatter back into x.
bool Model::moe_grouped(Layer& L, int n) {
    CudaTier& cu = cuda();
    const int E = hp.n_expert, K = std::min(hp.n_expert_used, E);
    unsigned d = hp.n_embd, ff = hp.n_ff_exp, tot = (unsigned)(n * K);
    // check first: the router kernel overwrites the logits, so a fallback after it would route wrongly.
    // IQ / MXFP4 experts: each stacked matrix goes to the Q8_0 layout once (scratch dW16_), then the same grouped kernels.
    auto ok_kind = [&](const Lin& l) { return l.gkind >= CudaTier::K_IQ ? gx().can_group(CudaTier::K_Q8_0) : gx().can_group(l.gkind); };
    if (!ok_kind(L.gate_exps) || !ok_kind(L.up_exps) || !ok_kind(L.down_exps)) return false;
    for (const Lin* l : {&L.gate_exps, &L.up_exps, &L.down_exps})
        if (l->gkind >= CudaTier::K_IQ) {
            const size_t nblk = (size_t)l->rows / E * std::min(E, kExpChunk) * (l->cols / 32), need = (nblk * 2 + 255) / 256 * 256 + nblk * 32;
            if (need > w16_cap_) { cu.sync(); if (dW16_) cu.free(dW16_); dW16_ = cu.alloc(need); w16_cap_ = dW16_ ? need : 0; }
            if (!dW16_) return false;
        }
    if (!cu.moe_route(dR_, n, E, K, hp.expert_norm, hp.res_scale * hp.exp_scale, dIdx_, dWt_, dOff_, dSI_, dSW_)) return false;
    auto grouped = [&](const Lin& l, uint64_t x, unsigned K_, unsigned N, uint64_t y, bool same_x) {
        const size_t nb = l.cols / 32, R = l.rows / E;
        if (l.gkind >= CudaTier::K_IQ) {   // a few experts at a time through the Q8_0 scratch (rows come from the absolute offsets)
            const size_t rb = gtype_row_bytes(l.t->type, l.cols);
            for (int e0 = 0; e0 < E; e0 += kExpChunk) {
                const int c = std::min(kExpChunk, E - e0);
                const size_t nblk = (size_t)c * R * nb, qo = (nblk * 2 + 255) / 256 * 256;
                if (!cu.iq_to_q8(l.dptr.load() + (uint64_t)e0 * R * rb, l.gkind, l.cols, (int)(c * R), dW16_, dW16_ + qo) ||
                    !gx().linear_quant_grouped(x, tot, K_, dW16_, dW16_ + qo, CudaTier::K_Q8_0, N, y, dOff_ + (uint64_t)e0 * 4, c, n, R * nb * 2, R * nb * 32, same_x || e0 > 0))
                    return false;
            }
            return true;
        }
        const size_t qb = l.gkind == CudaTier::K_Q8_0 ? 32 : l.gkind == CudaTier::K_Q6 ? 24 : 16;
        uint64_t S = l.dptr.load();
        return gx().linear_quant_grouped(x, tot, K_, S, S + l.gqoff, l.gkind, N, y, dOff_, E, n, CudaTier::scale_bytes(l.gkind, R * nb), R * nb * qb, same_x);
    };
    cu.ew(CudaTier::GATHER, tot * d, {&dXN_, &dSI_, &dMG_, &d});
    if (L.d_bgate) { unsigned ue = E; cu.ew(CudaTier::ROWEXP, tot, {&dIdx_, &dOff_, &ue}); }   // expert of each grouped row, for the biases
    if (!grouped(L.gate_exps, dMG_, d, ff, dMF1_, false) || !grouped(L.up_exps, dMG_, d, ff, dMF2_, true)) return false;
    if (L.d_bgate) { cu.ew(CudaTier::XBIAS, tot * ff, {&dMF1_, &L.d_bgate, &dIdx_, &ff}); cu.ew(CudaTier::XBIAS, tot * ff, {&dMF2_, &L.d_bup, &dIdx_, &ff}); }
    const bool fused = !hp.oai && gx().act_quantize(dMF1_, dMF2_, tot, ff, hp.gelu);
    if (!fused) cu.ew(hp.oai ? CudaTier::OAI : hp.gelu ? CudaTier::GELU : CudaTier::SILU, tot * ff, {&dMF1_, &dMF2_});
    if (!grouped(L.down_exps, dMF1_, ff, d, dMT_, fused)) return false;
    if (L.d_bdown) cu.ew(CudaTier::XBIAS, tot * d, {&dMT_, &L.d_bdown, &dIdx_, &d});
    cu.ew(CudaTier::SCATTER, tot * d, {&dX_, &dMT_, &dSI_, &dSW_, &d});
    return true;
}

// Grows every VRAM layer's KV buffers to hold `need` positions (doubling, capped at ctx).
// If VRAM is short, on_vram_short gets a chance to free memory (the app drops the draft model).
bool Model::ensure_kv(int need) {
    if (!dev_ok_) return true;
    std::lock_guard<std::mutex> kl(kv_m_);
    kv_want_ = std::max(kv_want_, need);
    const size_t kvd = (size_t)hp.n_kv * hp.hd;
    bool grew = false;
    for (auto& L : layers_) {
        if (!L.d_kc || L.kv_cap >= need) continue;
        int cap = std::min(ctx, std::max(need, L.kv_cap * 2));
        cap = std::min(ctx, (cap + 255) / 256 * 256);
        // K, V (and their scales when int8): bytes per position for each buffer
        const size_t eb = kvq8_ ? 1 : 2, per[4] = {kvd * eb, kvd * eb, (size_t)hp.n_kv * 4, (size_t)hp.n_kv * 4};
        uint64_t* cur[4] = {&L.d_kc, &L.d_vc, &L.d_ks, &L.d_vs};
        const int nbuf = kvq8_ ? 4 : 2;
        uint64_t nb[4] = {};
        for (int attempt = 0; attempt < 2; attempt++) {
            bool all = true;
            for (int i = 0; i < nbuf && all; i++) all = (nb[i] = cuda().alloc((size_t)cap * per[i])) != 0;
            if (all) break;
            for (int i = 0; i < nbuf; i++) if (nb[i]) { cuda().free(nb[i]); nb[i] = 0; }
            if (attempt || !on_vram_short || !on_vram_short()) return false;
        }
        for (int i = 0; i < nbuf; i++) cuda().dtod(nb[i], *cur[i], (size_t)L.kv_cap * per[i]);
        cuda().sync();
        for (int i = 0; i < nbuf; i++) { cuda().free(*cur[i]); *cur[i] = nb[i]; }
        L.kv_cap = cap;
        grew = true;
    }
    if (grew) {   // captured graphs hold the old KV pointers
        cuda().graph_destroy(graph_); graph_ = nullptr;
        for (auto& kv : seg_graphs_) cuda().graph_destroy(kv.second);
        seg_graphs_.clear();
    }
    return true;
}

int64_t Model::kv_bytes_gpu() const {
    int64_t b = 0;
    for (auto& L : layers_) if (L.d_kc) b += (int64_t)L.kv_cap * kv_pos_bytes();
    return b;
}
size_t Model::kv_pos_bytes() const {
    const size_t kvd = (size_t)hp.n_kv * hp.hd;
    return kvq8_ ? kvd * 2 + (size_t)hp.n_kv * 8 : kvd * 4;
}

void Model::rewind(int n) {
    if (recurrent_ && n < n_past) {   // a state can't be rolled back: start over
        n = 0;
        for (auto& L : layers_) {
            std::fill(L.conv_st.begin(), L.conv_st.end(), 0.f); std::fill(L.ssm_st.begin(), L.ssm_st.end(), 0.f);
            if (L.d_convst) cuda().zero(L.d_convst, L.conv_st.size() * 4);
            if (L.d_ssmst) cuda().zero(L.d_ssmst, L.ssm_st.size() * 4);
        }
    }
    if (n < n_past) n_past = n;
    for (auto& L : layers_) if (L.kv_dev > n) L.kv_dev = n;
}

// Mamba-2 on the GPU (same math as mamba_cpu): in-projection, conv (k_ssm_conv), scan (k_ssm_scan, the state stays in
// VRAM), gate + group norm (k_ssm_gnorm), out-projection into dT_. No host syncs, so decode stays graph-captured.
void Model::mamba_gpu(Layer& L, int n, bool same_x) {
    CudaTier& cu = cuda();
    unsigned un = n, di = hp.ssm_inner, ds = hp.ssm_state, ng = hp.ssm_groups, nh = hp.ssm_heads, hdim = di / nh, hpg = nh / ng;
    unsigned xbc = di + 2 * ng * ds, tot = di + xbc + nh, dto = di + xbc, bo = di, co = di + ng * ds, gs = di / ng;
    float eps = hp.eps;
    mmq(L.ssm_in.dptr.load(), L.ssm_in.gqoff, L.ssm_in.gkind, dXN_, n, L.ssm_in.cols, L.ssm_in.rows, dZX_, same_x);
    cu.launch(CudaTier::SSM_CONV, (xbc + 255) / 256, 1, 256, {&dZX_, &L.d_convw, &L.d_convb, &L.d_convst, &dXC_, &un, &xbc, &tot, &di});
    cu.launch(n == 1 ? CudaTier::SSM_SCAN1 : CudaTier::SSM_SCAN, di / (256 / ds), 1, 256, {&dXC_, &dZX_, &L.d_dtb, &L.d_ssma, &L.d_ssmd, &L.d_ssmst, &dSY_, &un, &xbc, &tot, &dto, &di, &ds, &hdim, &hpg, &bo, &co});
    cu.launch(CudaTier::SSM_GNORM, ng, un, 256, {&dSY_, &dZX_, &L.d_ssmnorm, &di, &tot, &gs, &eps});
    mmq(L.ssm_out.dptr.load(), L.ssm_out.gqoff, L.ssm_out.gkind, dSY_, n, L.ssm_out.cols, L.ssm_out.rows, dT_);
}

// DeepSeek V2 multi-head latent attention, expanded: kv_a = W x -> latent (RMS-normed) + one shared RoPE key;
// [k_nope | v] per head = W_kvb latent. Heads are written padded to hp.hd as [rope | nope | 0] (q and k) and
// [v | 0], so RoPE (first n_rot dims), the KV cache and the attention kernels are the ordinary ones.
void Model::mla_gpu(Layer& L, int n, bool same_x) {
    CudaTier& cu = cuda();
    unsigned un = n, nh = hp.n_head, hd = hp.hd, nope = hp.mla_nope, nr = hp.n_rot, vd = hp.mla_v, rank = hp.mla_rank, ka = rank + nr;
    float eps = hp.eps;
    mmq(L.kva.dptr.load(), L.kva.gqoff, L.kva.gkind, dXN_, n, L.kva.cols, L.kva.rows, dML2_, same_x);
    if (L.qa.t) {
        unsigned ql = L.qa.rows;
        mmq(L.qa.dptr.load(), L.qa.gqoff, L.qa.gkind, dXN_, n, L.qa.cols, L.qa.rows, dML3_, true);
        cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dML3_, &L.d_qan, &dML3_, &ql, &eps});
        mmq(L.qb.dptr.load(), L.qb.gqoff, L.qb.gkind, dML3_, n, L.qb.cols, L.qb.rows, dML1_);
    } else mmq(L.q.dptr.load(), L.q.gqoff, L.q.gkind, dXN_, n, L.q.cols, L.q.rows, dML1_, true);
    cu.ew(CudaTier::COPY2D, un * rank, {&dML2_, &ka, &dML3_, &rank, &rank});
    cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dML3_, &L.d_kvan, &dML3_, &rank, &eps});
    mmq(L.kvb.dptr.load(), L.kvb.gqoff, L.kvb.gkind, dML3_, n, L.kvb.cols, L.kvb.rows, dT_);
    cu.ew(CudaTier::MLA, un * nh * hd, {&dML1_, &dML2_, &dT_, &dQ_, &dK_, &dV_, &nh, &hd, &nope, &nr, &vd, &rank});
}

void Model::mla_cpu(Layer& L, int n) {
    const int nh = hp.n_head, hd = hp.hd, nope = hp.mla_nope, nr = hp.n_rot, vd = hp.mla_v, rank = hp.mla_rank, ka = rank + nr, qk = nope + nr;
    matmul(L.kva, xn_.data(), n, ml2_.data());
    if (L.qa.t) {
        matmul(L.qa, xn_.data(), n, ml3_.data());
        rmsnorm(ml3_.data(), L.qa_norm.data(), ml3_.data(), n, L.qa.rows);
        matmul(L.qb, ml3_.data(), n, ml1_.data());
    } else matmul(L.q, xn_.data(), n, ml1_.data());
    for (int t = 0; t < n; t++) memcpy(&ml3_[(size_t)t * rank], &ml2_[(size_t)t * ka], rank * 4);
    rmsnorm(ml3_.data(), L.kva_norm.data(), ml3_.data(), n, rank);
    matmul(L.kvb, ml3_.data(), n, ml4_.data());
    std::fill(q_.begin(), q_.begin() + (size_t)n * nh * hd, 0.f); std::fill(k_.begin(), k_.begin() + (size_t)n * nh * hd, 0.f); std::fill(v_.begin(), v_.begin() + (size_t)n * nh * hd, 0.f);
    for (int t = 0; t < n; t++)
        for (int h = 0; h < nh; h++) {
            const size_t th = (size_t)t * nh + h;
            float* q = &q_[th * hd]; float* k = &k_[th * hd]; float* v = &v_[th * hd];
            const float* qs = &ml1_[th * qk]; const float* kv = &ml4_[th * (nope + vd)];
            memcpy(q, qs + nope, nr * 4); memcpy(q + nr, qs, nope * 4);
            memcpy(k, &ml2_[(size_t)t * ka + rank], nr * 4); memcpy(k + nr, kv, nope * 4);
            memcpy(v, kv + nope, vd * 4);
        }
}

// Prompt batches go through the int8 tensor-core kernels; single tokens use the dp4a kernels.
void Model::mmq(uint64_t W, uint64_t qoff, int kind, uint64_t x, int n, int cols, int rows, uint64_t y, bool same_x) {
    if (kind >= CudaTier::K_IQ && n >= 16 && tc_prefill_ && gx().can_mmq(CudaTier::K_Q8_0)) {
        // IQ prompts: the matrix goes to the Q8_0 layout once, then the int8 tensor-core matmul
        const size_t nblk = (size_t)rows * (cols / 32), qo = (nblk * 2 + 255) / 256 * 256, need = qo + nblk * 32;
        if (need > w16_cap_) { cuda().sync(); if (dW16_) cuda().free(dW16_); dW16_ = cuda().alloc(need); w16_cap_ = dW16_ ? need : 0; }
        if (dW16_ && cuda().iq_to_q8(W, kind, cols, rows, dW16_, dW16_ + qo) &&
            gx().linear_quant(x, n, cols, dW16_, dW16_ + qo, CudaTier::K_Q8_0, rows, y, same_x)) return;
    }
    if (n >= 16 && tc_prefill_ && gx().linear_quant(x, n, cols, W, W + qoff, kind, rows, y, same_x)) return;
    cuda().matmul_dev(W, qoff, kind, x, n, cols, rows, y, same_x);
}

void Model::gpu_layer(Layer& L, int n, unsigned nc) {
    CudaTier& cu = cuda();
    unsigned d = hp.n_embd, hd = hp.hd, nh = hp.n_head, nkv = hp.n_kv, qd = nh * hd, kvd = nkv * hd, ff = hp.n_ff;
    unsigned half = hp.n_rot / 2, neox = hp.neox ? 1 : 0, grp = nh / nkv, ctxu = ctx, un = n;
    float eps = hp.eps, res = hp.res_scale, scale = hp.attn_scale > 0 ? hp.attn_scale : 1.0f / std::sqrt((float)hd);
    static bool pp = getenv("ITHACA_PREPROF") != nullptr;
    const bool ppn = pp && n > 1;
    static double t_mm = 0;
    auto mm = [&](Lin& w, uint64_t x, uint64_t y, bool same_x = false) {
        double t = ppn ? (cu.sync(), now_sec()) : 0;
        mmq(w.dptr.load(), w.gqoff, w.gkind, x, n, w.cols, w.rows, y, same_x);
        if (ppn) { cu.sync(); t_mm += now_sec() - t; }
    };
    double T0 = ppn ? (cu.sync(), now_sec()) : 0;
    // Prompts whose matmuls all take the int8 path: norm (and the residual add before it) quantizes straight into
    // the int8 input, so the matmuls skip their own quantize pass.
    auto int8 = [&](Lin& w) { return n >= 16 && tc_prefill_ && (w.gkind >= CudaTier::K_IQ ? gx().can_mmq(CudaTier::K_Q8_0) : gx().can_mmq(w.gkind)); };
    auto attn8 = [&](Layer& l) { return !hp.no_prenorm && (l.ssm_in.t ? int8(l.ssm_in) : l.qkv.t ? int8(l.qkv) : int8(l.q) && int8(l.k) && int8(l.v)); };
    static double t_attn = 0, t_all = 0; static int calls = 0;
    const bool fa = (xn_ready_ == &L && xn_n_ == n && xn_past_ == n_past) || (attn8(L) && gx().add_norm_quant(dX_, 0, 0, 0.f, L.d_an, dXN_, n, d, eps));   // xn_ready_: the last layer already did it
    xn_ready_ = nullptr;
    if (hp.no_prenorm) cu.dtod(dXN_, dX_, (size_t)n * d * 4);
    else if (!fa) cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &L.d_an, &dXN_, &d, &eps});
    if (L.dnet) dnet_gpu(L, n, fa);
    else if (L.sconv) {
        unsigned d3 = 3 * d;
        mmq(L.ssm_in.dptr.load(), L.ssm_in.gqoff, L.ssm_in.gkind, dXN_, n, L.ssm_in.cols, L.ssm_in.rows, dZX_, fa);
        cu.launch(CudaTier::SCONV, (d + 255) / 256, 1, 256, {&dZX_, &L.d_convw, &L.d_convst, &dSY_, &un, &d});
        mmq(L.ssm_out.dptr.load(), L.ssm_out.gqoff, L.ssm_out.gkind, dSY_, n, L.ssm_out.cols, L.ssm_out.rows, dT_);
        (void)d3;
    }
    else if (L.ssm_in.t) mamba_gpu(L, n, fa);
    else {
    if (L.kva.t) mla_gpu(L, n, fa);
    else if (L.qkv.t) {
        mm(L.qkv, dXN_, dT_, fa);
        unsigned W = qd + 2 * kvd;
        uint64_t sk = dT_ + qd * 4ull, sv = dT_ + (qd + kvd) * 4ull;
        cu.ew(CudaTier::COPY2D, un * qd, {&dT_, &W, &dQ_, &qd, &qd});
        cu.ew(CudaTier::COPY2D, un * kvd, {&sk, &W, &dK_, &kvd, &kvd});
        cu.ew(CudaTier::COPY2D, un * kvd, {&sv, &W, &dV_, &kvd, &kvd});
    } else {
        if (L.qgate) {   // per head [q | gate]: q to dQ_, the gate stays in dT_ for the output
            unsigned W2 = 2 * hd, nhq = un * nh;
            mm(L.q, dXN_, dT_, fa); cu.ew(CudaTier::COPY2D, nhq * hd, {&dT_, &W2, &dQ_, &hd, &hd});
        } else mm(L.q, dXN_, dQ_, fa);
        mm(L.k, dXN_, dK_, true); mm(L.v, dXN_, dV_, true);
    }
    if (L.d_bq) cu.ew(CudaTier::BIAS, un * qd, {&dQ_, &L.d_bq, &qd});
    if (L.d_bk) cu.ew(CudaTier::BIAS, un * kvd, {&dK_, &L.d_bk, &kvd});
    if (L.d_bv) cu.ew(CudaTier::BIAS, un * kvd, {&dV_, &L.d_bv, &kvd});
    if (L.d_qn) { const bool w = (int)L.q_norm.size() == (int)qd; cu.launch(CudaTier::RMSNORM, w ? un : un * nh, 1, 256, {&dQ_, &L.d_qn, &dQ_, w ? &qd : &hd, &eps}); }
    if (L.d_kn) { const bool w = (int)L.k_norm.size() == (int)kvd; cu.launch(CudaTier::RMSNORM, w ? un : un * nkv, 1, 256, {&dK_, &L.d_kn, &dK_, w ? &kvd : &hd, &eps}); }
    uint64_t cs = L.swa ? dCS2_ : dCS_;
    unsigned win = L.swa ? (unsigned)hp.swa_window : 0;
    if (!L.nope) {
        cu.launch(CudaTier::ROPE, un, 1, 256, {&dQ_, &cs, &nh, &hd, &half, &neox, &qd});
        cu.launch(CudaTier::ROPE, un, 1, 256, {&dK_, &cs, &nkv, &hd, &half, &neox, &kvd});
    }
    if (kvq8_) {
        cu.launch(CudaTier::KVSTORE_Q8, un * nkv, 1, 32, {&dK_, &L.d_kc, &L.d_ks, &dPOS_, &hd, &nkv});
        cu.launch(CudaTier::KVSTORE_Q8, un * nkv, 1, 32, {&dV_, &L.d_vc, &L.d_vs, &dPOS_, &hd, &nkv});
    } else {
        cu.ew(CudaTier::KVSTORE, un * kvd, {&dK_, &L.d_kc, &dPOS_, &kvd});
        cu.ew(CudaTier::KVSTORE, un * kvd, {&dV_, &L.d_vc, &dPOS_, &kvd});
    }
    // Decode splits each head's keys over 16 CTAs (flash-decoding) so long chats keep the GPU busy.
    unsigned spl = n == 1 ? 16 : 1;
    double Ta = ppn ? (cu.sync(), now_sec()) : 0;
    // Prompts: tensor-core flash attention straight from the KV cache. Decode: one CTA per kv head reads the
    // cache once for all query heads sharing it (GQA), split over 16 key ranges; else the per-head kernel.
    if (n == 1 && f_gqa_) {
        void* a[] = {&dQ_, &L.d_kc, &L.d_vc, &dPM_, &dPL_, &dPO_, &hd, &kvd, &grp, &dPOS_, &scale, &spl, &win, &L.d_ks, &L.d_vs, &nkv, &L.d_sinks, &hp.attn_softcap};
        cu.launch_raw(f_gqa_, nkv, spl, 1, 256, 0, a);
    }
    else if (!(n > 1 && tc_prefill_ && gx().attn_prefill(dQ_, L.d_kc, L.d_vc, dATT_, n, n_past, qd, kvd, nh, grp, hd, scale, win, nc,
                                                        kvq8_ ? L.d_ks : 0, kvq8_ ? L.d_vs : 0, L.d_sinks, hp.attn_softcap)))
        cu.launch(CudaTier::ATTN, nh, un * spl, 256, {&dQ_, &L.d_kc, &L.d_vc, &dATT_, &dSC_, &nh, &hd, &kvd, &grp, &dPOS_, &ctxu, &scale, &spl, &dPM_, &dPL_, &dPO_, &win, &nc});
    if (spl > 1) cu.launch(CudaTier::ATTN_COMB, nh, un, std::max(32u, (hd + 31) / 32 * 32), {&dPM_, &dPL_, &dPO_, &dATT_, &nh, &hd, &spl});
    if (ppn) { cu.sync(); t_attn += now_sec() - Ta; }
    if (L.qgate) { unsigned W2 = 2 * hd; uint64_t g = dT_ + (uint64_t)hd * 4; cu.ew(CudaTier::SIGMUL, un * qd, {&dATT_, &g, &hd, &W2}); }
    if (L.kva.t) {   // MLA: the first value_length of each padded head
        unsigned vd = hp.mla_v, nhv = un * nh;
        cu.ew(CudaTier::COPY2D, nhv * vd, {&dATT_, &hd, &dML1_, &vd, &vd});
        mm(L.o, dML1_, dT_);
    } else
    mm(L.o, dATT_, dT_);
    if (L.d_bo) cu.ew(CudaTier::BIAS, un * d, {&dT_, &L.d_bo, &d});
    }
    if (ppn && ++calls % 64 == 0) fprintf(stderr, "[prefill] attention %.0f ms, matmuls %.0f ms of %.0f ms layer time\n", t_attn * 1e3, t_mm * 1e3, t_all * 1e3);
    struct PP { bool on; double t0; double& acc; ~PP() { if (on) { cuda().sync(); acc += now_sec() - t0; } } } pp_guard{ppn, T0, t_all};
    // MoE: the experts gather their own tokens, so only the add + norm are saved (the int8 copy goes unused)
    const bool ff8 = !hp.no_prenorm && (L.gate_exps.t ? n >= 16 && tc_prefill_ : L.gateup.t ? int8(L.gateup) : int8(L.gate) && int8(L.up));
    const bool fb = ff8 && gx().add_norm_quant(dX_, dT_, L.d_pan, res, L.d_fn, dXN_, n, d, eps);
    if (!fb) {
        if (L.d_pan) cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dT_, &L.d_pan, &dT_, &d, &eps});
        cu.ew(CudaTier::ADD, un * d, {&dX_, &dT_, &res});
        if (hp.no_prenorm) cu.dtod(dXN_, dX_, (size_t)n * d * 4);
        else cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &L.d_fn, &dXN_, &d, &eps});
    }
    if (L.gate.t) ff = L.gate.rows; else if (L.gateup.t) ff = L.gateup.rows / 2;
    if (L.gate_exps.t) {
        if (L.shexp) {   // shared experts: dense SwiGLU on every token, (gated and) added, then the routed experts
            mm(L.gate, dXN_, dF1_, fb); mm(L.up, dXN_, dF2_, true);
            if (n >= 16 && tc_prefill_ && gx().can_mmq(L.down.gkind) && gx().act_quantize(dF1_, dF2_, n, ff, false)) mm(L.down, dF1_, dT_, true);
            else { cu.ew(CudaTier::SILU, un * ff, {&dF1_, &dF2_}); mm(L.down, dF1_, dT_); }
            if (L.d_shg) cu.launch(CudaTier::GADD, un, 1, 256, {&dX_, &dT_, &dXN_, &L.d_shg, &d, &res});
            else cu.ew(CudaTier::ADD, un * d, {&dX_, &dT_, &res});
        }
        moe_gpu(L, n); return;
    }
    if (L.gateup.t) {
        mm(L.gateup, dXN_, dT_, fb);
        cu.ew(CudaTier::SILU_SPLIT, un * ff, {&dT_, &dF1_, &ff});
    } else {
        mm(L.gate, dXN_, dF1_, fb); mm(L.up, dXN_, dF2_, true);
        // prompts: the activation goes straight into the down projection's int8 input (dF1_ itself is not written)
        if (n >= 16 && tc_prefill_ && gx().can_mmq(L.down.gkind) && gx().act_quantize(dF1_, dF2_, n, ff, hp.gelu)) { mm(L.down, dF1_, dT_, true); goto ffn_done; }
        cu.ew(hp.gelu ? CudaTier::GELU : CudaTier::SILU, un * ff, {&dF1_, &dF2_});
    }
    mm(L.down, dF1_, dT_);
ffn_done:
    {   // prompts: this residual add and the next layer's input norm + quantize in one pass
        const size_t li = &L - layers_.data();
        Layer* nx = li + 1 < layers_.size() ? &layers_[li + 1] : nullptr;
        if (nx && nx->state == TIER_GPU && nx->d_an && attn8(*nx) && gx().add_norm_quant(dX_, dT_, L.d_pfn, res, nx->d_an, dXN_, n, d, eps)) { xn_ready_ = nx; xn_n_ = n; xn_past_ = n_past; return; }
    }
    if (L.d_pfn) cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dT_, &L.d_pfn, &dT_, &d, &eps});
    cu.ew(CudaTier::ADD, un * d, {&dX_, &dT_, &res});
}

void Model::bench_gpu() {
    if (!dev_ok_ || layers_.empty() || layers_[0].state != TIER_GPU) { printf("gpu bench: layer 0 not on GPU\n"); return; }
    CudaTier& cu = cuda();
    Layer& L = layers_[0];
    const int R = 200;
    auto timeit = [&](const char* name, double bytes, auto fn) {
        float sink[4]; fn(); cu.dtoh(sink, dX_, 4);
        double t0 = now_sec();
        for (int i = 0; i < R; i++) fn();
        cu.dtoh(sink, dX_, 4);
        double us = (now_sec() - t0) / R * 1e6;
        printf("  %-14s %8.1f us  %6.1f GB/s\n", name, us, bytes / (us * 1e-6) / 1e9);
    };
    Lin* big = L.gate.t ? &L.gate : &L.gateup;
    Lin* down = &L.down;
    Lin* kk = L.k.t ? &L.k : &L.qkv;
    auto cost = [](Lin* l) { return (double)l->rows * (l->cols / 32) * (l->gkind == CudaTier::K_Q4_0 ? 18 : l->gkind == CudaTier::K_Q4M ? 20 : l->gkind == CudaTier::K_Q4K ? 18.5 : l->gkind == CudaTier::K_Q6 ? 28 : 34); };
    timeit("mm gate/up", cost(big), [&] { cu.matmul_dev(big->dptr.load(), big->gqoff, big->gkind, dXN_, 1, big->cols, big->rows, dT_); });
    timeit("mm down", cost(down), [&] { cu.matmul_dev(down->dptr.load(), down->gqoff, down->gkind, dF1_, 1, down->cols, down->rows, dT_); });
    timeit("mm k", cost(kk), [&] { cu.matmul_dev(kk->dptr.load(), kk->gqoff, kk->gkind, dXN_, 1, kk->cols, kk->rows, dK_); });
    unsigned d = hp.n_embd; float eps = hp.eps, one = 1.f;
    timeit("rmsnorm", d * 8.0, [&] { cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&dX_, &L.d_an, &dXN_, &d, &eps}); });
    timeit("add", d * 12.0, [&] { cu.ew(CudaTier::ADD, d, {&dXN_, &dT_, &one}); });
    unsigned hd = hp.hd, nh = hp.n_head, kvd = hp.n_kv * hp.hd, grp = hp.n_head / hp.n_kv, ctxu = ctx; float sc = 0.1f;
    ensure_kv(std::min(ctx, 7000));
    for (unsigned pos : {64u, 1024u, 3000u, 6000u}) {
        if ((int)pos >= ctx) break;
        char nm[32]; snprintf(nm, sizeof nm, "attn @%u", pos);
        cu.htod(dPOS_, &pos, 4);
        unsigned spl = 16;
        unsigned z0 = 0;
        if (!kvq8_) timeit(nm, (double)pos * kvd * 4, [&] {   // the per-head kernel reads fp16 caches only
            cu.launch(CudaTier::ATTN, nh, spl, 256, {&dQ_, &L.d_kc, &L.d_vc, &dATT_, &dSC_, &nh, &hd, &kvd, &grp, &dPOS_, &ctxu, &sc, &spl, &dPM_, &dPL_, &dPO_, &z0, &z0});
            cu.launch(CudaTier::ATTN_COMB, nh, 1, 128, {&dPM_, &dPL_, &dPO_, &dATT_, &nh, &hd, &spl});
        });
        if (f_gqa_) {
            snprintf(nm, sizeof nm, "gqa @%u", pos);
            unsigned z1 = 0;
            timeit(nm, (double)pos * kvd * 4, [&] {
                unsigned nkv1 = hp.n_kv;
                void* a[] = {&dQ_, &L.d_kc, &L.d_vc, &dPM_, &dPL_, &dPO_, &hd, &kvd, &grp, &dPOS_, &sc, &spl, &z1, &L.d_ks, &L.d_vs, &nkv1, &L.d_sinks, &hp.attn_softcap};
                cu.launch_raw(f_gqa_, hp.n_kv, spl, 1, 256, 0, a);
            });
        }
    }
    if (out_.dptr.load()) timeit("mm output", cost(&out_), [&] { cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_); });
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

static double g_ev[4]; static int g_evn;
const float* Model::eval(const int* toks, int n_all) {
    struct EvT { bool on; double t0; ~EvT() { if (on && ++g_evn % 20 == 0) fprintf(stderr, "[eval_all] layers %.1f head %.1f total %.1f ms\n", g_ev[0] / g_evn * 1e3, g_ev[1] / g_evn * 1e3, g_ev[2] / g_evn * 1e3); } };
    const bool evp = all_logits_ && getenv("ITHACA_SPECPROF");
    double ev0 = now_sec();
    EvT evt{evp, ev0};
    if (!loaded_ || n_all <= 0 || n_past + n_all > ctx) return nullptr;
    if (!parked_.empty() && !wake()) return nullptr;
    // A long prompt right after a (re)load reads far faster once the layers are in VRAM than on the CPU, and the
    // placement is fixed per request below, so give the background placement a moment to finish first.
    if (n_all > 64 && migrating_) for (double t0 = now_sec(); migrating_ && now_sec() - t0 < 30;) Sleep(20);
    if (!ensure_kv(n_past + n_all)) return nullptr;
    if (n_all == 1 && toks[0] >= 0 && dev_ok_ && !migrating_ && all_gpu()) {
        // Fully VRAM-resident decode: replay one CUDA graph for the whole forward pass.
        CudaTier& cu = cuda();
        const int d = hp.n_embd;
        bool kv_ok = true;
        for (auto& L : layers_) kv_ok &= L.kv_dev >= n_past;
        if (kv_ok) {
            ensure_buffers(1);
            size_t erb = gtype_row_bytes(embd_->type, d);
            int id = std::min(std::max(toks[0], 0), hp.n_vocab - 1);
            dequant_row(embd_->type, embd_->data + (size_t)id * erb, x_.data(), d);
            if (hp.emb_scale != 1.f) for (int i = 0; i < d; i++) x_[i] *= hp.emb_scale;
            fill_cs(1);
            unsigned p = n_past;
            cu.htod_async(dX_, x_.data(), (size_t)d * 4);
            cu.htod_async(dCS_, cs_.data(), cs_.size() * 4);
            if (!cs2_.empty()) cu.htod_async(dCS2_, cs2_.data(), cs2_.size() * 4);
            cu.htod_async(dPOS_, &p, 4);
            if (!graph_ && cu.begin_capture()) {
                for (auto& L : layers_) gpu_layer(L, 1);
                unsigned ud = d; float eps = hp.eps;
                cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&dX_, &d_onorm_, &dXN_, &ud, &eps});
                cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_);
                graph_ = cu.end_capture();
                if (!graph_) fprintf(stderr, "[gpu] graph capture failed, using direct launches\n");
            }
            if (graph_) {
                cu.graph_launch(graph_);
                cu.dtoh(logits_.data(), dLOG_, (size_t)hp.n_vocab * 4);
                post_logits(logits_.data(), logits_.size());
                for (auto& L : layers_) L.kv_dev = n_past + 1;
                n_past += 1;
                return logits_.data();
            }
        }
    }
    const int B = dev_ok_ ? gpu_batch_ : 256;
    // Placement is fixed for the whole request: a layer that reaches VRAM mid-request joins at the next one.
    std::vector<char> on_gpu(hp.n_layer);
    for (int i = 0; i < hp.n_layer; i++) on_gpu[i] = dev_ok_ && layers_[i].state == TIER_GPU;
    const bool head_gpu = dev_ok_ && out_state_ == TIER_GPU;
    for (int c0 = 0, n = 0; c0 < n_all; c0 += n) {
        // batches never mix text and image tokens: an image run is one non-causal block
        const bool img = toks[c0] < 0;
        n = 1;
        while (c0 + n < n_all && n < B && (toks[c0 + n] < 0) == img) n++;
        const unsigned nc = img ? (unsigned)n : 0;
        const int* tk = toks + c0;
        ensure_buffers(n);
        const int d = hp.n_embd, hd = hp.hd, qd = hp.n_head * hd, kvd = hp.n_kv * hd, ff_all = hp.n_ff;
        size_t erb = gtype_row_bytes(embd_->type, d);
        for (int t = 0; t < n; t++) {
            if (tk[t] < 0) {   // precomputed embedding (image): used as is, no embedding scale
                size_t row = (size_t)(-1 - tk[t]);
                if ((row + 1) * d <= ext_embd.size()) memcpy(&x_[(size_t)t * d], &ext_embd[row * d], d * 4);
                else memset(&x_[(size_t)t * d], 0, d * 4);
                continue;
            }
            int id = std::min(std::max(tk[t], 0), hp.n_vocab - 1);
            dequant_row(embd_->type, embd_->data + (size_t)id * erb, &x_[(size_t)t * d], d);
            if (hp.emb_scale != 1.f) for (int i = 0; i < d; i++) x_[(size_t)t * d + i] *= hp.emb_scale;
        }
        const float scale = hp.attn_scale > 0 ? hp.attn_scale : 1.0f / std::sqrt((float)hd);
        const int grp = hp.n_head / hp.n_kv;
        const size_t kvdz = (size_t)kvd;
        bool on_dev = false, cs_up = false;
        for (int li = 0; li < hp.n_layer; li++) {
            Layer& L = layers_[li];
            { std::lock_guard<std::mutex> l(pre_m_); pre_want_ = li + 1; }
            pre_cv_.notify_one();
            if (on_gpu[li]) {
                CudaTier& cu = cuda();
                if (!on_dev) { cu.htod_async(dX_, x_.data(), (size_t)n * d * 4); on_dev = true; }
                if (!cs_up) {
                    fill_cs(n);
                    cu.htod_async(dCS_, cs_.data(), cs_.size() * 4);
                    if (!cs2_.empty()) cu.htod_async(dCS2_, cs2_.data(), cs2_.size() * 4);
                    unsigned p = n_past; cu.htod_async(dPOS_, &p, 4);
                    cu.dtoh(&p, dPOS_, 4);  // small sync so the host buffers above are consumed
                    cs_up = true;
                }
                auto kv_sync = [&](int k) {  // layer just moved to VRAM: bring its CPU-computed KV along
                    Layer& M = layers_[k];
                    if (M.kv_dev >= n_past) return;
                    // recurrent layers: the host state is at n_past (the CPU ran this layer until now)
                    if (M.d_convst) cu.htod(M.d_convst, M.conv_st.data(), M.conv_st.size() * 4);
                    if (M.d_ssmst) cu.htod(M.d_ssmst, M.ssm_st.data(), M.ssm_st.size() * 4);
                    size_t off = ((size_t)k * ctx + M.kv_dev) * kvdz;
                    const size_t npos = (size_t)(n_past - M.kv_dev), cnt = npos * kvdz * 2;
                    if (kvq8_) {   // quantize the fp16 host rows the same way k_kvstore_q8 does
                        const int nkv = hp.n_kv, hdd = hp.hd;
                        std::vector<int8_t> qk(npos * kvdz), qv(npos * kvdz);
                        std::vector<float> sk(npos * nkv), sv(npos * nkv);
                        for (int which = 0; which < 2; which++) {
                            const uint16_t* src = which ? &vc_[off] : &kc_[off];
                            int8_t* q = which ? qv.data() : qk.data(); float* sc = which ? sv.data() : sk.data();
                            for (size_t p = 0; p < npos; p++)
                                for (int h = 0; h < nkv; h++) {
                                    const uint16_t* r = src + p * kvdz + (size_t)h * hdd; float mx = 0;
                                    for (int i = 0; i < hdd; i++) mx = std::max(mx, std::fabs(h2f(r[i])));
                                    float inv = mx > 0 ? 127.f / mx : 0.f;
                                    sc[p * nkv + h] = mx / 127.f;
                                    for (int i = 0; i < hdd; i++) q[p * kvdz + (size_t)h * hdd + i] = (int8_t)std::lrintf(h2f(r[i]) * inv);
                                }
                        }
                        cu.htod(M.d_kc + (uint64_t)M.kv_dev * kvdz, qk.data(), qk.size());
                        cu.htod(M.d_vc + (uint64_t)M.kv_dev * kvdz, qv.data(), qv.size());
                        cu.htod(M.d_ks + (uint64_t)M.kv_dev * nkv * 4, sk.data(), sk.size() * 4);
                        cu.htod(M.d_vs + (uint64_t)M.kv_dev * nkv * 4, sv.data(), sv.size() * 4);
                    } else {
                        cu.htod_async(M.d_kc + (uint64_t)M.kv_dev * kvdz * 2, &kc_[off], cnt);
                        cu.htod_async(M.d_vc + (uint64_t)M.kv_dev * kvdz * 2, &vc_[off], cnt);
                    }
                };
                if (n == 1 && !nc && !migrating_) {
                    // Decode: replay a cached CUDA graph for this whole run of VRAM layers.
                    int e = li;
                    while (e < hp.n_layer && on_gpu[e]) kv_sync(e++);
                    void*& g = seg_graphs_[li * 4096 + e];
                    if (!g && cu.begin_capture()) {
                        for (int k = li; k < e; k++) gpu_layer(layers_[k], 1);
                        g = cu.end_capture();
                    }
                    if (g && cu.graph_launch(g)) {
                        for (int k = li; k < e; k++) layers_[k].kv_dev = n_past + 1;
                        li = e - 1;
                        continue;
                    }
                }
                kv_sync(li);
                gpu_layer(L, n, nc);
                L.kv_dev = n_past + n;
                continue;
            }
            if (on_dev) { cuda().dtoh(x_.data(), dX_, (size_t)n * d * 4); on_dev = false; }

            if (hp.no_prenorm) memcpy(xn_.data(), x_.data(), (size_t)n * d * 4);
            else rmsnorm(x_.data(), L.attn_norm.data(), xn_.data(), n, d);
            if (L.dnet) dnet_cpu(L, n);
            else if (L.sconv) sconv_cpu(L, n);
            else if (L.ssm_in.t) mamba_cpu(L, n);
            else {
            if (L.kva.t) mla_cpu(L, n);
            else if (L.qkv.t) {
                matmul(L.qkv, xn_.data(), n, tmp_.data());
                for (int t = 0; t < n; t++) {
                    const float* s = &tmp_[(size_t)t * (qd + 2 * kvd)];
                    memcpy(&q_[(size_t)t * qd], s, qd * 4);
                    memcpy(&k_[(size_t)t * kvd], s + qd, kvd * 4);
                    memcpy(&v_[(size_t)t * kvd], s + qd + kvd, kvd * 4);
                }
            } else {
                if (L.qgate) {   // per head [q | gate]: q to q_, the gate kept for the output
                    qg_.resize((size_t)n * 2 * qd); matmul(L.q, xn_.data(), n, qg_.data());
                    for (int t = 0; t < n; t++) for (int h = 0; h < hp.n_head; h++) memcpy(&q_[(size_t)t * qd + (size_t)h * hd], &qg_[(size_t)t * 2 * qd + (size_t)h * 2 * hd], hd * 4);
                } else
                matmul(L.q, xn_.data(), n, q_.data());
                matmul(L.k, xn_.data(), n, k_.data());
                matmul(L.v, xn_.data(), n, v_.data());
            }
            for (int t = 0; t < n; t++) {
                if (!L.bq.empty()) for (int i = 0; i < qd; i++) q_[(size_t)t * qd + i] += L.bq[i];
                if (!L.bk.empty()) for (int i = 0; i < kvd; i++) k_[(size_t)t * kvd + i] += L.bk[i];
                if (!L.bv.empty()) for (int i = 0; i < kvd; i++) v_[(size_t)t * kvd + i] += L.bv[i];
            }
            // per-head q/k norms, or across all heads at once (OLMo 2, OLMoE: the weight spans the whole projection)
            if (!L.q_norm.empty()) { const bool w = (int)L.q_norm.size() == qd; rmsnorm(q_.data(), L.q_norm.data(), q_.data(), w ? n : n * hp.n_head, w ? qd : hd); }
            if (!L.k_norm.empty()) { const bool w = (int)L.k_norm.size() == kvd; rmsnorm(k_.data(), L.k_norm.data(), k_.data(), w ? n : n * hp.n_kv, w ? kvd : hd); }
            for (int t = 0; t < n; t++) {
                int pos = n_past + t;
                if (!L.nope) { rope(&q_[(size_t)t * qd], hp.n_head, pos, L.swa); rope(&k_[(size_t)t * kvd], hp.n_kv, pos, L.swa); }
                size_t off = ((size_t)li * ctx + pos) * kvd;
                f32_to_f16(&k_[(size_t)t * kvd], &kc_[off], kvd);
                f32_to_f16(&v_[(size_t)t * kvd], &vc_[off], kvd);
            }
            const uint16_t* kbase = &kc_[(size_t)li * ctx * kvd];
            const uint16_t* vbase = &vc_[(size_t)li * ctx * kvd];
            pool_.run((int64_t)n * hp.n_head, 1, [&](int64_t b, int64_t e, int tid) {
                float* sc = scratch_[tid].data();
                for (int64_t idx = b; idx < e; idx++) {
                    int t = (int)(idx / hp.n_head), h = (int)(idx % hp.n_head), kvh = h / grp;
                    // causal (or the whole block for image tokens), limited to the window on sliding layers
                    int pos = nc ? n_past + n - 1 : n_past + t;
                    int j0 = L.swa && hp.swa_window && pos + 1 > hp.swa_window ? pos + 1 - hp.swa_window : 0;
                    const float* qv = &q_[(size_t)t * qd + (size_t)h * hd];
                    float mx = -1e30f;
                    for (int j = j0; j <= pos; j++) {
                        float s = dot_f16(qv, kbase + (size_t)j * kvd + kvh * hd, hd) * scale;
                        if (hp.attn_softcap > 0) s = hp.attn_softcap * std::tanh(s / hp.attn_softcap);
                        sc[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0;
                    if (!L.sinks.empty()) { mx = std::max(mx, L.sinks[h]); sum = std::exp(L.sinks[h] - mx); }   // a logit with no value
                    for (int j = j0; j <= pos; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                    float inv = 1.0f / sum;
                    float* o = &att_[(size_t)t * qd + (size_t)h * hd];
                    memset(o, 0, hd * 4);
                    for (int j = j0; j <= pos; j++) axpy_f16(o, sc[j] * inv, vbase + (size_t)j * kvd + kvh * hd, hd);
                }
            });
            if (L.qgate)
                for (int t = 0; t < n; t++) for (int h = 0; h < hp.n_head; h++) for (int i = 0; i < hd; i++)
                    att_[(size_t)t * qd + (size_t)h * hd + i] /= 1.f + std::exp(-qg_[(size_t)t * 2 * qd + (size_t)h * 2 * hd + hd + i]);
            if (L.kva.t) {   // MLA: the first value_length of each padded head
                const int vd = hp.mla_v;
                for (size_t r = 0; r < (size_t)n * hp.n_head; r++) memcpy(&ml1_[r * vd], &att_[r * hd], vd * 4);
                matmul(L.o, ml1_.data(), n, tmp_.data());
            } else
            matmul(L.o, att_.data(), n, tmp_.data());
            if (!L.bo.empty()) for (int t = 0; t < n; t++) for (int i = 0; i < d; i++) tmp_[(size_t)t * d + i] += L.bo[i];
            }
            if (!L.post_attn_norm.empty()) rmsnorm(tmp_.data(), L.post_attn_norm.data(), tmp_.data(), n, d);
            for (size_t i = 0; i < (size_t)n * d; i++) x_[i] += tmp_[i] * hp.res_scale;

            if (hp.no_prenorm) memcpy(xn_.data(), x_.data(), (size_t)n * d * 4);
            else rmsnorm(x_.data(), L.ffn_norm.data(), xn_.data(), n, d);
            const int ff = L.gate.t ? L.gate.rows : L.gateup.t ? L.gateup.rows / 2 : ff_all;
            if (L.gate_exps.t) {
                if (L.shexp) {   // shared experts: dense SwiGLU on every token, (gated and) added, then the routed experts
                    matmul(L.gate, xn_.data(), n, ff1_.data()); matmul(L.up, xn_.data(), n, ff2_.data());
                    for (size_t i = 0; i < (size_t)n * ff; i++) ff1_[i] = silu(ff1_[i]) * ff2_[i];
                    matmul(L.down, ff1_.data(), n, tmp_.data());
                    for (int t = 0; t < n; t++) {
                        float g = hp.res_scale;
                        if (!L.shg.empty()) { float s = 0; for (int i = 0; i < d; i++) s += xn_[(size_t)t * d + i] * L.shg[i]; g /= 1.f + std::exp(-s); }
                        for (int i = 0; i < d; i++) x_[(size_t)t * d + i] += g * tmp_[(size_t)t * d + i];
                    }
                }
                moe_cpu(L, n); continue;
            }
            if (L.gateup.t) {
                matmul(L.gateup, xn_.data(), n, tmp_.data());
                for (int t = 0; t < n; t++)
                    for (int i = 0; i < ff; i++)
                        ff1_[(size_t)t * ff + i] = silu(tmp_[(size_t)t * 2 * ff + i]) * tmp_[(size_t)t * 2 * ff + ff + i];
            } else {
                matmul(L.gate, xn_.data(), n, ff1_.data());
                matmul(L.up, xn_.data(), n, ff2_.data());
                for (size_t i = 0; i < (size_t)n * ff; i++) ff1_[i] = (hp.gelu ? act_gelu(ff1_[i]) : silu(ff1_[i])) * ff2_[i];
            }
            matmul(L.down, ff1_.data(), n, tmp_.data());
            if (!L.post_ffn_norm.empty()) rmsnorm(tmp_.data(), L.post_ffn_norm.data(), tmp_.data(), n, d);
            for (size_t i = 0; i < (size_t)n * d; i++) x_[i] += tmp_[i] * hp.res_scale;
        }
        if (c0 + n >= n_all && head_gpu && out_.dptr.load()) {
            CudaTier& cu = cuda();
            if (!on_dev) { cu.htod_async(dX_, x_.data(), (size_t)n * d * 4); on_dev = true; }
            unsigned ud = d; float eps = hp.eps;
            if (evp) { cuda().sync(); g_ev[0] += now_sec() - ev0; }
            double evh = now_sec();
            if (all_logits_ && n > 1) {   // every position (draft verification)
                if (n > loga_rows_) {
                    if (dLOGA_) cu.free(dLOGA_);
                    loga_rows_ = std::max(n, 16); dLOGA_ = cu.alloc((size_t)loga_rows_ * hp.n_vocab * 4);
                }
                unsigned un = n;
                cu.launch(CudaTier::RMSNORM, un, 1, 256, {&dX_, &d_onorm_, &dXN_, &ud, &eps});
                mmq(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, n, out_.cols, out_.rows, dLOGA_);
                logits_.resize((size_t)n * hp.n_vocab);
                cu.dtoh(logits_.data(), dLOGA_, logits_.size() * 4);
                post_logits(logits_.data(), logits_.size());
                n_past += n;
                if (evp) { g_ev[1] += now_sec() - evh; g_ev[2] += now_sec() - ev0; }
                continue;
            }
            uint64_t last = dX_ + (uint64_t)(n - 1) * d * 4;
            cu.launch(CudaTier::RMSNORM, 1, 1, 256, {&last, &d_onorm_, &dXN_, &ud, &eps});
            cu.matmul_dev(out_.dptr.load(), out_.gqoff, out_.gkind, dXN_, 1, out_.cols, out_.rows, dLOG_);
            cu.dtoh(logits_.data(), dLOG_, (size_t)hp.n_vocab * 4);
            post_logits(logits_.data(), logits_.size());
            n_past += n;
            continue;
        }
        if (on_dev) { cuda().dtoh(x_.data(), dX_, (size_t)n * d * 4); on_dev = false; }
        n_past += n;
        if (c0 + n >= n_all) {
            const int rows = all_logits_ ? n : 1;
            logits_.resize((size_t)rows * hp.n_vocab);
            rmsnorm(&x_[(size_t)(n - rows) * d], out_norm_.data(), xn_.data(), rows, d);
            matmul(out_, xn_.data(), rows, logits_.data());
            post_logits(logits_.data(), logits_.size());
        }
    }
    return logits_.data();
}

std::string Model::tier_json() {
    std::string s = "{\"loaded\":";
    s += loaded_ ? "true" : "false";
    if (!loaded_) return s + ",\"gpu\":\"" + gpu_init_info() + "\"}";
    char buf[256];
    snprintf(buf, sizeof buf, ",\"gpu_mb\":%lld,\"ram_mb\":%lld,\"load_ms\":%.0f,\"threads\":%d,\"ctx\":%d,\"n_past\":%d,\"out\":[%d,%d],\"layers\":[",
             (long long)(gpu_bytes_ >> 20), (long long)(ram_bytes_ >> 20), load_sec_ * 1000, pool_.size(), ctx, n_past,
             out_plan_, out_state_.load());
    s += buf;
    for (size_t i = 0; i < layers_.size(); i++) {
        snprintf(buf, sizeof buf, "%s[%d,%d,%zu]", i ? "," : "", layers_[i].plan.load(), layers_[i].state.load(), layers_[i].bytes() >> 20);
        s += buf;
    }
    s += std::string("],\"dev\":") + (dev_ok_ ? "true" : "false") + ",\"gpu\":\"" + gpu_info + "\",\"arch\":\"" + hp.arch + "\"}";
    return s;
}

// --quanttest <file.gguf>: for each quant type in the file, load one tensor with the GPU converter
// and compare a GPU matmul against a CPU dequantized reference.
int quant_selftest(const std::wstring& path) {
    GGUF g; std::string err;
    if (!g.open(path, err)) { printf("open failed: %s\n", err.c_str()); return 1; }
    CudaTier& cu = Model::cuda();
    if (!cu.ok()) { printf("no GPU\n"); return 1; }
    HANDLE fh = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    std::vector<int> seen;
    for (auto& t : g.tensors) {
        static const bool layer0 = getenv("ITHACA_MMBENCH") && !strcmp(getenv("ITHACA_MMBENCH"), "layer");   // every matrix of block 0
        if (t.nd != 2 || !CudaTier::can_load(t.type)) continue;
        if (layer0 ? t.name.rfind("blk.0.", 0) != 0 : std::find(seen.begin(), seen.end(), t.type) != seen.end()) continue;
        seen.push_back(t.type);
        int cols = (int)t.ne[0], rows = (int)std::min<int64_t>(t.ne[1], getenv("ITHACA_QT_ROWS") ? atoi(getenv("ITHACA_QT_ROWS")) : 512);
        size_t nbytes = gtype_row_bytes(t.type, cols) * rows;
        int kind; uint64_t qoff;
        uint64_t W = cu.load_weight(fh, (uint64_t)(t.data - g.base), nbytes, t.type, rows, cols, kind, qoff);
        if (!W) { printf("  %-6s load failed\n", gtype_name(t.type)); continue; }
        std::vector<float> x(cols), y(rows), ref(rows), row(cols);
        uint32_t s = 1; for (auto& v : x) { s = s * 1664525 + 1013904223; v = ((s >> 8) & 0xFFFF) / 32768.f - 1.f; }
        cu.matmul(W, qoff, kind, x.data(), 1, cols, rows, y.data());
        double num = 0, den = 0;
        for (int r = 0; r < rows; r++) {
            dequant_row(t.type, t.data + r * gtype_row_bytes(t.type, cols), row.data(), cols);
            double a = 0; for (int c = 0; c < cols; c++) a += (double)row[c] * x[c];
            num += (y[r] - a) * (y[r] - a); den += a * a;
        }
        double e = std::sqrt(num / std::max(den, 1e-30));
        printf("  %-6s %-28s rel.err %.5f  %s\n", gtype_name(t.type), t.name.c_str(), e, e < 0.02 ? "OK" : "BAD");
        {   // several tokens at once (prompts use other kernels): each token row checked against its own reference
            const int T = 11; std::vector<float> X((size_t)T * cols), Y((size_t)T * rows);
            for (auto& v : X) { s = s * 1664525 + 1013904223; v = ((s >> 8) & 0xFFFF) / 32768.f - 1.f; }
            bool ran = cu.matmul(W, qoff, kind, X.data(), T, cols, rows, Y.data());
            double n2 = 0, d2 = 0;
            for (int r = 0; r < rows; r++) {
                dequant_row(t.type, t.data + r * gtype_row_bytes(t.type, cols), row.data(), cols);
                for (int k = 0; k < T; k++) { double a = 0; for (int c = 0; c < cols; c++) a += (double)row[c] * X[(size_t)k * cols + c]; double g = Y[(size_t)k * rows + r]; n2 += (g - a) * (g - a); d2 += a * a; }
            }
            double e2 = std::sqrt(n2 / std::max(d2, 1e-30));
            printf("  %-6s %-28s %d tokens: %s rel.err %.5f  %s\n", gtype_name(t.type), t.name.c_str(), T, ran ? "ran," : "LAUNCH FAILED,", e2, ran && e2 < 0.02 ? "OK" : "BAD");
        }
        if (getenv("ITHACA_PPBENCH")) {   // full-size tensor, a 512-token prompt through the int8 tensor-core matmul
            int R = (int)t.ne[1]; size_t nb2 = gtype_row_bytes(t.type, cols) * R; int k2; uint64_t q2;
            uint64_t W2 = cu.load_weight(fh, (uint64_t)(t.data - g.base), nb2, t.type, R, cols, k2, q2);
            std::string ge; if (!gx().ok()) gx().init(ge);
            const int M = 512; uint64_t X = cu.alloc((size_t)M * cols * 4), Y = cu.alloc((size_t)M * R * 4);
            std::vector<float> hx((size_t)M * cols); for (auto& v : hx) { s = s * 1664525 + 1013904223; v = ((s >> 8) & 0xFFFF) / 32768.f - 1.f; }
            cu.htod(X, hx.data(), hx.size() * 4);
            if (W2 && X && Y && gx().can_mmq(k2) && gx().linear_quant(X, M, cols, W2, W2 + q2, k2, R, Y)) {
                cu.sync(); const int it = 20; double t0 = now_sec();
                for (int i = 0; i < it; i++) gx().linear_quant(X, M, cols, W2, W2 + q2, k2, R, Y, true);
                cu.sync(); double us = (now_sec() - t0) / it * 1e6;
                printf("      prompt 512 x %d x %d: %8.1f us  %.1f TOPS\n", R, cols, us, 2.0 * M * R * cols / (us * 1e6));
            } else printf("      prompt: no int8 tensor-core path for this type\n");
            cu.free(X); cu.free(Y); cu.free(W2);
        }
        if (getenv("ITHACA_MMBENCH")) {   // full-size tensor, time n = 1..8 tokens
            int R = (int)t.ne[1]; size_t nb2 = gtype_row_bytes(t.type, cols) * R; int k2; uint64_t q2;
            uint64_t W2 = cu.load_weight(fh, (uint64_t)(t.data - g.base), nb2, t.type, R, cols, k2, q2);
            uint64_t X = cu.alloc((size_t)8 * cols * 4), Y = cu.alloc((size_t)8 * R * 4);
            for (int n : {1, 2, 4, 6, 8}) {
                cu.matmul_dev(W2, q2, k2, X, n, cols, R, Y); cu.sync();
                double t0 = now_sec();
                for (int it = 0; it < 50; it++) cu.matmul_dev(W2, q2, k2, X, n, cols, R, Y);
                cu.sync();
                double us = (now_sec() - t0) / 50 * 1e6;
                printf("      n=%d  %7.1f us  (%.0f GB/s of weights)\n", n, us, CudaTier::gpu_bytes(t.type, R, cols) / (us * 1e3));
            }
            cu.free(X); cu.free(Y); cu.free(W2);
        }
        cu.free(W);
    }
    CloseHandle(fh);
    return 0;
}
