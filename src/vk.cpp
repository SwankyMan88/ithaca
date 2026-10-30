#include "vk.h"
#include "common.h"
#include <cmath>
#include <cstring>
#include <map>
#include <random>

// ---- the small part of the Vulkan API used here, declared by hand so no SDK is needed ----
typedef void* VkH;          // dispatchable handles
typedef uint64_t VkN;       // non-dispatchable handles
typedef void (*VkFn)();
struct VkAppInfo { int sType; const void* next; const char* app; uint32_t appV; const char* eng; uint32_t engV, api; };
struct VkInstInfo { int sType; const void* next; uint32_t flags; const VkAppInfo* app; uint32_t nl; const char* const* l; uint32_t ne; const char* const* e; };
struct VkQueueProps { uint32_t flags, count, ts, w, h, d; };
struct VkQueueInfo { int sType; const void* next; uint32_t flags, family, count; const float* prio; };
struct VkDevInfo { int sType; const void* next; uint32_t flags, nq; const VkQueueInfo* q; uint32_t nl; const char* const* l; uint32_t ne; const char* const* e; const void* feat; };
struct VkMemProps { uint32_t ntypes; struct { uint32_t flags, heap; } types[32]; uint32_t nheaps; struct { uint64_t size; uint32_t flags; } heaps[16]; };
struct VkBufInfo { int sType; const void* next; uint32_t flags; uint64_t size; uint32_t usage; int share; uint32_t nf; const uint32_t* f; };
struct VkMemReq { uint64_t size, align; uint32_t bits; };
struct VkAllocInfo { int sType; const void* next; uint64_t size; uint32_t type; };
struct VkModInfo { int sType; const void* next; uint32_t flags; size_t bytes; const uint32_t* code; };
struct VkBinding { uint32_t binding; int type; uint32_t count, stages; const void* samplers; };
struct VkDslInfo { int sType; const void* next; uint32_t flags, n; const VkBinding* b; };
struct VkPushRange { uint32_t stages, offset, size; };
struct VkLayoutInfo { int sType; const void* next; uint32_t flags, nsets; const VkN* sets; uint32_t npush; const VkPushRange* push; };
struct VkStage { int sType; const void* next; uint32_t flags, stage; VkN mod; const char* name; const void* spec; };
struct VkCompInfo { int sType; const void* next; uint32_t flags; VkStage stage; VkN layout, base; int32_t baseIdx; };
struct VkPoolSize { int type; uint32_t count; };
struct VkPoolInfo { int sType; const void* next; uint32_t flags, maxSets, n; const VkPoolSize* s; };
struct VkSetAlloc { int sType; const void* next; VkN pool; uint32_t n; const VkN* layouts; };
struct VkBufDesc { VkN buf; uint64_t off, range; };
struct VkWrite { int sType; const void* next; VkN set; uint32_t binding, elem, n; int type; const void* img; const VkBufDesc* buf; const void* view; };
struct VkCmdPoolInfo { int sType; const void* next; uint32_t flags, family; };
struct VkCmdAlloc { int sType; const void* next; VkN pool; int level; uint32_t n; };
struct VkBegin { int sType; const void* next; uint32_t flags; const void* inh; };
struct VkSubmit { int sType; const void* next; uint32_t nw; const VkN* w; const uint32_t* wm; uint32_t nc; const VkH* c; uint32_t ns; const VkN* s; };
struct VkFenceInfo { int sType; const void* next; uint32_t flags; };
struct VkBarrier { int sType; const void* next; uint32_t src, dst; };
struct VkCopy { uint64_t src, dst, size; };

struct Pipe { VkN dsl, layout, pipe, pool; int bindings, push; };
struct Vk::Impl {
    HMODULE lib = nullptr;
    VkH inst = nullptr, phys = nullptr, dev = nullptr, queue = nullptr, cmd = nullptr;
    VkN cpool = 0, fence = 0;
    VkMemProps mem{};
    std::map<VkN, VkN> mems;   // buffer -> memory
    std::vector<Pipe> pipes;
#define VKF(n) VkFn n = nullptr;
    VKF(vkCreateDevice) VKF(vkGetDeviceQueue) VKF(vkCreateBuffer) VKF(vkGetBufferMemoryRequirements) VKF(vkAllocateMemory) VKF(vkBindBufferMemory)
    VKF(vkMapMemory) VKF(vkUnmapMemory) VKF(vkDestroyBuffer) VKF(vkFreeMemory) VKF(vkCreateShaderModule) VKF(vkDestroyShaderModule) VKF(vkCreateDescriptorSetLayout)
    VKF(vkCreatePipelineLayout) VKF(vkCreateComputePipelines) VKF(vkCreateDescriptorPool) VKF(vkResetDescriptorPool) VKF(vkAllocateDescriptorSets) VKF(vkUpdateDescriptorSets)
    VKF(vkCreateCommandPool) VKF(vkAllocateCommandBuffers) VKF(vkBeginCommandBuffer) VKF(vkEndCommandBuffer) VKF(vkResetCommandBuffer) VKF(vkCmdBindPipeline)
    VKF(vkCmdBindDescriptorSets) VKF(vkCmdPushConstants) VKF(vkCmdDispatch) VKF(vkCmdPipelineBarrier) VKF(vkCmdCopyBuffer) VKF(vkQueueSubmit) VKF(vkCreateFence)
    VKF(vkWaitForFences) VKF(vkResetFences) VKF(vkDeviceWaitIdle) VKF(vkDestroyDevice) VKF(vkDestroyInstance)
#undef VKF
    VkN make_buf(size_t bytes, uint32_t usage, uint32_t want, uint32_t avoid = 0) {
        VkBufInfo bi{12, nullptr, 0, bytes, usage, 0, 0, nullptr};
        VkN b = 0;
        if (((int(*)(VkH, const VkBufInfo*, const void*, VkN*))vkCreateBuffer)(dev, &bi, nullptr, &b)) return 0;
        VkMemReq r{};
        ((void(*)(VkH, VkN, VkMemReq*))vkGetBufferMemoryRequirements)(dev, b, &r);
        int type = -1;
        for (uint32_t i = 0; i < mem.ntypes && type < 0; i++) if ((r.bits >> i & 1) && (mem.types[i].flags & want) == want && !(mem.types[i].flags & avoid)) type = (int)i;
        for (uint32_t i = 0; i < mem.ntypes && type < 0; i++) if ((r.bits >> i & 1) && (mem.types[i].flags & want) == want) type = (int)i;
        VkN m = 0;
        VkAllocInfo ai{5, nullptr, r.size, (uint32_t)type};
        if (type < 0 || ((int(*)(VkH, const VkAllocInfo*, const void*, VkN*))vkAllocateMemory)(dev, &ai, nullptr, &m) ||
            ((int(*)(VkH, VkN, VkN, uint64_t))vkBindBufferMemory)(dev, b, m, 0)) { ((void(*)(VkH, VkN, const void*))vkDestroyBuffer)(dev, b, nullptr); return 0; }
        mems[b] = m;
        return b;
    }
    void drop(VkN b) {
        auto it = mems.find(b); if (it == mems.end()) return;
        ((void(*)(VkH, VkN, const void*))vkDestroyBuffer)(dev, b, nullptr);
        ((void(*)(VkH, VkN, const void*))vkFreeMemory)(dev, it->second, nullptr);
        mems.erase(it);
    }
    void begin() { VkBegin b{42, nullptr, 1, nullptr}; ((int(*)(VkH, uint32_t))vkResetCommandBuffer)(cmd, 0); ((int(*)(VkH, const VkBegin*))vkBeginCommandBuffer)(cmd, &b); }
    bool submit() {
        ((int(*)(VkH))vkEndCommandBuffer)(cmd);
        VkSubmit s{4, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
        ((int(*)(VkH, uint32_t, const VkN*))vkResetFences)(dev, 1, &fence);
        if (((int(*)(VkH, uint32_t, const VkSubmit*, VkN))vkQueueSubmit)(queue, 1, &s, fence)) return false;
        return ((int(*)(VkH, uint32_t, const VkN*, uint32_t, uint64_t))vkWaitForFences)(dev, 1, &fence, 1, ~0ull) == 0;
    }
    void barrier() {
        VkBarrier b{46, nullptr, 0x40 | 0x1000, 0x20 | 0x40 | 0x800 | 0x1000};
        ((void(*)(VkH, uint32_t, uint32_t, uint32_t, uint32_t, const VkBarrier*, uint32_t, const void*, uint32_t, const void*))vkCmdPipelineBarrier)(cmd, 0x800 | 0x1000, 0x800 | 0x1000, 0, 1, &b, 0, nullptr, 0, nullptr);
    }
    bool copy(VkN src, VkN dst, size_t bytes) {
        begin(); VkCopy c{0, 0, bytes};
        ((void(*)(VkH, VkN, VkN, uint32_t, const VkCopy*))vkCmdCopyBuffer)(cmd, src, dst, 1, &c);
        barrier();
        return submit();
    }
};

bool Vk::init(std::string& info) {
    Impl* d = new Impl; p_ = d;
    d->lib = LoadLibraryA("vulkan-1.dll");
    if (!d->lib) { info = "no Vulkan driver"; return false; }
    auto gip = (VkFn(*)(VkH, const char*))GetProcAddress(d->lib, "vkGetInstanceProcAddr");
    if (!gip) { info = "bad Vulkan loader"; return false; }
    VkAppInfo app{0, nullptr, "Ithaca", 1, "Ithaca", 1, (1u << 22) | (1u << 12)};   // Vulkan 1.1
    VkInstInfo ii{1, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    if (((int(*)(const VkInstInfo*, const void*, VkH*))gip(nullptr, "vkCreateInstance"))(&ii, nullptr, &d->inst)) { info = "Vulkan 1.1 is not available"; return false; }
    auto ip = [&](const char* n) { return gip(d->inst, n); };
    uint32_t n = 0;
    auto enumd = (int(*)(VkH, uint32_t*, VkH*))ip("vkEnumeratePhysicalDevices");
    enumd(d->inst, &n, nullptr);
    if (!n) { info = "no Vulkan GPU"; return false; }
    std::vector<VkH> devs(n); enumd(d->inst, &n, devs.data());
    auto props = (void(*)(VkH, void*))ip("vkGetPhysicalDeviceProperties");
    std::string name; int best = -1;   // discrete beats integrated beats anything else
    for (auto h : devs) {
        char pb[2048]{}; props(h, pb);
        int type = *(int*)(pb + 16), score = type == 2 ? 3 : type == 1 ? 2 : type == 3 ? 1 : 0;
        if (score > best) { best = score; d->phys = h; name = pb + 20; }
    }
    ((void(*)(VkH, VkMemProps*))ip("vkGetPhysicalDeviceMemoryProperties"))(d->phys, &d->mem);
    uint32_t nq = 0;
    auto qp = (void(*)(VkH, uint32_t*, VkQueueProps*))ip("vkGetPhysicalDeviceQueueFamilyProperties");
    qp(d->phys, &nq, nullptr);
    std::vector<VkQueueProps> qs(nq); qp(d->phys, &nq, qs.data());
    int fam = -1;
    for (uint32_t i = 0; i < nq; i++) if ((qs[i].flags & 2) && (fam < 0 || !(qs[i].flags & 1))) fam = (int)i;   // prefer a compute-only queue
    if (fam < 0) { info = "no compute queue"; return false; }
    float prio = 1.f;
    VkQueueInfo qi{2, nullptr, 0, (uint32_t)fam, 1, &prio};
    VkDevInfo di{3, nullptr, 0, 1, &qi, 0, nullptr, 0, nullptr, nullptr};
    if (((int(*)(VkH, const VkDevInfo*, const void*, VkH*))ip("vkCreateDevice"))(d->phys, &di, nullptr, &d->dev)) { info = "could not open the GPU"; return false; }
    auto dp = (VkFn(*)(VkH, const char*))ip("vkGetDeviceProcAddr");
#define VKL(x) d->x = dp(d->dev, #x);
    VKL(vkGetDeviceQueue) VKL(vkCreateBuffer) VKL(vkGetBufferMemoryRequirements) VKL(vkAllocateMemory) VKL(vkBindBufferMemory) VKL(vkMapMemory) VKL(vkUnmapMemory)
    VKL(vkDestroyBuffer) VKL(vkFreeMemory) VKL(vkCreateShaderModule) VKL(vkDestroyShaderModule) VKL(vkCreateDescriptorSetLayout) VKL(vkCreatePipelineLayout) VKL(vkCreateComputePipelines)
    VKL(vkCreateDescriptorPool) VKL(vkResetDescriptorPool) VKL(vkAllocateDescriptorSets) VKL(vkUpdateDescriptorSets) VKL(vkCreateCommandPool) VKL(vkAllocateCommandBuffers)
    VKL(vkBeginCommandBuffer) VKL(vkEndCommandBuffer) VKL(vkResetCommandBuffer) VKL(vkCmdBindPipeline) VKL(vkCmdBindDescriptorSets) VKL(vkCmdPushConstants) VKL(vkCmdDispatch)
    VKL(vkCmdPipelineBarrier) VKL(vkCmdCopyBuffer) VKL(vkQueueSubmit) VKL(vkCreateFence) VKL(vkWaitForFences) VKL(vkResetFences) VKL(vkDeviceWaitIdle) VKL(vkDestroyDevice)
#undef VKL
    d->vkDestroyInstance = ip("vkDestroyInstance");
    ((void(*)(VkH, uint32_t, uint32_t, VkH*))d->vkGetDeviceQueue)(d->dev, fam, 0, &d->queue);
    VkCmdPoolInfo cp{39, nullptr, 2, (uint32_t)fam};
    ((int(*)(VkH, const VkCmdPoolInfo*, const void*, VkN*))d->vkCreateCommandPool)(d->dev, &cp, nullptr, &d->cpool);
    VkCmdAlloc ca{40, nullptr, d->cpool, 0, 1};
    ((int(*)(VkH, const VkCmdAlloc*, VkH*))d->vkAllocateCommandBuffers)(d->dev, &ca, &d->cmd);
    VkFenceInfo fi{8, nullptr, 0};
    ((int(*)(VkH, const VkFenceInfo*, const void*, VkN*))d->vkCreateFence)(d->dev, &fi, nullptr, &d->fence);
    info = name;
    return true;
}
uint64_t Vk::alloc(size_t bytes) { return p_->make_buf(bytes, 0x20 | 1 | 2, 1); }
void Vk::free(uint64_t b) { p_->drop(b); }
bool Vk::upload(uint64_t buf, const void* data, size_t bytes) {
    VkN st = p_->make_buf(bytes, 1, 2 | 4); if (!st) return false;
    void* m = nullptr;
    ((int(*)(VkH, VkN, uint64_t, uint64_t, uint32_t, void**))p_->vkMapMemory)(p_->dev, p_->mems[st], 0, ~0ull, 0, &m);
    memcpy(m, data, bytes);
    ((void(*)(VkH, VkN))p_->vkUnmapMemory)(p_->dev, p_->mems[st]);
    bool ok = p_->copy(st, buf, bytes);
    p_->drop(st);
    return ok;
}
bool Vk::download(void* data, uint64_t buf, size_t bytes) {
    VkN st = p_->make_buf(bytes, 2, 2 | 4); if (!st) return false;
    bool ok = p_->copy(buf, st, bytes);
    void* m = nullptr;
    ((int(*)(VkH, VkN, uint64_t, uint64_t, uint32_t, void**))p_->vkMapMemory)(p_->dev, p_->mems[st], 0, ~0ull, 0, &m);
    memcpy(data, m, bytes);
    ((void(*)(VkH, VkN))p_->vkUnmapMemory)(p_->dev, p_->mems[st]);
    p_->drop(st);
    return ok;
}
uint64_t Vk::pipeline(const std::vector<uint32_t>& code, int bindings, int push) {
    Impl* d = p_;
    Pipe x{}; x.bindings = bindings; x.push = push;
    std::vector<VkBinding> bs;
    for (int i = 0; i < bindings; i++) bs.push_back({(uint32_t)i, 7, 1, 0x20, nullptr});
    VkDslInfo dl{32, nullptr, 0, (uint32_t)bindings, bs.data()};
    if (((int(*)(VkH, const VkDslInfo*, const void*, VkN*))d->vkCreateDescriptorSetLayout)(d->dev, &dl, nullptr, &x.dsl)) return 0;
    VkPushRange pr{0x20, 0, (uint32_t)push};
    VkLayoutInfo li{30, nullptr, 0, 1, &x.dsl, push ? 1u : 0u, &pr};
    if (((int(*)(VkH, const VkLayoutInfo*, const void*, VkN*))d->vkCreatePipelineLayout)(d->dev, &li, nullptr, &x.layout)) return 0;
    VkModInfo mi{16, nullptr, 0, code.size() * 4, code.data()};
    VkN mod = 0;
    if (((int(*)(VkH, const VkModInfo*, const void*, VkN*))d->vkCreateShaderModule)(d->dev, &mi, nullptr, &mod)) return 0;
    VkCompInfo ci{29, nullptr, 0, {18, nullptr, 0, 0x20, mod, "main", nullptr}, x.layout, 0, -1};
    int r = ((int(*)(VkH, VkN, uint32_t, const VkCompInfo*, const void*, VkN*))d->vkCreateComputePipelines)(d->dev, 0, 1, &ci, nullptr, &x.pipe);
    ((void(*)(VkH, VkN, const void*))d->vkDestroyShaderModule)(d->dev, mod, nullptr);
    if (r) return 0;
    VkPoolSize ps{7, (uint32_t)bindings};
    VkPoolInfo po{33, nullptr, 0, 1, 1, &ps};
    ((int(*)(VkH, const VkPoolInfo*, const void*, VkN*))d->vkCreateDescriptorPool)(d->dev, &po, nullptr, &x.pool);
    d->pipes.push_back(x);
    return d->pipes.size();   // handle = index + 1
}
bool Vk::dispatch(uint64_t pipe, const std::vector<uint64_t>& bufs, const void* push, unsigned gx) {
    Impl* d = p_;
    if (!pipe || pipe > d->pipes.size()) return false;
    Pipe& x = d->pipes[pipe - 1];
    ((int(*)(VkH, VkN, uint32_t))d->vkResetDescriptorPool)(d->dev, x.pool, 0);
    VkSetAlloc sa{34, nullptr, x.pool, 1, &x.dsl};
    VkN set = 0;
    if (((int(*)(VkH, const VkSetAlloc*, VkN*))d->vkAllocateDescriptorSets)(d->dev, &sa, &set)) return false;
    std::vector<VkBufDesc> bi; std::vector<VkWrite> w;
    bi.reserve(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) { bi.push_back({bufs[i], 0, ~0ull}); w.push_back({35, nullptr, set, (uint32_t)i, 0, 1, 7, nullptr, &bi[i], nullptr}); }
    ((void(*)(VkH, uint32_t, const VkWrite*, uint32_t, const void*))d->vkUpdateDescriptorSets)(d->dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    d->begin();
    ((void(*)(VkH, int, VkN))d->vkCmdBindPipeline)(d->cmd, 1, x.pipe);
    ((void(*)(VkH, int, VkN, uint32_t, uint32_t, const VkN*, uint32_t, const uint32_t*))d->vkCmdBindDescriptorSets)(d->cmd, 1, x.layout, 0, 1, &set, 0, nullptr);
    if (x.push) ((void(*)(VkH, VkN, uint32_t, uint32_t, uint32_t, const void*))d->vkCmdPushConstants)(d->cmd, x.layout, 0x20, 0, x.push, push);
    ((void(*)(VkH, uint32_t, uint32_t, uint32_t))d->vkCmdDispatch)(d->cmd, gx, 1, 1);
    d->barrier();
    return d->submit();
}
void Vk::shutdown() {
    if (!p_) return;
    if (p_->dev) {
        ((int(*)(VkH))p_->vkDeviceWaitIdle)(p_->dev);
        while (!p_->mems.empty()) p_->drop(p_->mems.begin()->first);
        ((void(*)(VkH, const void*))p_->vkDestroyDevice)(p_->dev, nullptr);
    }
    if (p_->inst && p_->vkDestroyInstance) ((void(*)(VkH, const void*))p_->vkDestroyInstance)(p_->inst, nullptr);
    if (p_->lib) FreeLibrary(p_->lib);
    delete p_; p_ = nullptr;
}

// ---- a small SPIR-V writer: one compute entry point, storage buffers of u32/f32, push constants of 32-bit values ----
namespace {
struct Sp {
    std::vector<uint32_t> head, deco, types, body;
    uint32_t next = 1;
    uint32_t tVoid, tBool, tU, tF, tV3, tPIn, tPU32F, tPU32FV, tPPush, tFn, gid;
    std::vector<uint32_t> ifaces;
    static void op(std::vector<uint32_t>& v, uint32_t code, std::initializer_list<uint32_t> a) { v.push_back((uint32_t)(a.size() + 1) << 16 | code); v.insert(v.end(), a); }
    uint32_t id() { return next++; }
    std::map<uint32_t, uint32_t> ku; std::map<uint32_t, uint32_t> kf;
    uint32_t cu(uint32_t x) { auto it = ku.find(x); if (it != ku.end()) return it->second; uint32_t i = id(); op(types, 43, {tU, i, x}); return ku[x] = i; }
    uint32_t cf(float x) { uint32_t b; memcpy(&b, &x, 4); auto it = kf.find(b); if (it != kf.end()) return it->second; uint32_t i = id(); op(types, 43, {tF, i, b}); return kf[b] = i; }
    uint32_t tPu, tPf, tPuE, tPfE, tPfV, tPuV;
    explicit Sp(unsigned local) {
        tVoid = id(); tBool = id(); tU = id(); tF = id(); tV3 = id(); tPIn = id(); tFn = id();
        op(types, 19, {tVoid}); op(types, 20, {tBool}); op(types, 21, {tU, 32, 0}); op(types, 22, {tF, 32});
        op(types, 23, {tV3, tU, 3}); op(types, 32, {tPIn, 1, tV3}); op(types, 33, {tFn, tVoid});
        tPuE = id(); tPfE = id(); tPuV = id(); tPfV = id();
        op(types, 32, {tPuE, 2, tU}); op(types, 32, {tPfE, 2, tF}); op(types, 32, {tPuV, 7, tU}); op(types, 32, {tPfV, 7, tF});
        gid = id(); op(types, 59, {tPIn, gid, 1});
        op(deco, 71, {gid, 11, 28});
        ifaces.push_back(gid);
        local_ = local;
    }
    unsigned local_;
    // a storage buffer (runtime array) at binding b
    uint32_t buffer(int b, bool is_float) {
        uint32_t arr = id(), st = id(), pt = id(), var = id();
        op(types, 29, {arr, is_float ? tF : tU}); op(types, 30, {st, arr}); op(types, 32, {pt, 2, st}); op(types, 59, {pt, var, 2});
        op(deco, 71, {arr, 6, 4}); op(deco, 71, {st, 3}); op(deco, 72, {st, 0, 35, 0});
        op(deco, 71, {var, 34, 0}); op(deco, 71, {var, 33, (uint32_t)b});
        return var;
    }
    // push constants: n 32-bit members with the given float flags
    uint32_t tPushU = 0, tPushF = 0, push = 0;
    void push_block(std::initializer_list<bool> floats) {
        uint32_t st = id(), pt = id(); push = id(); tPushU = id(); tPushF = id();
        std::vector<uint32_t> m = {st}; for (bool f : floats) m.push_back(f ? tF : tU);
        types.push_back((uint32_t)(m.size() + 1) << 16 | 30); types.insert(types.end(), m.begin(), m.end());
        op(types, 32, {pt, 9, st}); op(types, 59, {pt, push, 9}); op(types, 32, {tPushU, 9, tU}); op(types, 32, {tPushF, 9, tF});
        op(deco, 71, {st, 2});
        uint32_t i = 0; for (bool f : floats) { (void)f; op(deco, 72, {st, i, 35, i * 4}); i++; }
    }
    uint32_t r(uint32_t code, uint32_t type, std::initializer_list<uint32_t> a) { uint32_t i = id(); std::vector<uint32_t> w = {type, i}; w.insert(w.end(), a); body.push_back((uint32_t)(w.size() + 1) << 16 | code); body.insert(body.end(), w.begin(), w.end()); return i; }
    void o(uint32_t code, std::initializer_list<uint32_t> a) { op(body, code, a); }
    uint32_t pushu(uint32_t m) { return r(61, tU, {r(65, tPushU, {push, cu(m)})}); }
    uint32_t pushf(uint32_t m) { return r(61, tF, {r(65, tPushF, {push, cu(m)})}); }
    uint32_t loadf(uint32_t buf, uint32_t idx) { return r(61, tF, {r(65, tPfE, {buf, cu(0), idx})}); }
    uint32_t loadu(uint32_t buf, uint32_t idx) { return r(61, tU, {r(65, tPuE, {buf, cu(0), idx})}); }
    void storef(uint32_t buf, uint32_t idx, uint32_t v) { o(62, {r(65, tPfE, {buf, cu(0), idx}), v}); }
    uint32_t add(uint32_t a, uint32_t b) { return r(128, tU, {a, b}); }
    uint32_t mul(uint32_t a, uint32_t b) { return r(132, tU, {a, b}); }
    uint32_t shr(uint32_t a, uint32_t b) { return r(194, tU, {a, b}); }
    uint32_t band(uint32_t a, uint32_t b) { return r(199, tU, {a, b}); }
    uint32_t fadd(uint32_t a, uint32_t b) { return r(129, tF, {a, b}); }
    uint32_t fmul(uint32_t a, uint32_t b) { return r(133, tF, {a, b}); }
    uint32_t ult(uint32_t a, uint32_t b) { return r(176, tBool, {a, b}); }
    // for (var = 0; var < n; var++) body()
    template <class F> void loop(uint32_t var, uint32_t n, F inner) {
        uint32_t hdr = id(), chk = id(), bod = id(), cont = id(), end = id();
        o(62, {var, cu(0)}); o(249, {hdr}); o(248, {hdr}); o(246, {end, cont, 0}); o(249, {chk}); o(248, {chk});
        uint32_t c = ult(r(61, tU, {var}), n); o(250, {c, bod, end}); o(248, {bod});
        inner(); o(249, {cont}); o(248, {cont});
        o(62, {var, add(r(61, tU, {var}), cu(1))}); o(249, {hdr}); o(248, {end});
    }
    std::vector<uint32_t> finish() {
        std::vector<uint32_t> w = {0x07230203, 0x00010000, 0, next + 1, 0};
        op(w, 17, {1}); op(w, 14, {0, 1});
        std::vector<uint32_t> ep = {5, main_, 0x6E69616D, 0}; ep.insert(ep.end(), ifaces.begin(), ifaces.end());
        w.push_back((uint32_t)(ep.size() + 1) << 16 | 15); w.insert(w.end(), ep.begin(), ep.end());
        op(w, 16, {main_, 17, local_, 1, 1});
        w.insert(w.end(), deco.begin(), deco.end()); w.insert(w.end(), types.begin(), types.end()); w.insert(w.end(), body.begin(), body.end());
        w[3] = next;
        return w;
    }
    uint32_t main_ = 0;
    void begin_main() { main_ = id(); o(54, {tVoid, main_, 0, tFn}); o(248, {id()}); }
    void end_main() { o(253, {}); o(56, {}); }
    uint32_t invocation() { return r(81, tU, {r(61, tV3, {gid}), 0}); }
};
}

std::vector<uint32_t> vk_kernel_add() {
    Sp s(256);
    uint32_t y = s.buffer(0, true), x = s.buffer(1, true);
    s.push_block({false, true});
    s.begin_main();
    uint32_t i = s.invocation(), n = s.pushu(0), a = s.pushf(1), then = s.id(), merge = s.id();
    uint32_t in = s.ult(i, n); s.o(247, {merge, 0}); s.o(250, {in, then, merge}); s.o(248, {then});
    s.storef(y, i, s.fadd(s.loadf(y, i), s.fmul(a, s.loadf(x, i))));
    s.o(249, {merge}); s.o(248, {merge});
    s.end_main();
    return s.finish();
}
// Weights as the CUDA path stores Q4_0 (a f32 scale per block here; 16 bytes of nibbles: byte j holds elements j and j + 16).
std::vector<uint32_t> vk_kernel_mv_q4() {
    Sp s(64);
    uint32_t S = s.buffer(0, true), Q = s.buffer(1, false), X = s.buffer(2, true), Y = s.buffer(3, true);
    s.push_block({false, false});
    s.begin_main();
    uint32_t vb = s.r(59, s.tPuV, {7}), vacc = s.r(59, s.tPfV, {7});
    uint32_t r = s.invocation(), rows = s.pushu(0), nb = s.pushu(1), then = s.id(), merge = s.id();
    uint32_t in = s.ult(r, rows); s.o(247, {merge, 0}); s.o(250, {in, then, merge}); s.o(248, {then});
    s.o(62, {vacc, s.cf(0.f)});
    s.loop(vb, nb, [&] {
        uint32_t b = s.r(61, s.tU, {vb}), wb = s.add(s.mul(r, nb), b), xb = s.mul(b, s.cu(32));
        // unrolled: each weight word and each x are loaded once, then 8 multiply-adds per word
        uint32_t blk = s.cf(0.f), wb4 = s.mul(wb, s.cu(4));
        for (uint32_t w = 0; w < 4; w++) {
            uint32_t word = s.loadu(Q, s.add(wb4, s.cu(w)));
            for (uint32_t k = 0; k < 8; k++) {   // bytes low nibble: element 4w+k; high nibble: element 16+4w+k-4
                const uint32_t byte = k & 3, high = k >> 2, j = 4 * w + byte + 16 * high;
                uint32_t q = s.band(s.shr(word, s.cu(byte * 8 + high * 4)), s.cu(15));
                uint32_t fq = s.fadd(s.r(112, s.tF, {q}), s.cf(-8.f));
                blk = s.fadd(blk, s.fmul(fq, s.loadf(X, s.add(xb, s.cu(j)))));
            }
        }
        s.o(62, {vacc, s.fadd(s.r(61, s.tF, {vacc}), s.fmul(blk, s.loadf(S, wb)))});
    });
    s.storef(Y, r, s.r(61, s.tF, {vacc}));
    s.o(249, {merge}); s.o(248, {merge});
    s.end_main();
    return s.finish();
}

// --vktest: proves the path end to end against values computed on the CPU.
int vk_selftest() {
    Vk v; std::string info;
    if (!v.init(info)) { printf("Vulkan: %s\n", info.c_str()); return 1; }
    printf("Vulkan GPU: %s\n", info.c_str());
    std::mt19937 g(1); std::uniform_real_distribution<float> u(-1.f, 1.f);
    {
        const unsigned n = 1000000; std::vector<float> a(n), b(n), r(n);
        for (unsigned i = 0; i < n; i++) { a[i] = u(g); b[i] = u(g); }
        uint64_t A = v.alloc(n * 4), B = v.alloc(n * 4), P = v.pipeline(vk_kernel_add(), 2, 8);
        if (!A || !B || !P) { printf("add: setup failed\n"); return 1; }
        v.upload(A, a.data(), n * 4); v.upload(B, b.data(), n * 4);
        struct { uint32_t n; float k; } pc{n, 2.5f};
        bool ok = v.dispatch(P, {A, B}, &pc, (n + 255) / 256) && v.download(r.data(), A, n * 4);
        double e = 0; for (unsigned i = 0; i < n; i++) e = std::max(e, (double)std::fabs(r[i] - (a[i] + 2.5f * b[i])));
        printf("  add: %s, max error %.2e\n", ok && e < 1e-5 ? "OK" : "BAD", e);
        v.free(A); v.free(B);
    }
    const unsigned rows = 18944, nb = 112, cols = nb * 32; const size_t nblk = (size_t)rows * nb;
    std::vector<float> S(nblk), X(cols), Y(rows); std::vector<uint32_t> Q(nblk * 4);
    for (auto& x : S) x = 0.001f + std::fabs(u(g)) * 0.01f;
    for (auto& x : X) x = u(g);
    for (auto& x : Q) x = g();
    uint64_t bs = v.alloc(S.size() * 4), bq = v.alloc(Q.size() * 4), bx = v.alloc(cols * 4), by = v.alloc(rows * 4), P = v.pipeline(vk_kernel_mv_q4(), 4, 8);
    if (!bs || !bq || !bx || !by || !P) { printf("mv: setup failed\n"); return 1; }
    v.upload(bs, S.data(), S.size() * 4); v.upload(bq, Q.data(), Q.size() * 4); v.upload(bx, X.data(), cols * 4);
    uint32_t pc[2] = {rows, nb};
    double t0 = now_sec(); bool ok = true; const int it = 20;
    for (int i = 0; i < it; i++) ok = ok && v.dispatch(P, {bs, bq, bx, by}, pc, (rows + 63) / 64);
    double sec = (now_sec() - t0) / it;
    {   // the cost of one submit-and-wait with almost no work: real use must batch dispatches into one command buffer
        uint32_t one[2] = {1, nb}; double t1 = now_sec();
        for (int i = 0; i < it; i++) v.dispatch(P, {bs, bq, bx, by}, one, 1);
        printf("  submit and wait: %.0f us per dispatch\n", (now_sec() - t1) / it * 1e6);
    }
    ok = ok && v.download(Y.data(), by, rows * 4);
    double num = 0, den = 0;
    for (unsigned r = 0; r < rows; r++) {
        double acc = 0;
        for (unsigned b = 0; b < nb; b++) {
            const uint8_t* q = (const uint8_t*)&Q[((size_t)r * nb + b) * 4]; double t = 0;
            for (int j = 0; j < 16; j++) t += ((q[j] & 15) - 8) * (double)X[b * 32 + j] + ((q[j] >> 4) - 8) * (double)X[b * 32 + 16 + j];
            acc += t * S[(size_t)r * nb + b];
        }
        num += (Y[r] - acc) * (Y[r] - acc); den += acc * acc;
    }
    double err = std::sqrt(num / std::max(den, 1e-30));
    printf("  Q4_0 matvec (%u x %u): %s, rel.err %.2e, %.1f ms, %.0f GB/s of weights\n", rows, cols, ok && err < 1e-3 ? "OK" : "BAD", err, sec * 1e3, nblk * 20 / sec / 1e9);
    return ok && err < 1e-3 ? 0 : 1;
}
