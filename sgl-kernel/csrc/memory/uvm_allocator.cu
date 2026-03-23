/**
 * uvm_allocator.cu — SGLang UVM Activation Allocator (Pool-based)
 *
 * 在进程启动时一次性分配大块统一内存（cudaMallocManaged），并立即设置：
 *   SetPreferredLocation = CPU   → 首选驻留 CPU RAM
 *   SetAccessedBy        = GPU   → 建立 GPU page table 映射，GPU 直接访问无需迁移
 *
 * 关键：advise 在任何 GPU kernel 运行前整块设置，CUDA 在首次 GPU 访问时直接走
 * 已建立的 page table 映射（远程访问 CPU），而非触发页迁移。
 * 逐 tensor 分配（cudaMallocManaged per-call）无此保证，因为 advise 和 GPU
 * kernel 发射之间窗口太短，映射来不及建立，CUDA 会 fallback 到页迁移。
 *
 * Pool 内部布局：
 *   - bump allocator 负责首次分配（线性递增 offset）
 *   - free-list 按 aligned_size 分桶（unordered_map<size_t, vector<void*>>）
 *   - uvm_free 将指针归还 free-list；下次相同大小优先复用，热路径无 CUDA API 调用
 *
 * 对外接口（extern "C"）：
 *   bool  uvm_init_pool(size_t size_bytes, int device)
 *   void* uvm_malloc   (size_t size, int device, cudaStream_t stream)
 *   void  uvm_free     (void* ptr,  size_t size, int device, cudaStream_t stream)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
//  Pool 状态（进程全局，每个 TP worker 进程各有一份）
// ──────────────────────────────────────────────────────────────────────────────

static void*  g_pool_base   = nullptr;
static size_t g_pool_size   = 0;
static int    g_pool_device = -1;

// bump allocator
static char* g_bump_ptr = nullptr;
static char* g_bump_end = nullptr;

// free-list: aligned_size → [ptr, ptr, ...]（LIFO，cache friendly）
static std::unordered_map<size_t, std::vector<void*>> g_free_list;
static std::mutex g_pool_mutex;

static constexpr size_t ALIGN_BYTES = 256;

static inline size_t align_up(size_t n) {
    return (n + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1);
}

// ──────────────────────────────────────────────────────────────────────────────
//  对外接口
// ──────────────────────────────────────────────────────────────────────────────

extern "C" {

/**
 * uvm_init_pool — 预分配 UVM pool 并统一设置 advise（启动时调用一次）
 *
 * @param size_bytes  pool 字节数（通常由 --uvm-pool-size-gb 换算）
 * @param device      GPU device index
 * @return            成功返回 true，失败返回 false
 */
bool uvm_init_pool(size_t size_bytes, int device) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);

    if (g_pool_base != nullptr) {
        fprintf(stderr, "[SGLang UVM] Pool already initialized (%.2f GB on device %d)\n",
                (double)g_pool_size / (1024.0 * 1024.0 * 1024.0), g_pool_device);
        return false;
    }

    // 一次性分配整块 UVM 内存
    cudaError_t err = cudaMallocManaged(&g_pool_base, size_bytes, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SGLang UVM] cudaMallocManaged(%.2f GB) failed: %s\n",
                (double)size_bytes / (1024.0 * 1024.0 * 1024.0),
                cudaGetErrorString(err));
        g_pool_base = nullptr;
        return false;
    }

    // 整块统一 advise：PreferredLocation=CPU + AccessedBy=GPU
    // 在任何 GPU kernel 运行前完成，确保 GPU page table 映射已建立，
    // GPU 首次访问走远程映射而非触发页迁移
    err = cudaMemAdvise(g_pool_base, size_bytes,
                        cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (err != cudaSuccess)
        fprintf(stderr, "[SGLang UVM] cudaMemAdvise(PreferredLocation=CPU): %s\n",
                cudaGetErrorString(err));

    err = cudaMemAdvise(g_pool_base, size_bytes,
                        cudaMemAdviseSetAccessedBy, device);
    if (err != cudaSuccess)
        fprintf(stderr, "[SGLang UVM] cudaMemAdvise(AccessedBy=GPU%d): %s\n",
                device, cudaGetErrorString(err));

    g_pool_size   = size_bytes;
    g_pool_device = device;
    g_bump_ptr    = static_cast<char*>(g_pool_base);
    g_bump_end    = g_bump_ptr + size_bytes;

    fprintf(stderr, "[SGLang UVM] Pool initialized: %.2f GB on device %d"
                    " (PreferredLocation=CPU, AccessedBy=GPU)\n",
            (double)size_bytes / (1024.0 * 1024.0 * 1024.0), device);
    return true;
}

/**
 * uvm_malloc — 从 pool 子分配（优先 free-list，其次 bump）
 * 当 pool 耗尽时 fallback 到 cudaMalloc（GPU 显存），保证分配不失败。
 * uvm_free 根据指针是否在 pool 地址范围内决定走 free-list 还是 cudaFree。
 */
void* uvm_malloc(size_t size, int device, cudaStream_t stream) {
    if (size == 0) return nullptr;

    // pool 未初始化时直接走 cudaMalloc
    if (g_pool_base == nullptr) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[SGLang UVM] cudaMalloc fallback(%zu B) failed: %s\n",
                    size, cudaGetErrorString(err));
            return nullptr;
        }
        return ptr;
    }

    const size_t aligned = align_up(size);

    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);

        // 优先从 free-list 复用相同大小的块
        auto it = g_free_list.find(aligned);
        if (it != g_free_list.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }

        // bump 分配
        if (g_bump_ptr + aligned <= g_bump_end) {
            void* ptr = g_bump_ptr;
            g_bump_ptr += aligned;
            return ptr;
        }

        // pool 已满，记录日志后走 cudaMalloc fallback
        size_t used = g_bump_ptr - static_cast<char*>(g_pool_base);
        fprintf(stderr,
                "[SGLang UVM] Pool OOM: requested %zu B (aligned %zu B),"
                " used %zu / %zu B — falling back to cudaMalloc\n",
                size, aligned, used, g_pool_size);
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[SGLang UVM] cudaMalloc fallback(%zu B) failed: %s\n",
                size, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

/**
 * uvm_free — 若指针在 pool 范围内归还 free-list，否则 cudaFree（不释放 UVM 内存）
 */
void uvm_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr == nullptr) return;

    // 判断指针是否属于 UVM pool 地址范围
    bool in_pool = (g_pool_base != nullptr) &&
                   (ptr >= g_pool_base) &&
                   (ptr < static_cast<char*>(g_pool_base) + g_pool_size);

    if (in_pool) {
        const size_t aligned = align_up(size);
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        g_free_list[aligned].push_back(ptr);
    } else {
        cudaFree(ptr);
    }
}

} // extern "C"
