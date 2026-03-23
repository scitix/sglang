"""
uvm_patch.py — UVM activation pool backed by PyTorch MemPool.

原理：
  1. 通过 libuvm_allocator.so 初始化 UVM pool（一次性 cudaMallocManaged，
     整块 advise: SetPreferredLocation=CPU + SetAccessedBy=GPU）。
  2. 用 torch.cuda.memory.CUDAPluggableAllocator 将 uvm_malloc / uvm_free
     包装为 PyTorch 的 CUDAAllocator 接口。
  3. 用 torch.cuda.MemPool(allocator=...) 创建与 NativeCachingAllocator 集成
     的 MemPool：
       - 在 use_mem_pool(pool) 上下文中分配时，NativeCachingAllocator 通过
         raw_alloc 调用 uvm_malloc，从 UVM pool 获取内存段；
       - 段释放时调用 raw_delete → uvm_free → C 层 free-list（不真正 cudaFree）；
       - NativeCachingAllocator 的缓存层减少 uvm_malloc 调用，热路径无额外
         CUDA API 开销。
  4. activation 的 forward_cuda 通过 with torch.cuda.use_mem_pool(get_pool()):
     将输出 tensor 路由到 UVM pool。

生命周期安全：
  PrivatePool（NativeCachingAllocator 内部）存储 CUDAAllocator 裸指针。
  ~MemPool() → emptyCache → raw_delete 需要通过裸指针访问 allocation_metadata_。
  若 Python CUDAPluggableAllocator 对象（持有 shared_ptr）被 GC，
  裸指针悬空 → segfault。

  解决方案：用 ctypes.pythonapi.Py_IncRef 将 _uvm_allocator._allocator
  的引用计数永久 +1，使 C++ shared_ptr 永不释放（进程生命周期内有效）。
  进程退出时 OS 回收全部内存，无实际泄漏。

使用方法：
    import sglang.srt.utils.uvm_patch as uvm_patch
    uvm_patch.enable(pool_size_gb=4.0, device=0)  # 启动时调用一次

    pool = uvm_patch.get_pool()
    if pool is not None:
        with torch.cuda.use_mem_pool(pool):
            out = torch.empty(output_shape, dtype=dtype, device=device)
"""

import ctypes
import os

import torch
from torch.cuda.memory import CUDAPluggableAllocator

# ──────────────────────────────────────────────────────────────────────────────
#  libuvm_allocator.so 路径
# ──────────────────────────────────────────────────────────────────────────────

_SO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libuvm_allocator.so"
)

_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(_SO_PATH):
            raise FileNotFoundError(
                f"UVM allocator plugin not found: {_SO_PATH}\n"
                "Rebuild sgl-kernel to generate libuvm_allocator.so."
            )
        _lib = ctypes.CDLL(_SO_PATH)
        _lib.uvm_init_pool.restype = ctypes.c_bool
        _lib.uvm_init_pool.argtypes = [ctypes.c_size_t, ctypes.c_int]
    return _lib


# ──────────────────────────────────────────────────────────────────────────────
#  全局 MemPool 和 Allocator（每 TP-worker 进程各一份）
# ──────────────────────────────────────────────────────────────────────────────

_uvm_allocator: CUDAPluggableAllocator | None = None
_uvm_pool: torch.cuda.MemPool | None = None


def enable(pool_size_gb: float = 4.0, device: int | None = None) -> None:
    """初始化 UVM pool 并创建 PyTorch MemPool。

    必须在任何 CUDA kernel 运行前调用（KV-cache 初始化之后、首次 forward 之前）。

    Args:
        pool_size_gb: UVM pool 总大小（GB）。所有 activation 分配从此 pool 子分配。
        device:       GPU device index；None 时取 torch.cuda.current_device()。
    """
    global _uvm_pool, _uvm_allocator
    if _uvm_pool is not None:
        return  # 已初始化，幂等

    lib = _get_lib()
    if device is None:
        device = torch.cuda.current_device()
    size_bytes = int(pool_size_gb * 1024**3)

    # 1. 在 C 层一次性分配 UVM pool 并整块 advise（PreferredLocation=CPU, AccessedBy=GPU）
    ok = lib.uvm_init_pool(size_bytes, device)
    if not ok:
        raise RuntimeError(
            f"uvm_init_pool failed: {pool_size_gb:.1f} GB on device {device}. "
            "Pool may already be initialized or GPU OOM."
        )

    # 2. 将 uvm_malloc / uvm_free 包装为 PyTorch CUDAAllocator 接口。
    #    NativeCachingAllocator 调用 raw_alloc(size) → uvm_malloc(size, dev, stream)
    #    获取 UVM 段；调用 raw_delete(ptr) → uvm_free(ptr, size, dev, stream) 归还。
    _uvm_allocator = CUDAPluggableAllocator(_SO_PATH, "uvm_malloc", "uvm_free")

    # 3. 创建 MemPool：与 NativeCachingAllocator 集成的私有 pool，
    #    use_mem_pool(pool) 上下文内的 torch.empty 等将路由到此 pool。
    _uvm_pool = torch.cuda.MemPool(allocator=_uvm_allocator._allocator)

    # 4. 生命周期安全：用 Py_IncRef 将 _uvm_allocator._allocator（持有
    #    shared_ptr<CUDAAllocator>）的 Python 引用计数永久 +1，防止 Python GC
    #    在 ~MemPool() 运行前释放 C++ CUDAPluggableAllocator 对象。
    #
    #    原因：pybind11 的 subtype_dealloc 在调用 C++ 析构前先清空 __dict__，
    #    因此 pool._allocator_ref 无法保证析构顺序。
    #    Py_IncRef 使 shared_ptr 的引用计数永不归零，裸指针始终有效。
    #    进程退出时 OS 回收全部 CUDA 内存，无实际泄漏。
    ctypes.pythonapi.Py_IncRef(ctypes.py_object(_uvm_allocator._allocator))


def get_pool() -> torch.cuda.MemPool | None:
    """返回 UVM MemPool 对象，未初始化时返回 None。"""
    return _uvm_pool


def is_enabled() -> bool:
    """返回 UVM pool 是否已初始化。"""
    return _uvm_pool is not None


def disable() -> None:
    """停止使用 UVM pool（测试/调试用，不释放 CUDA 内存）。"""
    global _uvm_pool, _uvm_allocator
    _uvm_pool = None
    _uvm_allocator = None
