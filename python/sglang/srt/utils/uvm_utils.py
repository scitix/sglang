import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cupy as cp
import numpy as np
import pynvml
import torch

logger = logging.getLogger(__name__)


class UVMTensorAllocator:
    def __init__(self, size_gb, device_id=None):
        # If device_id is not specified, use the current CUDA device of this process.
        if device_id is None:
            device_id = torch.cuda.current_device()

        self.total_bytes = size_gb * 1024**3
        self.device_id = device_id

        # set GPU
        torch.cuda.set_device(self.device_id)
        self.device = cp.cuda.Device(self.device_id)
        self.device.use()

        # create mempool
        self.mempool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)

        # alloc memory
        with self.device:
            self.raw_ptr = self.mempool.malloc(self.total_bytes)
            cp.cuda.runtime.memset(self.raw_ptr.ptr, 0, self.total_bytes)

            # set memAdvise: prefer in CPU
            cp.cuda.runtime.memAdvise(
                self.raw_ptr.ptr,
                self.total_bytes,
                cp.cuda.runtime.cudaMemAdviseSetPreferredLocation,
                cp.cuda.runtime.cudaCpuDeviceId,
            )
            cp.cuda.runtime.memAdvise(
                self.raw_ptr.ptr,
                self.total_bytes,
                cp.cuda.runtime.cudaMemAdviseSetAccessedBy,
                self.device_id,
            )
        
        self._current_offset = 0
        logger.info(
            f"Init UVM Pool on GPU:{self.device_id} succeed, size={size_gb:.2f}GB"
        )
    
    def _align(self, size):
        return (size + 255) & ~255

    def _alloc_raw(self, shape, dtype):
        """calculate the offset and check the boundary"""
        if dtype == torch.bfloat16:
            np_dtype = cp.uint16
        else:
            np_dtype = torch.as_tensor([], dtype=dtype).numpy().dtype
        element_size = np.dtype(np_dtype).itemsize
        num_elements = np.prod(shape)
        required_bytes = self._align(num_elements * element_size)

        if self._current_offset + required_bytes > self.total_bytes:
            raise MemoryError("UnifiedMemoryPool: Pool Exceeded!")

        # calculate the current address pointer
        memptr = self.raw_ptr + self._current_offset
        self._current_offset += required_bytes
        
        # construct the cupy array
        cp_arr = cp.ndarray(shape, dtype=np_dtype, memptr=memptr)

        # zero copy to torch tensor
        return torch.as_tensor(cp_arr, device=f"cuda:{self.device_id}").view(dtype)

    def empty(self, shape, dtype=torch.float32):
        return self._alloc_raw(shape, dtype)

    def ones(self, shape, dtype=torch.float32):
        t = self.empty(shape, dtype=dtype)
        t.fill_(1.0)
        return t

    def zeros(self, shape, dtype=torch.float32):
        t = self.empty(shape, dtype=dtype)
        t.zero_()
        return t
    
    def free_all(self):
        self._current_offset = 0


_uvm_tensor_allocator = None


def init_uvm_tensor_allocator(size_gb, device_id=None):
    """Initialize a global UVMTensorAllocator instance.

    This function should typically be called once per process (e.g. per TP
    subprocess). By default it binds the allocator to the current CUDA device
    of that process. Subsequent calls in the same process will reuse the
    existing allocator.
    """
    global _uvm_tensor_allocator

    if _uvm_tensor_allocator is not None:
        logger.warning(
            "UVMTensorAllocator is already initialized on device %d with %.2f GB. "
            "Reusing the existing allocator.",
            _uvm_tensor_allocator.device_id,
            _uvm_tensor_allocator.total_bytes / 1024**3,
        )
        return _uvm_tensor_allocator

    _uvm_tensor_allocator = UVMTensorAllocator(size_gb=size_gb, device_id=device_id)
    return _uvm_tensor_allocator


def get_uvm_tensor_allocator():
    """Return the global UVMTensorAllocator instance, or None if not initialized."""
    return _uvm_tensor_allocator
