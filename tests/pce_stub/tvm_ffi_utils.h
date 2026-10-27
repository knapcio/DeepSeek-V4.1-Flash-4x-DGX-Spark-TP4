// Compile-only stand-in for FlashInfer's csrc/tvm_ffi_utils.h (tests/pce_nvcc_check.sh). It declares just enough of
// the TVM-FFI surface the sparse_mla_sm120 orchestrator uses to type-check the generated pce sources offline; it is
// never linked or run. The real header comes from the image's FlashInfer at JIT time.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <sstream>
namespace tvm { namespace ffi {
struct DLDataType { uint8_t code; uint8_t bits; uint16_t lanes; };
struct DLDevice { int device_type; int device_id; };
struct TensorView {
  int64_t size(int64_t) const;
  int64_t stride(int64_t) const;
  int ndim() const;
  DLDataType dtype() const;
  DLDevice device() const;
  void* data_ptr() const;
};
template <typename T> struct Optional {
  bool has_value() const;
  const T& value() const;
};
struct CUDADeviceGuard { explicit CUDADeviceGuard(int); };
struct Check { template <typename X> Check& operator<<(const X&) { return *this; } };
}}  // namespace tvm::ffi
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;
inline cudaStream_t get_stream(tvm::ffi::DLDevice) { return nullptr; }
constexpr tvm::ffi::DLDataType dl_bfloat16{4, 16, 1}, dl_float32{2, 32, 1}, dl_int32{0, 32, 1}, dl_uint8{1, 8, 1};
#define TVM_FFI_ICHECK(c) if (!(c)) tvm::ffi::Check()
#define TVM_FFI_ICHECK_EQ(a, b) TVM_FFI_ICHECK((a) == (b))
#define TVM_FFI_ICHECK_GT(a, b) TVM_FFI_ICHECK((a) > (b))
#define TVM_FFI_ICHECK_LE(a, b) TVM_FFI_ICHECK((a) <= (b))
#define CHECK_CUDA(x) TVM_FFI_ICHECK((x).device().device_type == 2)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ((x).ndim(), d)
#define CHECK_LAST_DIM_CONTIGUOUS(x) TVM_FFI_ICHECK_EQ((x).stride(-1), 1)
#define CHECK_INPUT_TYPE(x, t) TVM_FFI_ICHECK((x).dtype().code == (t).code && (x).dtype().bits == (t).bits)
#define CHECK_INPUT_AND_TYPE(x, t) CHECK_CUDA(x); CHECK_INPUT_TYPE(x, t)
#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(name, fn) extern "C" void* __pce_export_##name() { return (void*)&fn; }
