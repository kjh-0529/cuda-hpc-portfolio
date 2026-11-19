// PyTorch bindings for the softmax and LayerNorm kernels in 02-llm-kernels.
//
// Exposes forward-only (inference) ops:
//   softmax(x)                          softmax over the last dimension
//   layer_norm(x, weight, bias, eps)    LayerNorm over the last dimension
// x may have any number of leading dimensions; it is viewed as [rows x cols]
// with cols = x.size(-1). Kernels run on PyTorch's current CUDA stream, and the
// row-length dispatchers from softmax step 05 / LayerNorm step 03 pick the kernel.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

// Report CUDA errors as Python exceptions instead of exiting the process.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        TORCH_CHECK(err_ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err_)); \
    } while (0)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#include "layernorm/kernels/03_registers.cuh"
#include "softmax/kernels/05_registers.cuh"

namespace {

void check_input(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

// The kernels use float4 loads, which need 16-byte aligned rows. A tensor that is
// a view with an odd storage offset may not be aligned; copy it if so.
torch::Tensor contiguous_aligned(const torch::Tensor& t) {
    torch::Tensor c = t.contiguous();
    if (reinterpret_cast<std::uintptr_t>(c.data_ptr()) % 16 != 0) c = c.clone();
    return c;
}

void rows_cols(const torch::Tensor& x, int& rows, int& cols) {
    TORCH_CHECK(x.dim() >= 1, "input must have at least one dimension");
    const int64_t c = x.size(-1);
    const int64_t r = c == 0 ? 0 : x.numel() / c;
    TORCH_CHECK(c <= INT_MAX && r <= INT_MAX, "input too large");
    rows = static_cast<int>(r);
    cols = static_cast<int>(c);
}

}  // namespace

torch::Tensor softmax(const torch::Tensor& x_in) {
    check_input(x_in, "x");
    const c10::cuda::CUDAGuard guard(x_in.device());
    torch::Tensor x = contiguous_aligned(x_in);
    torch::Tensor y = torch::empty_like(x);
    int rows, cols;
    rows_cols(x, rows, cols);
    if (rows == 0 || cols == 0) return y;
    launch_softmax_auto(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols,
                        at::cuda::getCurrentCUDAStream());
    return y;
}

torch::Tensor layer_norm(const torch::Tensor& x_in, const torch::Tensor& weight_in,
                         const torch::Tensor& bias_in, double eps) {
    check_input(x_in, "x");
    check_input(weight_in, "weight");
    check_input(bias_in, "bias");
    TORCH_CHECK(weight_in.device() == x_in.device() && bias_in.device() == x_in.device(),
                "x, weight and bias must be on the same device");
    const c10::cuda::CUDAGuard guard(x_in.device());
    torch::Tensor x = contiguous_aligned(x_in);
    int rows, cols;
    rows_cols(x, rows, cols);
    TORCH_CHECK(weight_in.dim() == 1 && weight_in.size(0) == cols,
                "weight must have shape [", cols, "]");
    TORCH_CHECK(bias_in.dim() == 1 && bias_in.size(0) == cols,
                "bias must have shape [", cols, "]");
    torch::Tensor weight = contiguous_aligned(weight_in);
    torch::Tensor bias = contiguous_aligned(bias_in);
    torch::Tensor y = torch::empty_like(x);
    if (rows == 0 || cols == 0) return y;
    launch_layernorm_auto(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                          y.data_ptr<float>(), rows, cols, static_cast<float>(eps),
                          at::cuda::getCurrentCUDAStream());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Hand-written CUDA softmax and LayerNorm (forward only, float32)";
    m.def("softmax", &softmax, "Softmax over the last dimension", py::arg("x"));
    m.def("layer_norm", &layer_norm, "LayerNorm over the last dimension", py::arg("x"),
          py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
