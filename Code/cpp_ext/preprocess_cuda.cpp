#include <torch/extension.h>

// Declaration of the CUDA launcher implemented in the .cu file
void preprocess_cuda_launcher(
    at::Tensor th_coords,
    at::Tensor invM,
    at::Tensor R,
    at::Tensor t,
    at::Tensor n_voxels,
    at::Tensor th_coords_rot,
    at::Tensor invM_rot,
    at::Tensor th_min,
    at::Tensor th_max,
    at::Tensor in_roi_indices,
    at::Tensor valid_count);

// CUDA wrapper function
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
preprocess_cuda(
    at::Tensor th_coords,
    at::Tensor invM,
    at::Tensor R,
    at::Tensor t,
    at::Tensor n_voxels)
{
    // Input Validation
    TORCH_CHECK(th_coords.is_cuda(), "th_coords must be a CUDA tensor");
    TORCH_CHECK(invM.is_cuda(), "invM must be a CUDA tensor");
    TORCH_CHECK(R.is_cuda(), "R must be a CUDA tensor");
    TORCH_CHECK(t.is_cuda(), "t must be a CUDA tensor");
    TORCH_CHECK(n_voxels.dim() == 1 && n_voxels.size(0) == 3, "n_voxels must be 1D tensor of length 3");

    int n_tetra = th_coords.size(0);

    // Prepare output tensors
    auto options = th_coords.options();
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(th_coords.device());

    auto th_coords_rot = torch::empty_like(th_coords);
    auto invM_rot = torch::empty_like(invM);
    auto th_min = torch::empty({n_tetra, 3}, int_options);
    auto th_max = torch::empty({n_tetra, 3}, int_options);
    auto in_roi_indices = torch::empty({n_tetra}, int_options);
    auto valid_count = torch::zeros({1}, int_options);

    // Execute CUDA kernel via launcher
    preprocess_cuda_launcher(
        th_coords, invM, R, t, n_voxels,
        th_coords_rot, invM_rot,
        th_min, th_max,
        in_roi_indices, valid_count);

    // Extract valid tetrahedral indices (those within the ROI)
    int n_valid = valid_count.item<int>();
    auto in_roi_slice = in_roi_indices.slice(0, 0, n_valid);

    return std::make_tuple(
        th_coords_rot,
        invM_rot,
        th_min,
        th_max,
        in_roi_slice);
}

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("preprocess_cuda", &preprocess_cuda, "CUDA preprocessing for tetrahedra");
}
