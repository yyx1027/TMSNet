#include <torch/extension.h>

// Forward declaration of the CUDA launcher defined in the .cu file
void interp_voxels_cuda_launcher(
    at::Tensor n_voxels,
    at::Tensor field,
    at::Tensor th_coords_rot,
    at::Tensor invM_rot,
    at::Tensor th_min,
    at::Tensor th_max,
    at::Tensor in_roi,
    at::Tensor image);

at::Tensor interp_voxels_cuda(
    at::Tensor n_voxels,
    at::Tensor field,
    at::Tensor th_coords_rot,
    at::Tensor invM_rot,
    at::Tensor th_min,
    at::Tensor th_max,
    at::Tensor in_roi)
{
    TORCH_CHECK(field.is_cuda(), "field must be CUDA tensor");
    TORCH_CHECK(th_coords_rot.is_cuda(), "th_coords_rot must be CUDA tensor");
    TORCH_CHECK(invM_rot.is_cuda(), "invM_rot must be CUDA tensor");
    TORCH_CHECK(th_min.is_cuda(), "th_min must be CUDA tensor");
    TORCH_CHECK(th_max.is_cuda(), "th_max must be CUDA tensor");
    TORCH_CHECK(in_roi.is_cuda(), "in_roi must be CUDA tensor");

    // n_voxels is a small metadata tensor; moving it to CPU for indexing
    auto nvox_cpu = n_voxels.to(at::kCPU);
    int nx = nvox_cpu[0].item<int>();
    int ny = nvox_cpu[1].item<int>();
    int nz = nvox_cpu[2].item<int>();
    int ncomp = field.size(1);

    auto image = at::zeros({nx, ny, nz, ncomp}, field.options());

    interp_voxels_cuda_launcher(
        n_voxels,
        field,
        th_coords_rot,
        invM_rot,
        th_min,
        th_max,
        in_roi,
        image);

    return image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("interp_voxels_cuda",
          &interp_voxels_cuda,
          "Interp voxels CUDA wrapper");
}
