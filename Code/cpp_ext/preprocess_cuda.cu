#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================
// preprocess kernel
// - Rotate tetrahedral vertices
// - Rotate invM (Inverse Matrix)
// - Compute voxel-space bounding box (bbox)
// - Generate in_roi (Region of Interest) indices
// ============================================================
__global__ void preprocess_kernel(
    int n_tetra, int nx, int ny, int nz,
    const float *__restrict__ th_coords, // [n_tetra, 12]
    const float *__restrict__ invM,      // [n_tetra, 9]
    const float *__restrict__ R,         // [3, 3]
    const float *__restrict__ t,         // [3]
    float *__restrict__ th_coords_rot,   // [n_tetra, 12]
    float *__restrict__ invM_rot,        // [n_tetra, 9]
    int *__restrict__ th_min,            // [n_tetra, 3]
    int *__restrict__ th_max,            // [n_tetra, 3]
    int *__restrict__ in_roi_indices,    // [n_tetra]
    int *__restrict__ valid_count)       // scalar
{
    int tetra = blockIdx.x * blockDim.x + threadIdx.x;
    if (tetra >= n_tetra)
        return;

    // ---- load R, t ----
    const float R00 = R[0], R01 = R[1], R02 = R[2];
    const float R10 = R[3], R11 = R[4], R12 = R[5];
    const float R20 = R[6], R21 = R[7], R22 = R[8];
    const float tx = t[0], ty = t[1], tz = t[2];

    // ---- rotate vertices ----
    const float *coords = th_coords + tetra * 12;
    float *coords_rot = th_coords_rot + tetra * 12;

#pragma unroll
    for (int v = 0; v < 4; ++v)
    {
        float x = coords[v * 3 + 0];
        float y = coords[v * 3 + 1];
        float z = coords[v * 3 + 2];

        coords_rot[v * 3 + 0] = R00 * x + R01 * y + R02 * z + tx;
        coords_rot[v * 3 + 1] = R10 * x + R11 * y + R12 * z + ty;
        coords_rot[v * 3 + 2] = R20 * x + R21 * y + R22 * z + tz;
    }

    // ---- rotate invM: invM_rot = invM * R^T ----
    const float *invM_src = invM + tetra * 9;
    float *invM_out = invM_rot + tetra * 9;

#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        invM_out[i * 3 + 0] =
            invM_src[i * 3 + 0] * R[0] +
            invM_src[i * 3 + 1] * R[1] +
            invM_src[i * 3 + 2] * R[2];

        invM_out[i * 3 + 1] =
            invM_src[i * 3 + 0] * R[3] +
            invM_src[i * 3 + 1] * R[4] +
            invM_src[i * 3 + 2] * R[5];

        invM_out[i * 3 + 2] =
            invM_src[i * 3 + 0] * R[6] +
            invM_src[i * 3 + 1] * R[7] +
            invM_src[i * 3 + 2] * R[8];
    }

    // ---- compute bounding box ----
    float min_x = coords_rot[0], max_x = coords_rot[0];
    float min_y = coords_rot[1], max_y = coords_rot[1];
    float min_z = coords_rot[2], max_z = coords_rot[2];

#pragma unroll
    for (int v = 1; v < 4; ++v)
    {
        float x = coords_rot[v * 3 + 0];
        float y = coords_rot[v * 3 + 1];
        float z = coords_rot[v * 3 + 2];

        min_x = fminf(min_x, x);
        max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y);
        max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z);
        max_z = fmaxf(max_z, z);
    }

    int imin_x = (int)floorf(min_x);
    int imax_x = (int)floorf(max_x);
    int imin_y = (int)floorf(min_y);
    int imax_y = (int)floorf(max_y);
    int imin_z = (int)floorf(min_z);
    int imax_z = (int)floorf(max_z);

    // ---- clamp to image bounds ----
    imin_x = max(0, min(nx - 1, imin_x));
    imax_x = max(0, min(nx - 1, imax_x));
    imin_y = max(0, min(ny - 1, imin_y));
    imax_y = max(0, min(ny - 1, imax_y));
    imin_z = max(0, min(nz - 1, imin_z));
    imax_z = max(0, min(nz - 1, imax_z));

    th_min[tetra * 3 + 0] = imin_x;
    th_min[tetra * 3 + 1] = imin_y;
    th_min[tetra * 3 + 2] = imin_z;

    th_max[tetra * 3 + 0] = imax_x;
    th_max[tetra * 3 + 1] = imax_y;
    th_max[tetra * 3 + 2] = imax_z;

    int idx = atomicAdd(valid_count, 1);
    in_roi_indices[idx] = tetra;
}

// ============================================================
// launcher
// ============================================================
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
    at::Tensor valid_count)
{
    // voxel size on CPU
    auto nvox_cpu = n_voxels.to(at::kCPU);
    int nx = nvox_cpu[0].item<int>();
    int ny = nvox_cpu[1].item<int>();
    int nz = nvox_cpu[2].item<int>();

    int n_tetra = th_coords.size(0);

    const int threads = 256;
    const int blocks = (n_tetra + threads - 1) / threads;

    preprocess_kernel<<<blocks, threads>>>(
        n_tetra, nx, ny, nz,
        th_coords.data_ptr<float>(),
        invM.data_ptr<float>(),
        R.data_ptr<float>(),
        t.data_ptr<float>(),
        th_coords_rot.data_ptr<float>(),
        invM_rot.data_ptr<float>(),
        th_min.data_ptr<int>(),
        th_max.data_ptr<int>(),
        in_roi_indices.data_ptr<int>(),
        valid_count.data_ptr<int>());
}
