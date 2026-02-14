#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// 主 CUDA kernel：每个 thread 处理一个 tetra（遍历其 bbox）
// ============================================================
__global__ void interp_voxels_kernel(
    const int nx, const int ny, const int nz,
    const int ncomp,
    const float *__restrict__ field,
    const float *__restrict__ th_coords, // [n_tetra, 12]
    const float *__restrict__ invM,      // [n_tetra, 9]
    const int *__restrict__ th_min,      // [n_tetra, 3]
    const int *__restrict__ th_max,      // [n_tetra, 3]
    const int *__restrict__ in_roi,      // [n_inroi]
    const int n_inroi,
    float *__restrict__ image) // [nx, ny, nz, ncomp]
{
    int tetra_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tetra_idx >= n_inroi)
        return;

    int tetra = in_roi[tetra_idx];

    // 读取 bounding box
    int xmin = th_min[tetra * 3 + 0];
    int ymin = th_min[tetra * 3 + 1];
    int zmin = th_min[tetra * 3 + 2];
    int xmax = th_max[tetra * 3 + 0];
    int ymax = th_max[tetra * 3 + 1];
    int zmax = th_max[tetra * 3 + 2];

    // clamp 到图像范围
    xmin = max(0, xmin);
    ymin = max(0, ymin);
    zmin = max(0, zmin);
    xmax = min(nx - 1, xmax);
    ymax = min(ny - 1, ymax);
    zmax = min(nz - 1, zmax);

    if (xmax < xmin || ymax < ymin || zmax < zmin)
        return;

    const float *fld = field + tetra * ncomp;
    const float *thc = th_coords + tetra * 12;
    const float *invMt = invM + tetra * 9;

    // ref 顶点（第 4 个）
    float refx = thc[9];
    float refy = thc[10];
    float refz = thc[11];

    const float eps = 1e-5f;

    // 遍历该 tetra 的 voxel bbox
    for (int x = xmin; x <= xmax; ++x)
    {
        float xc = x - refx;
        for (int y = ymin; y <= ymax; ++y)
        {
            float yc = y - refy;
            for (int z = zmin; z <= zmax; ++z)
            {
                float zc = z - refz;

                // 重心坐标
                float b0 = invMt[0] * xc + invMt[1] * yc + invMt[2] * zc;
                float b1 = invMt[3] * xc + invMt[4] * yc + invMt[5] * zc;
                float b2 = invMt[6] * xc + invMt[7] * yc + invMt[8] * zc;
                float b3 = 1.0f - b0 - b1 - b2;

                // 是否在四面体内
                if (b0 >= -eps && b1 >= -eps && b2 >= -eps && b3 >= -eps)
                {
                    int lin = ((x * ny + y) * nz + z) * ncomp;
#pragma unroll
                    for (int c = 0; c < ncomp; ++c)
                        image[lin + c] = fld[c];
                }
            }
        }
    }
}

// ============================================================
// C++ / PyTorch launcher
// ============================================================
void interp_voxels_cuda_launcher(
    at::Tensor n_voxels,
    at::Tensor field,
    at::Tensor th_coords_rot,
    at::Tensor invM_rot,
    at::Tensor th_min,
    at::Tensor th_max,
    at::Tensor in_roi,
    at::Tensor image)
{
    const int nx = n_voxels[0].item<int>();
    const int ny = n_voxels[1].item<int>();
    const int nz = n_voxels[2].item<int>();
    const int ncomp = field.size(1);
    const int n_inroi = in_roi.numel();

    if (n_inroi == 0)
        return;

    const int threads = 256;
    const int blocks = (n_inroi + threads - 1) / threads;

    interp_voxels_kernel<<<blocks, threads>>>(
        nx, ny, nz,
        ncomp,
        field.data_ptr<float>(),
        th_coords_rot.data_ptr<float>(),
        invM_rot.data_ptr<float>(),
        th_min.data_ptr<int>(),
        th_max.data_ptr<int>(),
        in_roi.data_ptr<int>(),
        n_inroi,
        image.data_ptr<float>());

#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
#endif
}
