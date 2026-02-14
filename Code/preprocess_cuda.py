import os
from torch.utils.cpp_extension import load

# 设置编译器
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'

this_dir = os.path.dirname(__file__)
srcs = [
    os.path.join(this_dir, 'cpp_ext', 'preprocess_cuda.cpp'),
    os.path.join(this_dir, 'cpp_ext', 'preprocess_cuda.cu'),
]

# 编译并加载 CUDA 扩展
preprocess_ext = load(
    name='preprocess_cuda',
    sources=srcs,
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def preprocess_cuda(th_coords, invM, R, t, n_voxels):
    """
    CUDA 预处理四面体数据。

    Args:
        th_coords: (n_tetra, 4, 3) float32 tensor，原始四面体顶点坐标
        invM: (n_tetra, 3, 3) float32 tensor，原始逆矩阵
        R: (3, 3) float32 tensor，旋转矩阵
        t: (3,) float32 tensor，平移向量
        n_voxels: (3,) int32 tensor，体素网格尺寸 [nx, ny, nz]

    Returns:
        th_coords_rot: (n_tetra, 4, 3) float32 tensor，旋转后的顶点坐标
        invM_rot: (n_tetra, 3, 3) float32 tensor，旋转后的逆矩阵
        th_min: (n_tetra, 3) int32 tensor，每个四面体包围盒最小值
        th_max: (n_tetra, 3) int32 tensor，每个四面体包围盒最大值
        in_roi: (n_valid,) int32 tensor，有效四面体索引
    """
    return preprocess_ext.preprocess_cuda(th_coords, invM, R, t, n_voxels)
 