import os
from torch.utils.cpp_extension import load

# Set the specified compilers
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'

this_dir = os.path.dirname(__file__)
srcs = [
    os.path.join(this_dir, 'cpp_ext', 'interp_voxels_cuda.cpp'),
    os.path.join(this_dir, 'cpp_ext', 'interp_voxels_cuda.cu'),
]

# Compile / Load CUDA extension
interp_ext = load(
    name='interp_voxels_cuda',
    sources=srcs,
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

def interp_voxels_cuda(
    n_voxels,
    field,
    th_coords_rot,
    invM_rot,
    th_min,
    th_max,
    in_roi,
):
    """
    CUDA Tetrahedron -> Voxel Interpolation

    Args:
        n_voxels      : (3,) int tensor, [nx, ny, nz]
        field         : (n_tetra, ncomp) float32 CUDA tensor
        th_coords_rot : (n_tetra, 4, 3) float32 CUDA tensor
        invM_rot      : (n_tetra, 3, 3) float32 CUDA tensor
        th_min        : (n_tetra, 3) int32 CUDA tensor (Bounding box min)
        th_max        : (n_tetra, 3) int32 CUDA tensor (Bounding box max)
        in_roi        : (n_inroi,) int32 CUDA tensor (Indices within Region of Interest)

    Returns:
        image         : (nx, ny, nz, ncomp) float32 CUDA tensor
    """

    field = field.contiguous()
    th_coords_rot = th_coords_rot.contiguous()
    invM_rot = invM_rot.contiguous()
    th_min = th_min.contiguous()
    th_max = th_max.contiguous()
    in_roi = in_roi.contiguous()

    return interp_ext.interp_voxels_cuda(
        n_voxels,
        field,
        th_coords_rot,
        invM_rot,
        th_min,
        th_max,
        in_roi,
    )
