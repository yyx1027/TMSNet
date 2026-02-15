import os
from torch.utils.cpp_extension import load

# Set the compilers
os.environ['CC'] = 'gcc-11'
os.environ['CXX'] = 'g++-11'

this_dir = os.path.dirname(__file__)
srcs = [
    os.path.join(this_dir, 'cpp_ext', 'preprocess_cuda.cpp'),
    os.path.join(this_dir, 'cpp_ext', 'preprocess_cuda.cu'),
]

# Compile and load the CUDA extension JIT (Just-In-Time)
preprocess_ext = load(
    name='preprocess_cuda',
    sources=srcs,
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

def preprocess_cuda(th_coords, invM, R, t, n_voxels):
    """
    CUDA preprocessing for tetrahedral data.

    Args:
        th_coords: (n_tetra, 4, 3) float32 tensor, original tetrahedral vertex coordinates.
        invM: (n_tetra, 3, 3) float32 tensor, original inverse matrices.
        R: (3, 3) float32 tensor, rotation matrix.
        t: (3,) float32 tensor, translation vector.
        n_voxels: (3,) int32 tensor, voxel grid dimensions [nx, ny, nz].

    Returns:
        th_coords_rot: (n_tetra, 4, 3) float32 tensor, rotated vertex coordinates.
        invM_rot: (n_tetra, 3, 3) float32 tensor, rotated inverse matrices.
        th_min: (n_tetra, 3) int32 tensor, min bounding box coordinates for each tetrahedron.
        th_max: (n_tetra, 3) int32 tensor, max bounding box coordinates for each tetrahedron.
        in_roi: (n_valid,) int32 tensor, indices of tetrahedra within the region of interest.
    """
    return preprocess_ext.preprocess_cuda(th_coords, invM, R, t, n_voxels)
 