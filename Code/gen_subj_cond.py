import os
import torch
from simnibs import mesh_io
import numpy as np
from simnibs.mesh_tools.mesh_io import ElementData
import nibabel as nib
from Code.interp_voxels_cuda import interp_voxels_cuda
from Code.preprocess_cuda import preprocess_cuda
from tqdm import tqdm

# 刚性变换的逆
def inv_rigid_transform(mat):
    # mat: (4,4) 刚性变换矩阵
    R = mat[:3, :3]
    t = mat[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    inv = torch.eye(4, device=mat.device, dtype=mat.dtype)
    inv[:3, :3] = R_inv
    inv[:3, 3] = t_inv
    return inv

# 生成单个被试的电导率图
def gen_subj_cond(
    sub_id: str,
    Data_dir: str,
    cond_dir: str,
    dadt_path: str,
    MNI152_image_path: str,
    matsimnibs_path: str,
    coil_name: str = "MagStim_D70",
    device: str = "cuda",
):
    """
    为单个被试生成电导率图，并按线圈中心和方向保存到 cond_dir。

    参数：
    ----------
    sub_id : str
        被试 ID
    Data_dir : str
        数据根目录
    cond_dir : str
        输出电导率目录(cond_hr 或 cond_lr)
    dadt_path : str
        目标空间 DADT 图像路径
    MNI152_image_path : str
        原空间 MNI152 图像路径
    matsimnibs_path : str
        包含线圈矩阵的 .npy 文件路径
    coil_name : str, optional
        线圈名称，默认 "MagStim_D70"
    device : str, optional
        计算设备，默认 "cuda"
    """

    subj_dir = os.path.join(Data_dir, sub_id, "tans", "HeadModel", f"m2m_{sub_id}")
    mesh_path = os.path.join(subj_dir, f"{sub_id}.msh")


    cond_name = os.path.basename(cond_dir)

    if cond_name == "cond_hr":
        z_shift = 0
    elif cond_name == "cond_lr":
        z_shift = 120
    else:
        z_shift = 0

    cond_name_readable = "high-resolution conductivity" if cond_name == "cond_hr" else "low-resolution conductivity"
    print(f"[TMSNet INFO] Subject {sub_id}: Generating {cond_name_readable} started...")

    # Step 1：读取原空间图像及 affine
    orig_image = nib.load(MNI152_image_path)
    orig_aff = orig_image.affine

    # Step 2：读取 mesh 与 tetra
    mesh = mesh_io.read_msh(mesh_path)
    ed = ElementData(mesh.elm.tag1)
    ed.mesh = mesh

    msh_th = mesh.crop_mesh(elm_type=4)
    msh_th.elmdata = []
    msh_th.nodedata = []

    v = np.atleast_2d(ed.value)
    if v.shape[0] < v.shape[1]:
        v = v.T
    v = v[mesh.elm.elm_type == 4]

    field = v.astype(np.float32)
    labels = np.rint(field).astype(np.int32)

    label_to_cond_list = [
        0, 0.126, 0.275, 1.654, 0.01,
        0.465, 0.5, 0.008, 0.025, 0.6, 0.16
    ]
    conds = np.array(label_to_cond_list, dtype=np.float32)[labels]

    # Step 3：mesh node 坐标转换到 voxel
    nd = np.hstack([
        msh_th.nodes.node_coord,
        np.ones((msh_th.nodes.nr, 1))
    ])
    inv_affine = np.linalg.inv(orig_aff)
    nd = inv_affine.dot(nd.T).T[:, :3]

    tetra = (msh_th.elm.node_number_list - 1).astype(np.int32)
    th_coords = nd[tetra]

    A = th_coords[:, :3, :3] - th_coords[:, 3, None, :]
    invM = np.linalg.inv(A.transpose(0, 2, 1))

    # Step 4：读取目标空间图像及 affine
    ref_image = nib.load(dadt_path)
    ref_aff = ref_image.affine
    ref_n_voxels = ref_image.header['dim'][1:4]

    # Step 5：准备 CUDA 张量
    ref_aff_t  = torch.tensor(ref_aff,  dtype=torch.float32, device=device)
    orig_aff_t = torch.tensor(orig_aff, dtype=torch.float32, device=device)
    th_coords_t = torch.tensor(th_coords, dtype=torch.float32, device=device)
    invM_t = torch.tensor(invM, dtype=torch.float32, device=device)
    n_voxels_t = torch.tensor(ref_n_voxels, dtype=torch.int32, device=device)
    field_t = torch.tensor(conds, dtype=torch.float32, device=device)

    # Step 6：读取线圈矩阵
    coil_array = np.load(matsimnibs_path)  # (n_CoilCenters, n_directions, 4, 4)
    n_CoilCenters, n_dirs, _, _ = coil_array.shape

    # Step 7：构造 Ry 变换矩阵（加入 z_shift）
    Ry = torch.tensor(
        [
            [-1,  0,  0, 0],
            [ 0,  1,  0, 0],
            [ 0,  0, -1, z_shift],
            [ 0,  0,  0, 1],
        ],
        dtype=torch.float32,
        device=device
    )

    ref_aff_inv_t = torch.linalg.inv(ref_aff_t)
    M1 = ref_aff_inv_t @ Ry
    M2 = orig_aff_t
 
    # Step 8：按线圈中心和方向批处理 coil
    with torch.no_grad():
        for t in tqdm(
            range(n_CoilCenters),
            desc=f"[TMSNet INFO] Subject {sub_id} - Generating {cond_name_readable}",
            unit="CoilCenter"
        ):
            CoilCenter_dir = os.path.join(cond_dir, f"CoilCenter_{t+1}")
            os.makedirs(CoilCenter_dir, exist_ok=True)

            for d in range(n_dirs):
                mat = torch.tensor(
                    coil_array[t, d],
                    dtype=torch.float32,
                    device=device
                )

                torch.cuda.synchronize()

                matsimnibs_inv = inv_rigid_transform(mat)
                T_voxel_t = M1 @ matsimnibs_inv @ M2

                R = T_voxel_t[:3, :3].contiguous()
                t_vec = T_voxel_t[:3, 3].contiguous()

                th_coords_rot_t, invM_rot_t, th_min_t, th_max_t, in_roi_t = preprocess_cuda(
                    th_coords_t,
                    invM_t,
                    R,
                    t_vec,
                    n_voxels_t
                )

                image_t = interp_voxels_cuda(
                    n_voxels_t,
                    field_t,
                    th_coords_rot_t,
                    invM_rot_t,
                    th_min_t,
                    th_max_t,
                    in_roi_t
                )

                torch.cuda.synchronize()

                # 保存 NIfTI
                image = np.squeeze(image_t.cpu().numpy(), axis=3)
                img = nib.Nifti1Pair(image, ref_aff)
                img.set_data_dtype(np.float32)
                filename = os.path.join(
                    CoilCenter_dir,
                    f"{sub_id}_TMS_{t+1}-{d+1:04d}_{coil_name}_{cond_name}.nii.gz"
                )
                nib.save(img, filename)

    print(f"[TMSNet INFO] Subject {sub_id}: Generating {cond_name_readable} completed!")