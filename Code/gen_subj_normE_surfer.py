import os
import numpy as np
import nibabel as nib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

 
# Rotate surface coordinates
def rotate_surface(surf_in, affine, surf_out):
    gii = nib.load(surf_in)
    xyz = gii.darrays[0].data
    xyz = np.c_[xyz, np.ones(len(xyz))]
    xyz = (affine @ xyz.T).T[:, :3].astype(np.float32)

    out = nib.gifti.GiftiImage()
    out.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            xyz,
            intent="NIFTI_INTENT_POINTSET",
            datatype=np.float32,
        )
    )
    for da in gii.darrays[1:]:
        out.add_gifti_data_array(da)
    nib.save(out, surf_out)



# Merge multiple metric files
def merge_metrics(files, out):
    data = np.stack([nib.load(f).darrays[0].data for f in files], axis=1)
    data[np.isnan(data) | (data == 0)] = 0.1
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            data.astype(np.float32),
            intent="NIFTI_INTENT_TIME_SERIES",
        )
    )
    nib.save(gii, out)



# Processing function for each individual CoilCenter
def process_CoilCenter(CoilCenterIdx, mats_all, sub_id, surf_dir, mask_dir, normE_dir):
    try:
        SURFS = {
            "L": {
                "mid": os.path.join(surf_dir, f"{sub_id}.L.midthickness.32k_fs_LR.surf.gii"),
                "mask": os.path.join(mask_dir, f"{sub_id}.L.atlasroi.32k_fs_LR.shape.gii"),
            },
            "R": {
                "mid": os.path.join(surf_dir, f"{sub_id}.R.midthickness.32k_fs_LR.surf.gii"),
                "mask": os.path.join(mask_dir, f"{sub_id}.R.atlasroi.32k_fs_LR.shape.gii"),
            },
        }

        CoilCenter = os.path.join(normE_dir, f"CoilCenter_{CoilCenterIdx+1}")
        Ry = np.diag([-1, 1, -1, 1])
        affines = [Ry @ np.linalg.inv(R) for R in mats_all[CoilCenterIdx]]
        temp_files = []

        tms_idx = CoilCenterIdx + 1

        for i, A in enumerate(affines, 1):
            for hemi in ("L", "R"):
                vol = os.path.join(CoilCenter, f"{sub_id}_TMS_{tms_idx}-{i:04d}_MagStim_D70_normE.nii.gz")
                if not os.path.exists(vol):
                    print(f"[Warning] File does not exist: {vol}, skipping")
                    continue

                mid_r = os.path.join(CoilCenter, f"mid_{hemi}_{i}.surf.gii")
                normE_r = os.path.join(CoilCenter, f"normE.{hemi}.{i}.shape.gii")

                rotate_surface(SURFS[hemi]["mid"], A, mid_r)

                subprocess.run([
                    "wb_command", "-volume-to-surface-mapping",
                    vol, mid_r, normE_r,
                    "-trilinear"
                ], check=True)

                temp_files.extend([mid_r, normE_r])

        merged_files = []
        for hemi in ("L", "R"):
            out = os.path.join(CoilCenter, f"normE.{hemi}.32k_fs_LR.shape.gii")
            merge_metrics(
                [os.path.join(CoilCenter, f"normE.{hemi}.{i}.shape.gii") for i in range(1, len(affines)+1)],
                out
            )
            subprocess.run([
                "wb_command", "-metric-mask",
                out, SURFS[hemi]["mask"], out
            ], check=True)

            merged_files.append(out)

        dtseries_file = os.path.join(CoilCenter, f"CoilCenter_{CoilCenterIdx+1}_normE.dtseries.nii")
        subprocess.run([
            "wb_command", "-cifti-create-dense-timeseries",
            dtseries_file,
            "-left-metric",  os.path.join(CoilCenter, "normE.L.32k_fs_LR.shape.gii"),
            "-roi-left",     SURFS["L"]["mask"],
            "-right-metric", os.path.join(CoilCenter, "normE.R.32k_fs_LR.shape.gii"),
            "-roi-right",    SURFS["R"]["mask"],
        ], check=True)

        for f in temp_files + merged_files:
            if os.path.exists(f):
                os.remove(f)

        return f"CoilCenter {CoilCenterIdx+1} done"

    except Exception as e:
        return f"CoilCenter {CoilCenterIdx+1} failed: {e}"


# Multi-process execution for all CoilCenters
def gen_subj_normE_surfer(sub_id, surf_dir, mask_dir, normE_dir, matsimnibs_path, n_workers=4):
    """
    Parallelly processes all CoilCenters, assigning one process per CoilCenter.
    """

    print(f"[TMSNet INFO] Subject {sub_id}: Generating normE_surfer started...")

    mats_all = np.load(matsimnibs_path)
    CoilCenter_count = len(mats_all)

    args_list = [(i, mats_all, sub_id, surf_dir, mask_dir, normE_dir) for i in range(CoilCenter_count)]
 
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_CoilCenter, *args): args[0]+1 for args in args_list}
        for fut in tqdm(
            as_completed(futures), 
            total=CoilCenter_count, 
            desc= f"[TMSNet INFO] Subject {sub_id} - Generating normE_surfer", 
            unit="CoilCenter"
        ):
            try:
              _ = fut.result()
            except Exception as e:
                idx = futures[fut]
                print(f"[TMSNet WARNING] CoilCenter {idx} failed: {e}")

    print(f"[TMSNet INFO] Subject {sub_id}: Generating normE_surfer completed!")
