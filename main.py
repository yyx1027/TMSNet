import os
import shutil
import os
from Code.gen_subj_cond import gen_subj_cond
from Code.gen_subj_dataset import gen_subj_dataset
from Code.gen_subj_normE_voxel import gen_subj_normE_voxel
from Code.gen_subj_normE_surfer import gen_subj_normE_surfer

Data_dir = "Data"
subject_ids = [
    d for d in os.listdir(Data_dir)
    if os.path.isdir(os.path.join(Data_dir, d))
]

src_dadt_hr_path = "resources/MagStim_D70_dadt_hr.nii.gz"
src_dadt_lr_path = "resources/MagStim_D70_dadt_lr.nii.gz"
MNI152_image_path = "resources/MNI152_T1_0.7mm.nii.gz"
model_path = "resources/TMSNet_best_metric_model.pth"
coil_name = "MagStim_D70"

for sub_id in subject_ids:

    TMSNet_dir = os.path.join(Data_dir, sub_id, "TMSNet")

    if os.path.exists(TMSNet_dir):
        print(f"[TMSNet INFO] WARNING: Directory {TMSNet_dir} already exists. Deleting it...")
        shutil.rmtree(TMSNet_dir)

    os.makedirs(TMSNet_dir, exist_ok=True)
    print(f"[TMSNet INFO] Subject {sub_id}: Directory {TMSNet_dir} successfully created.")

    dadt_hr_path = os.path.join(TMSNet_dir, "MagStim_D70_dadt_hr.nii.gz")
    dadt_lr_path = os.path.join(TMSNet_dir, "MagStim_D70_dadt_lr.nii.gz")
    shutil.copy2(src_dadt_hr_path, dadt_hr_path)
    shutil.copy2(src_dadt_lr_path, dadt_lr_path)

    cond_hr_dir = os.path.join(TMSNet_dir, "cond_hr")
    cond_lr_dir = os.path.join(TMSNet_dir, "cond_lr")
    normE_dir = os.path.join(TMSNet_dir, "normE")
    json_path = os.path.join(TMSNet_dir, f"dataset_{sub_id}.json")
    os.makedirs(cond_hr_dir, exist_ok=True)
    os.makedirs(cond_lr_dir, exist_ok=True)
    os.makedirs(normE_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    surf_dir = os.path.join(Data_dir, sub_id, "anat", "T1w", "fsaverage_LR32k")
    mask_dir = os.path.join(Data_dir, sub_id, "anat", "MNINonLinear", "fsaverage_LR32k")
    matsimnibs_path = os.path.join(Data_dir, sub_id, "tans", "Network_Frontoparietal", "SearchGrid", "matsimnibs_per_CoilCenter.npy")

    gen_subj_cond(
        sub_id = sub_id,
        Data_dir = Data_dir,
        cond_dir = cond_hr_dir,
        dadt_path = dadt_hr_path,
        MNI152_image_path = MNI152_image_path,
        matsimnibs_path = matsimnibs_path,
        coil_name = coil_name,
        device = "cuda"
    )

    gen_subj_cond(
        sub_id = sub_id,
        Data_dir = Data_dir,
        cond_dir = cond_lr_dir,
        dadt_path = dadt_lr_path,
        MNI152_image_path = MNI152_image_path,
        matsimnibs_path = matsimnibs_path,
        coil_name = coil_name,
        device = "cuda"
    )

    gen_subj_dataset(
        sub_id = sub_id,
        cond_hr_dir = cond_hr_dir,
        cond_lr_dir = cond_lr_dir,
        dadt_hr_path = dadt_hr_path,
        dadt_lr_path = dadt_lr_path,
        json_path = json_path
    )

    gen_subj_normE_voxel(
        sub_id = sub_id,
        json_path = json_path,
        model_path = model_path,
        dadt_hr_path = dadt_hr_path,
        cond_hr_dir = cond_hr_dir,
        normE_dir = normE_dir,
        batch_size = 2,
        num_workers = 14
    )

    gen_subj_normE_surfer(
        sub_id = sub_id,
        surf_dir = surf_dir,
        mask_dir = mask_dir,
        normE_dir = normE_dir,
        matsimnibs_path = matsimnibs_path,
        n_workers = 14
    )