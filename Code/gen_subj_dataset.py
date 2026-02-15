import os
import json
import re

def gen_subj_dataset(
    sub_id: str,
    cond_hr_dir: str,
    cond_lr_dir: str,
    dadt_hr_path: str,
    dadt_lr_path: str,
    json_path: str,
) -> str:
    """
    Generates a dataset JSON file for a single subject, used for TMSNet inference.

    Args:
        sub_id (str): Subject identifier.
        cond_hr_dir (str): Directory containing high-resolution conductivity files.
        cond_lr_dir (str): Directory containing low-resolution conductivity files.
        dadt_hr_path (str): Path to the high-resolution dadt file.
        dadt_lr_path (str): Path to the low-resolution dadt file.
        json_path (str): Output path where the generated JSON will be saved.
    """

    print(f"[TMSNet INFO] Subject {sub_id}: Generating dataset JSON started...")

    def sort_CoilCenter_folders(folders):
        def numeric_key(name):
            match = re.search(r"CoilCenter_(\d+)", name)
            return int(match.group(1)) if match else 0
        return sorted(folders, key=numeric_key)
 
    samples = []
    CoilCenter_folders = sort_CoilCenter_folders(os.listdir(cond_hr_dir))
 
    for CoilCenter_dir in CoilCenter_folders:
        cond_hr_CoilCenter_dir = os.path.join(cond_hr_dir, CoilCenter_dir)
        cond_lr_CoilCenter_dir = os.path.join(cond_lr_dir, CoilCenter_dir)
        if not os.path.isdir(cond_hr_CoilCenter_dir) or not os.path.isdir(cond_lr_CoilCenter_dir):
            continue

        for cond_hr_file in sorted(os.listdir(cond_hr_CoilCenter_dir)):
            if not cond_hr_file.endswith("_cond_hr.nii.gz") or not cond_hr_file.startswith(sub_id):
                continue

            cond_lr_file = cond_hr_file.replace("_cond_hr.nii.gz", "_cond_lr.nii.gz")
            cond_hr_path = os.path.join(cond_hr_CoilCenter_dir, cond_hr_file)
            cond_lr_path = os.path.join(cond_lr_CoilCenter_dir, cond_lr_file)

            if not os.path.exists(cond_lr_path):
                print(f"[WARNING] Missing corresponding cond_lr file: {cond_lr_path}")
                continue

            samples.append({
                "dadt_hr": dadt_hr_path.replace("\\", "/"),
                "dadt_lr": dadt_lr_path.replace("\\", "/"),
                "cond_hr": cond_hr_path.replace("\\", "/"),
                "cond_lr": cond_lr_path.replace("\\", "/")
            })

    with open(json_path, "w") as f:
        json.dump({"data": samples}, f, indent=4)

    print(f"[TMSNet INFO] Subject {sub_id}: Generating dataset JSON completed!")

