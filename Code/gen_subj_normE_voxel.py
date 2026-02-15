import os
import json
import torch
import nibabel as nib
from tqdm import tqdm
from Code.TMSNet import TMSNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    MaskIntensityd,
)
from monai.data import Dataset, DataLoader
 
def gen_subj_normE_voxel(
    sub_id: str,
    json_path: str,
    model_path: str,
    dadt_hr_path: str,
    cond_hr_dir: str,
    normE_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
):
    """
    Perform TMSNet inference on a single subject and save the normE results.

    Args:
    -----------
    sub_id : str
        Subject ID.
    json_path : str
        Path to the dataset JSON file.
    model_path : str
        Path to the pre-trained TMSNet model weight file.
    dadt_hr_path : str
        Path to the high-resolution dadt (dadt_H) file.
    cond_hr_dir : str
        Directory path for high-resolution conductivity (cond_H).
    normE_dir : str
        Output directory to save the generated normE files.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        Number of worker processes for the DataLoader.
    """

    print(f"[TMSNet INFO] Subject {sub_id}: Generating normE_voxel started...")

    # Define MONAI preprocessing transforms
    infer_transform = Compose([
        LoadImaged(keys=["dadt_hr", "dadt_lr", "cond_hr", "cond_lr"]),
        EnsureChannelFirstd(keys=["dadt_hr", "dadt_lr", "cond_hr", "cond_lr"]),
        EnsureTyped(keys=["dadt_hr", "dadt_lr", "cond_hr", "cond_lr"]),
        MaskIntensityd(keys=["dadt_hr"], mask_key="cond_hr"),
        MaskIntensityd(keys=["dadt_lr"], mask_key="cond_lr"),
    ])

    # Load dataset JSON
    with open(json_path, "r") as f:
        data_dict = json.load(f)
    data_files = data_dict["data"]


    # Initialize DataLoader
    infer_ds = Dataset(data=data_files, transform=infer_transform)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize Model
    device = torch.device("cuda")

    model = TMSNet(
        spatial_dims=3,
        embedding_dim=192,
        seq_length_H=3240,
        seq_length_L=4096,
        in_channels_H=4,
        in_channels_L=4,
        out_channels=3,
        features=(16, 32, 64, 128, 256, 192, 128, 64, 32),
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load reference affine matrix from high-res NIfTI
    ref_nii = nib.load(dadt_hr_path)
    affine = ref_nii.affine

    # Inference & Save normE
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        for i, batch in enumerate(tqdm(
            infer_loader,
            desc=f"[TMSNet INFO] Subject {sub_id} - Generating normE_voxel",
            unit="batch"
        )):

            # Concatenate inputs (dadt + conductivity) for HR and LR streams
            inputs_hr = torch.cat([batch["dadt_hr"].to(device), batch["cond_hr"].to(device)], dim=1)
            inputs_lr = torch.cat([batch["dadt_lr"].to(device), batch["cond_lr"].to(device)], dim=1)

            # Forward pass
            E_pred = model(inputs_hr, inputs_lr)  # (B, 3, D, H, W)
            normE_pred = torch.norm(E_pred, dim=1, keepdim=True)  # (B,1,D,H,W)

            batch_size_curr = normE_pred.shape[0]

            for b in range(batch_size_curr):
                # Mask out non-GM regions (Gray Matter is defined where cond_hr == 0.275)
                mask = (batch["cond_hr"][b] == 0.275).to(device)
                normE_pred_gm = normE_pred[b] * mask.float()
                normE_np = normE_pred_gm[0].cpu().numpy()  # (D,H,W)

                # Map back to dataset index to determine output file path
                dataset_idx = i * infer_loader.batch_size + b
                cond_hr_file = infer_loader.dataset.data[dataset_idx]['cond_hr']

                # Construct output path based on relative input structure
                rel_path = os.path.relpath(cond_hr_file, cond_hr_dir)
                out_file_name = os.path.basename(rel_path).replace("_cond_hr", "_normE")
                out_path = os.path.join(normE_dir, os.path.dirname(rel_path), out_file_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Save as NIfTI file
                nii = nib.Nifti1Image(normE_np, affine)
                nib.save(nii, out_path)

    print(f"[TMSNet INFO] Subject {sub_id}: Generating normE_voxel completed!")