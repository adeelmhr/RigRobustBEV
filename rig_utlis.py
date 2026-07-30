# rig_utils.py
import torch

# ➊  hard-code the measured SUB→SUV deltas (in the ego frame, metres)
CAM_DELTA = {
    'CAM_FRONT'       : torch.tensor([+0.45,  0.00, +0.50]),
    'CAM_FRONT_LEFT'  : torch.tensor([+0.45, +0.10, +0.50]),
    'CAM_FRONT_RIGHT' : torch.tensor([+0.45, -0.10, +0.50]),
    'CAM_BACK'        : torch.tensor([-0.45,  0.00, +0.50]),
    'CAM_BACK_LEFT'   : torch.tensor([-0.45, +0.10, +0.50]),
    'CAM_BACK_RIGHT'  : torch.tensor([-0.45, -0.10, +0.50]),
}

def apply_delta(lidar2img: torch.Tensor, cam_name: str) -> torch.Tensor:
    """
    Shift a lidar-to-image 4×4 matrix from the SUB rig to the SUV rig.

    lidar2img : [..., 4, 4]  (torch)  original matrix for SUB
    cam_name  : str          one of the six camera tokens
    returns   : [..., 4, 4]  shifted matrix for SUV
    """
    out = lidar2img.clone()
    δ = CAM_DELTA[cam_name].to(out.device)

    # lidar2img =  K · [ R | t ]
    # R is unchanged; t becomes  t + δ  (δ expressed in ego frame)
    out[..., :3, 3] += δ            # works because ego == lidar frame in CamShift

    return out