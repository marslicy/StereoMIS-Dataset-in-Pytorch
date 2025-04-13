import os

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models.optical_flow import raft_large

from utils.stereoanywhere.models.depth_anything_v2 import get_depth_anything_v2
from utils.stereoanywhere.models.stereoanywhere import StereoAnywhere


class RAFT:
    def __init__(self, device="cuda"):
        self.model = raft_large(
            weights="Raft_Large_Weights.DEFAULT", progress=False
        ).to(device)
        self.model = self.model.eval()
        self.device = device

    def depth_estimation(self, framel, framer, baseline):
        flow = self.__call__(framel, framer)
        flow = flow[:, 0, :, :]
        depth = baseline / -flow

        valid = torch.logical_and((depth > 0), (depth <= 1.0))
        depth[~valid] = 1.0

        return depth.detach().cpu()

    def __call__(self, framel, framer):
        framel = framel.to(self.device)
        framer = framer.to(self.device)
        # normalize the frames
        framel = framel.float() / 255.0  # Normalize to [0, 1]
        framer = framer.float() / 255.0  # Normalize to [0, 1]
        framel = (framel - 0.5) / 0.5  # Normalize to [-1, 1]
        framer = (framer - 0.5) / 0.5  # Normalize to [-1, 1]

        flow = self.model(framel, framer)[-1]

        # if flow is in the shape of (B, 2, H, W) then return flow
        if len(flow.shape) == 4:
            return flow.detach().cpu()
        # if flow is in the shape of (2, H, W) then add batch dimension
        else:
            flow = flow.unsqueeze(0)
            return flow.detach().cpu()


class StereoAntwhere:
    def __init__(self, mono_model="base", device="cuda"):
        if mono_model not in ["small", "base", "large"]:
            raise ValueError(
                "mono_model must be one of ['small', 'base', 'large'], got {}".format(
                    mono_model
                )
            )

        os.makedirs("utils/stereoanywhere/weights", exist_ok=True)
        stereo_path = "utils/stereoanywhere/weights/StereoAnywhere.tar"
        # download the model if it does not exist
        if not os.path.exists(stereo_path):
            # download the model from https://drive.usercontent.google.com/u/0/uc?id=1f6JG7HGoVIlwRpfp4EEzf3iZGclEgyOY&export=download
            os.system(
                f"wget -O {stereo_path} https://drive.usercontent.google.com/u/0/uc?id=1f6JG7HGoVIlwRpfp4EEzf3iZGclEgyOY&export=download"
            )
        checkpoint = torch.load(stereo_path)

        if mono_model == "small":
            mono_path = "utils/stereoanywhere/weights/depth_anything_v2_vits.pth"
            if not os.path.exists(mono_path):
                os.system(
                    f"wget -O {mono_path} https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true"
                )
        elif mono_model == "base":
            mono_path = "utils/stereoanywhere/weights/depth_anything_v2_vitb.pth"
            if not os.path.exists(mono_path):
                os.system(
                    f"wget -O {mono_path} https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true"
                )
        elif mono_model == "large":
            mono_path = "utils/stereoanywhere/weights/depth_anything_v2_vitl.pth"
            if not os.path.exists(mono_path):
                os.system(
                    f"wget -O {mono_path} https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
                )

        self.device = device

        self.stereo_model = StereoAnywhere({})

        try:
            self.stereo_model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            # Remove 'module.' prefix from keys if it exists
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                new_key = key.replace("module.", "")  # Remove 'module.' prefix
                new_state_dict[new_key] = value
            self.stereo_model.load_state_dict(new_state_dict)

        self.stereo_model.eval()
        self.stereo_model.to(device)

        self.mono_model = get_depth_anything_v2(mono_path)
        self.mono_model.eval()
        self.mono_model.to(device)

    def depth_estimation(self, framel, framer, baseline):
        pred_disps = self.__call__(framel, framer)

        pred_disps = pred_disps[:, 0, :, :]
        depth = baseline / -pred_disps

        valid = torch.logical_and((depth > 0), (depth <= 1.0))
        depth[~valid] = 1.0

        return depth.detach().cpu()

    def __call__(self, framel, framer):
        # output is the disparity map
        framel = framel.float().to(self.device) / 255
        framer = framer.float().to(self.device) / 255

        mono_depths = self.mono_model.infer_image(
            torch.cat([framel, framer], 0),
            input_size_height=framel.shape[-2],
            input_size_width=framel.shape[-1],
        )
        # Normalize depth between 0 and 1
        mono_depths = (mono_depths - mono_depths.min()) / (
            mono_depths.max() - mono_depths.min()
        )
        mono_depths = torch.stack(
            [
                (mono_depth - mono_depth.min()) / (mono_depth.max() - mono_depth.min())
                for mono_depth in mono_depths
            ]
        )

        pred_disps, _ = self.stereo_model(
            framel,
            framer,
            mono_depths[: len(mono_depths) // 2],
            mono_depths[len(mono_depths) // 2 :],
            test_mode=True,
        )

        # flow in the shape of (B, 1, H, W)
        return pred_disps.detach().cpu()
