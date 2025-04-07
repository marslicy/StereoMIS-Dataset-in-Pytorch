import torch
from torchvision.models.optical_flow import raft_large


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

        flow = self.model(framel, framer)[-1]

        # flow in the shape of (B, 2, H, W) or (2, H, W)
        return flow.detach().cpu()
