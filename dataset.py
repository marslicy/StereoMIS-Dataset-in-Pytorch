import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.flow import RAFT
from utils.rectification import StereoRectifier
from utils.utils import mask_specularities, tq2RT


class StereoMISDataset(Dataset):
    def __init__(self, data_path, split, depth_map=True, depth_cutoff=300, size=None):
        # depth_cutoff: assume the maximum depth is 300mm
        assert split in [
            "train",
            "test",
        ], "Split must be one of ['train', 'test']"

        super().__init__()
        self.data_path = data_path
        self.size = size
        self.split = split
        self.depth_cutoff = depth_cutoff
        self.depth_map = depth_map
        if self.depth_map:
            self.flow = RAFT()

        # load sequences
        self.sequences = [
            seq
            for seq in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, seq))
        ]
        self.sequences.sort()

        self.video_paths = {}  # dictionary to store video paths
        self.calibration = {}  # dictionary to store calibration objects
        self.frame_indexes = []  # list of tuples (seq, start_frame, num_frames, stride)
        self.set_path_calib_index()
        # print(self.frame_indexes)

        self.pose = {}
        self.load_pose()

    def set_path_calib_index(self):
        # read sequencest.txt as a diction
        stride_dict = {}
        with open(os.path.join(self.data_path, "sequences.txt"), "r") as f:
            lines = f.readlines()[1:]  # Ignore the first line
            for line in lines:
                line = line.strip().split(",")
                if len(line) == 2:
                    seq, stride_seq = line
                    stride_seq = int(stride_seq)
                else:
                    raise ValueError(
                        f"Invalid line in sequences.txt: {line}. Expected format: <sequence_name>,<stride>"
                    )
                stride_dict[seq] = stride_seq

        for seq in self.sequences:
            stride = stride_dict[seq]
            # find path to the video
            all_file = os.listdir(os.path.join(self.data_path, seq))
            for f in all_file:
                if f.endswith(".mp4"):
                    self.video_paths[seq] = os.path.join(self.data_path, seq, f)
                    break
            else:
                raise FileNotFoundError(
                    f"No mp4 file found in {os.path.join(self.data_path, seq)}"
                )

            # load stereo calibration
            ini_path = os.path.join(self.data_path, seq, "StereoCalibration.ini")
            if os.path.exists(ini_path):
                self.calibration[seq] = StereoRectifier(
                    ini_path, img_size_new=self.size
                )
            else:
                raise FileNotFoundError(
                    f"No StereoCalibration.ini file found in {os.path.join(self.data_path, seq)}"
                )

            # find the csv file
            csv_file = os.path.join(self.data_path, seq, f"{self.split}_split.csv")
            if os.path.exists(csv_file):
                with open(csv_file, "r") as f:
                    lines = f.readlines()[1:]  # Ignore the first line
                    for line in lines:
                        frame_index = line.strip().split(",")
                        self.frame_indexes.append(
                            (
                                seq,
                                int(frame_index[0]),
                                (int(frame_index[1]) - int(frame_index[0])) // stride
                                - 1,
                                stride,
                            )
                        )

    def load_pose(self):
        for seq in self.sequences:
            # find the pose file
            pose_path = os.path.join(self.data_path, seq, "groundtruth.txt")
            if os.path.exists(pose_path):
                with open(pose_path, "r") as f:
                    data = f.read()
                    lines = data.replace(",", " ").replace("\t", " ").split("\n")
                    l = [
                        [v.strip() for v in line.split(" ") if v.strip() != ""]
                        for line in lines
                        if len(line) > 0 and line[0] != "#"
                    ]

                    pose_tq = np.array(
                        [[float(v) for v in l[1:]] for l in l if len(l) > 0],
                        dtype=float,
                    )
                    pose_tq = torch.from_numpy(pose_tq)
                    # the first 3 columns are translation, the last 4 columns are quaternion
                    pose_tq[:, :3] = pose_tq[:, :3] * 1000.0  # m to mm
                self.pose[seq] = pose_tq
            else:
                raise FileNotFoundError(
                    f"No pose file found in {os.path.join(self.data_path, seq)}"
                )

    def __len__(self):
        # Return the total number of samples in the dataset
        return sum(num_frames for _, _, num_frames, _ in self.frame_indexes)

    def load_sample(self, seq, idx, video_path):
        # load the video
        cv2_video = cv2.VideoCapture(video_path)
        cv2_video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cv2_video.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from video {video_path}")

        # upper side is the left frame, lower side is the right frame
        framel = frame[: frame.shape[0] // 2]
        framer = frame[frame.shape[0] // 2 :]

        if self.size is not None:
            # resize the frame
            framel = cv2.resize(framel, self.size)
            framer = cv2.resize(framer, self.size)

        # rectify the frame
        framel, framer = self.calibration[seq](framel, framer)  # in shape of [H, W, 3]

        # load mask
        mask_path = os.path.join(self.data_path, seq, f"masks/{idx:06d}l.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size) if self.size is not None else mask
        mask = mask > 0
        mask = mask_specularities(framel, mask)

        framel = torch.from_numpy(framel).permute(2, 0, 1)  # HWC to CHW
        framer = torch.from_numpy(framer).permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask)

        # normalize the frames
        framel = framel.float() / 255.0  # Normalize to [0, 1]
        framer = framer.float() / 255.0  # Normalize to [0, 1]
        framel = (framel - 0.5) / 0.5  # Normalize to [-1, 1]
        framer = (framer - 0.5) / 0.5  # Normalize to [-1, 1]

        return framel, framer, mask

    def __getitem__(self, idx):
        # Load and return a sample from the dataset at the given index
        for seq, start_frame, num_frames, stride in self.frame_indexes:
            if idx < num_frames:
                # Load the video
                video_path = self.video_paths[seq]
                idx = start_frame // 2 * 2 + idx * stride + 1
                break
            else:
                idx -= num_frames

        # Load the sample
        framel1, framer1, mask1 = self.load_sample(seq, idx, video_path)
        framel2, framer2, mask2 = self.load_sample(seq, idx + stride, video_path)

        pose1 = self.pose[seq][idx]
        pose2 = self.pose[seq][idx + stride]

        pose1_RT = tq2RT(pose1, self.depth_cutoff)
        pose2_RT = tq2RT(pose2, self.depth_cutoff)

        # calculate relative pose
        pose = np.linalg.inv(pose1_RT) @ pose2_RT
        pose = torch.from_numpy(pose)

        intrinsics = (
            self.calibration[seq].calib["intrinsics"]["left"].astype(np.float32)
        )
        baseline = (
            self.calibration[seq].calib["bf"].astype(np.float32) / self.depth_cutoff
        )

        framel = torch.stack([framel1, framel2], dim=0)
        framer = torch.stack([framer1, framer2], dim=0)
        if self.depth_map:
            depth = self.flow.depth_estimation(framel, framer, baseline)

        # Return the sample as a dictionary
        return {
            "framel1": framel1,
            "framer1": framer1,
            "depth1": depth[0],
            "mask1": mask1,
            "framel2": framel2,
            "framer2": framer2,
            "depth2": depth[1],
            "mask2": mask2,
            "pose": pose,
            "intrinsics": intrinsics,
            "baseline": baseline,
        }


if __name__ == "__main__":
    # import dataloader
    from torch.utils.data import DataLoader

    data_path = "StereoMIS_0_0_1"
    # Example usage
    dataset = StereoMISDataset(
        data_path=data_path, split="test", size=(1280 // 4, 1024 // 4)
    )
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=False
    )
    print(f"DataLoader size: {len(dataloader)}")
    # Iterate through the dataset
    for i, sample in enumerate(dataloader):
        print(f"Sample {i}:")
        print(sample["framel1"].shape)  # (B, 3, H, W)
        print(sample["framer1"].shape)  # (B, 3, H, W)
        print(sample["depth1"].shape)  # (B, H, W)
        print(sample["mask1"].shape)  # (B, H, W)
        print(sample["framel2"].shape)  # (B, 3, H, W)
        print(sample["framer2"].shape)  # (B, 3, H, W)
        print(sample["depth2"].shape)  # (B, H, W)
        print(sample["mask2"].shape)  # (B, H, W)
        print(sample["pose"].shape)  # (B, 4, 4)
        print(sample["intrinsics"].shape)  # (B, 3, 3)
        print(sample["baseline"].shape)  # (B, )

        framel1_np = sample["framel1"][0].permute(1, 2, 0).numpy()  # Convert to HWC
        mask1_np = sample["mask1"][0].numpy().astype(np.uint8)

        framel1_np = (framel1_np * 0.5 + 0.5) * 255.0  # Convert back to [0, 255]
        framel1_np = framel1_np.astype(np.uint8)

        depth1_np = sample["depth1"][0].numpy()
        depth1_np = (depth1_np * 255.0).astype(np.uint8)

        cv2.imwrite("framel.png", framel1_np)
        cv2.imwrite("mask.png", mask1_np * 255)
        cv2.imwrite("depth.png", depth1_np)
        break
