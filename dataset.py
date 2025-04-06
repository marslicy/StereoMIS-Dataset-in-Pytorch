import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.rectification import StereoRectifier


class StereoMISDataset(Dataset):
    def __init__(self, data_path, split, size=None):
        assert split in [
            "train",
            "test",
        ], "Split must be one of ['train', 'test']"

        super().__init__()
        self.data_path = data_path
        self.size = size
        self.split = split

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
        with open(
            os.path.join(self.data_path, "sequences.txt"), "r"
        ) as f:
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
                    list = [
                        [v.strip() for v in line.split(" ") if v.strip() != ""]
                        for line in lines
                        if len(line) > 0 and line[0] != "#"
                    ]

                    pose_tq = np.array(
                        [[float(v) for v in l[1:]] for l in list if len(l) > 0],
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
        # load mask
        mask_path = os.path.join(self.data_path, seq, f"masks/{idx:06d}l.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.size) if self.size is not None else mask
        mask = torch.from_numpy(mask).unsqueeze(0)

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

        framel = torch.from_numpy(framel).permute(2, 0, 1)  # HWC to CHW
        framer = torch.from_numpy(framer).permute(2, 0, 1)  # HWC to CHW

        # rectify the frame
        framel, framer = self.calibration[seq](framel, framer)  # in shape of [3, H, W]

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

        intrinsics = (
            self.calibration[seq].calib["intrinsics"]["left"].astype(np.float32)
        )
        baseline = self.calibration[seq].calib["bf"].astype(np.float32)

        # Return the sample as a dictionary
        return {
            "framel1": framel1,
            "framer1": framer1,
            "mask1": mask1,
            "framel2": framel2,
            "framer2": framer2,
            "mask2": mask2,
            "pose1": pose1,
            "pose2": pose2,
            "intrinsics": intrinsics,
            "baseline": baseline,
        }


if __name__ == "__main__":
    data_path = "/mnt/cluster/workspaces/chenyang/StereoMIS_0_0_1"
    # Example usage
    dataset = StereoMISDataset(
        data_path=data_path, split="test", size=(1280 // 2, 1024 // 2)
    )
    print(f"Dataset size: {len(dataset)}")
    for _ in range(10):
        sample = dataset[np.random.randint(0, len(dataset) - 1)]
    sample = dataset[len(dataset) - 1]
    sample = dataset[0]
