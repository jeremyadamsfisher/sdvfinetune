from pathlib import Path

import torch
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video


class VideoDataset(Dataset):
    def __init__(self, data_dir):
        fps = Path(data_dir).glob("**/*.mp4")
        # Check that all videos are non-zero in size
        fps = [fp for fp in fps if fp.stat().st_size > 0]
        self.fps = sorted(fps)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        fp = self.fps[idx]
        video, audio, _ = read_video(str(fp), pts_unit="sec")
        assert video.shape[1:] == (
            576,
            1024,
            3,
        ), f"{Path(fp).name} is incorrectly shaped: {video.shape}"
        return video, audio


def collate_av(batch):
    videos, audios = zip(*batch)

    video = torch.nn.utils.rnn.pad_sequence(videos, padding_value=0.0)
    video = rearrange(video, "t b h w c -> b t h w c")

    audios = [rearrange(audio, "c t -> t c") for audio in audios]
    audio = torch.nn.utils.rnn.pad_sequence(audios, padding_value=0.0)
    audio_left, _audio_right = rearrange(audio, "t b c -> c b t")

    return video, audio_left


class VideoDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = VideoDataset(self.data_dir / "train")
        self.test_dataset = VideoDataset(self.data_dir / "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_av,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_av,
        )
