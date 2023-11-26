from pathlib import Path
import multiprocessing as mp

from datasets import Dataset, DatasetDict
from torchvision.io import read_video

from datasets import Dataset
from torchvision.io import read_video


def tensor_dataset_from_fps(fp_out="vox_celeb", workers=None):
    """Create a tensor dataset from a list of video filepaths"""
    "video-preprocessing/data"
    splits = {}
    for split in ["train", "test"]:
        fps = list((Path("Path") / split).glob("**/*.mp4"))
        fps = fps[:10]
        ds = Dataset.from_dict({"fp": fps})
        ds = ds.map(
            read_video_from_example,
            batched=False,
            num_proc=workers or mp.cpu_count() - 1,
        )
        splits[split] = ds
    dsd = DatasetDict(splits)
    dsd.save_to_disk(fp_out)


def read_video_from_example(example):
    """Read a video from a dataset example"""
    fp = example["fp"]
    video, audio, info = read_video(fp, pts_unit="sec")
    return {"video": video, "audio": audio, "info": info}


def cli():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fp-out", type=str, default="vox_celeb")
    args = parser.parse_args()

    tensor_dataset_from_fps(**vars(args))


if __name__ == "__main__":
    cli()