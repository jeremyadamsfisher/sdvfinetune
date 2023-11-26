import multiprocessing as mp

from datasets import Dataset
from torchvision.io import read_video


def tensor_dataset_from_fps(fps):
    """Given a list of filepaths, return a dataset"""
    ds = Dataset.from_dict({"fp": fps})
    ds = ds.map(
        vid_map,
        batched=True,
        num_proc=mp.cpu_count() - 1,
    )
    return ds


def vid_map(batch):
    batch = {"audio": [], "video": [], "audio_fps": [], "video_fps": [], **batch}
    for fp in batch["fp"]:
        # TODO: probably want to re-encode everything to the same FPS
        video, audio, metadata = read_video(fp)
        audio = audio[0]  # discard second channel, if it exists
        batch["audio"].append(audio)
        batch["video"].append(video)
        batch["audio_fps"].append(metadata["audio_fps"])
        batch["video_fps"].append(metadata["video_fps"])
    return batch
