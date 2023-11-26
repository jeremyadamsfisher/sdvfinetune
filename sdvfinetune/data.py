# %%
import datasets
import multiprocessing as mp
from datasets import Dataset

# %%
import torchaudio
import numpy as np
import IPython.display as ipd
from torchvision.io import read_video

# %%
from pathlib import Path
import subprocess
from tqdm import tqdm

# %%
base = Path("/home/jeremy/Desktop/video-retalking/video-preprocessing/data")
vids = list(trn.glob("*/*.mp4"))
[fp.name for fp in trn_vids][:5]

# %%
ds = Dataset.from_dict({"fp": [str(fp) for fp in trn_vids]})

# %%
def vid_map(batch):
    proc = {"audio": [], "video": [], "fp": []}
    for fp in batch["fp"]:
        video, audio, meta = read_video(fp)
        audio = audio[0],  # discard second channel, if it exists
        T, H, W, C = video.shape
        if T == 0:
            print(f"skipping {fp}")
            continue
        proc["audio"].append(audio)
        proc["video"].append(video)
        proc["fp"].append(fp)
    return proc

dds = ds.map(
    vid_map,
    batched=True,
    remove_columns=["fp"],
    # num_proc=mp.cpu_count() - 1,
)

# %%


# %%


# %%
Dataset

# %%



