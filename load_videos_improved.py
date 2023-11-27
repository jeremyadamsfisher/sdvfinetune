"""Adapted from https://github.com/AliaksandrSiarohin/video-preprocessing"""
import subprocess
from argparse import ArgumentParser
from itertools import cycle
from multiprocessing import Pool
from pathlib import Path

import imageio
import pandas as pd
from tqdm import tqdm


def run(run_args):
    """Run the data pipeline for a single video"""
    video_id, args = run_args

    video_folder = Path(args.video_folder)
    video_file = video_folder / (video_id.split("#")[0] + ".mp4")

    if not video_file.exists():
        download(video_file, video_id, args.youtube)

    if not video_file.exists():
        print("Can not load video %s, broken link" % video_id.split("#")[0])
        return

    reader = imageio.get_reader(video_file)
    height, width = reader.get_data(0).shape[:2]

    for _, chunk in pd.read_csv(args.metadata).iterrows():
        if chunk.video_id != video_id:
            continue

        left, top, right, bot = map(int, chunk.bbox.split("-"))

        # Compare to the actual video and resize the bounding box and time stamps
        left = int(left / (chunk.width / width))
        top = int(top / (chunk.height / height))
        right = int(right / (chunk.width / width))
        bot = int(bot / (chunk.height / height))
        start_time = chunk.start / chunk.fps
        end_time = chunk.end / chunk.fps

        # We ultimately need a 576x1024 video, so we take the specified bounding box
        # and pad it to the correct aspect ratio
        delta = int(((right - left) / 1024 * 576 - (bot - top)) / 2)
        if delta > 0:
            top -= delta
            bot += delta
        elif delta < 0:
            left += delta
            right -= delta

        fname = (
            chunk.person_id
            + "#"
            + video_id
            + "#"
            + str(chunk.start).zfill(6)
            + "#"
            + str(chunk.end).zfill(6)
            + ".mp4"
        )
        fp = Path(args.out_folder) / partition / fname

        if fp.exists():
            continue

        try:
            crop_video(video_file, start_time, end_time, left, top, right, bot, fp)
        except subprocess.CalledProcessError:
            print("Error processing video {}".format(video_id))
            continue


def crop_video(video_file, start_time, end_time, left, top, right, bot, out_file):
    """Crop the video"""
    cmd = [
        "ffmpeg",
        "-i",
        str(video_file),
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-vf",
        "crop={}:{}:{}:{}, scale=1024:576, fps=24".format(right - left, bot - top, left, top),
        "-c:a",
        "copy",
        out_file,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def download(video_file, video_id, youtube):
    """Download the video, audio, and subtitles from youtube"""
    cmd = [
        youtube,
        "-f",
        "''best/mp4''",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang",
        "en",
        "--skip-unavailable-fragments",
        "https://www.youtube.com/watch?v=" + video_id,
        "--output",
        video_file,
    ]
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--video_folder", default="youtube-taichi", help="Path to youtube videos"
    )
    parser.add_argument(
        "--metadata", default="taichi-metadata-new.csv", help="Path to metadata"
    )
    parser.add_argument("--out_folder", default="taichi-png", help="Path to output")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")
    parser.add_argument("--youtube", default="./youtube-dl", help="Path to youtube-dl")

    args = parser.parse_args()
    video_folder = Path(args.video_folder)
    out_folder = Path(args.out_folder)
    if not video_folder.exists():
        video_folder.mkdir(parents=True)
    if not out_folder.exists():
        out_folder.mkdir(parents=True)
    for partition in ["test", "train"]:
        partition_folder = out_folder / partition
        if not partition_folder.exists():
            partition_folder.mkdir()

    df = pd.read_csv(args.metadata)
    video_ids = sorted(set(df["video_id"]))
    with Pool(processes=args.workers) as pool:
        iter_ = zip(video_ids, cycle([args]))
        iter_ = pool.imap_unordered(run, iter_)
        for _ in tqdm(iter_, total=len(video_ids)):
            None
