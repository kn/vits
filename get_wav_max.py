import argparse
import os

from utils import load_wav_to_torch


def main(dir):
    print(f"Processing dataset directory: {dir}")
    wav_max = None
    for root, _, files in os.walk(dir):
        for file in files:
            if file.startswith(".") or not file.endswith(".wav"):
                continue
            audio, sampling_rate = load_wav_to_torch(os.path.join(root, file))
            this_max = audio.max().item()
            if wav_max is None:
                wav_max = this_max
            else:
                wav_max = max(this_max, wav_max)
    print(f"Max wav value: {wav_max}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./out/mydataset")
    args = parser.parse_args()
    main(args.dir)
