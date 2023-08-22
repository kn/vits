import argparse
import fnmatch
import os
import random
import re
from encodec.utils import convert_audio
import torchaudio
from tqdm import tqdm


def main(dataset_dir, out_dir, num_val, sampling_rate):
    print(f"Processing dataset directory: {dataset_dir}")
    audio_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.startswith(".") or not file.endswith(".wav"):
                continue
            audio_paths.append(os.path.join(root, file))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Converting audio files to {sampling_rate} Hz")
    for audio_path in tqdm(audio_paths):
        audio, sr = torchaudio.load(audio_path) # type: ignore
        audio = convert_audio(audio, sr, sampling_rate, 1)
        new_audio_path = os.path.join(out_dir, os.path.basename(audio_path))
        torchaudio.save(new_audio_path, audio, sampling_rate) # type: ignore

    id_to_txt = {}
    for root, _, filenames in os.walk(dataset_dir):
        for filename in fnmatch.filter(filenames, "transcripts.txt"):
            with open(os.path.join(root, filename)) as f:
                next_id = None
                for line in f:
                    text = line.strip()
                    if re.match(r"^\d{5}:$", text):
                        assert next_id is None, next_id
                        next_id = text[:-1]
                    elif text:
                        assert next_id is not None, text
                        txt_path = os.path.join(out_dir, f"{next_id}.txt")
                        id_to_txt[next_id] = text
                        with open(txt_path, "w") as f:
                            f.write(text)
                        next_id = None

    manifest_path = os.path.join(out_dir, "mydataset_filelist.txt")
    with open(manifest_path, "w") as f:
        for root, _, filenames in os.walk(out_dir):
            for filename in fnmatch.filter(filenames, "*.wav"):
                id = filename.split(".")[0]
                with open(os.path.join(root, f"{id}.txt")) as t:
                    text = t.read().replace("\n", " ").strip()
                f.write(f"{os.path.join(root, filename)}|{text}\n")
    
    with open(manifest_path) as f:
        lines = f.readlines()
        random.shuffle(lines)
        with open(manifest_path.replace("_filelist.txt", "_train_filelist.txt"), "w") as w:
            w.writelines(lines[:-num_val])
        with open(manifest_path.replace("_filelist.txt", "_val_filelist.txt"), "w") as w:
            w.writelines(lines[-num_val:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="./out/mydataset")
    parser.add_argument("--num_val", type=int)
    parser.add_argument("--sampling_rate", type=int, default=22050)
    args = parser.parse_args()
    main(args.dataset_dir, args.out_dir, args.num_val, args.sampling_rate)
