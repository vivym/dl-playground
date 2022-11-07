import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


def save_fig(x, y, file_path):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.plot(x, y, color="black")
    fig.tight_layout()
    fig.savefig(
        file_path,
        bbox_inches="tight",
        pad_inches=0.,
    )
    plt.close("all")
    img = Image.open(file_path)
    img = img.convert("L")
    img.save(file_path)


def save_specgram_fig(x, file_path):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.specgram(x, Fs=12000, cmap="gray")
    fig.tight_layout()
    fig.savefig(
        file_path,
        bbox_inches="tight",
        pad_inches=0.,
    )
    plt.close("all")
    img = Image.open(file_path)
    img = img.convert("L")
    img.save(file_path)


def main():
    samples = {
        "useful": [],
        "unuseful": []
    }

    root_path = Path("data/microseismic_event")
    for category in ["useful", "unuseful"]:
        for file_path in tqdm(list((root_path / category).glob("*.txt"))):
            samples[category].append((
                str(file_path.relative_to(root_path)),
                1 if category == "useful" else 0,
            ))
            x = np.loadtxt(file_path)
            x = (x - x.mean()) / x.std()

            image_path = file_path.parent / f"{file_path.stem}_waveform.jpg"
            save_fig(range(x.shape[0]), x, image_path)

            freq = np.fft.fftfreq(x.shape[0], 1. / 12000)
            ft = np.fft.fft(x)
            mag = np.abs(ft) / ft.shape[0]

            image_path = file_path.parent / f"{file_path.stem}_frequency_spectrum.jpg"
            save_fig(freq, mag, image_path)

            image_path = file_path.parent / f"{file_path.stem}_specgram.jpg"
            save_specgram_fig(x, image_path)

    random.seed(233)
    random.shuffle(samples["useful"])
    random.shuffle(samples["unuseful"])

    train_samples = samples["useful"][600:] + samples["unuseful"][600:]
    val_samples = samples["useful"][:600] + samples["unuseful"][:600]

    print("train_samples", len(train_samples))
    print("val_samples", len(val_samples))

    with open("data/microseismic_event/train.json", "w") as f:
        json.dump(train_samples,  f)

    with open("data/microseismic_event/val.json", "w") as f:
        json.dump(val_samples,  f)


if __name__ == "__main__":
    main()
