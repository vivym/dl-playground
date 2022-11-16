import argparse
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/preprocessed_npz")
    parser.add_argument("--out-path", type=str, default="data/waymo_open_dataset_motion_v_1_1_0/rasterized")
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    for split in ["traning", "validation"]:
        print("split", split)
        p = Pool(args.num_workers)
        for file_path in tqdm((data_path / split).glob("*.npz")):
            p.apply_async()


if __name__ == "__main__":
    main()
