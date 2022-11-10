from pathlib import Path

from tqdm import tqdm

from dl_playground.data.datasets.waymo_motion import WaymoMotionDataset


def check_dataset(root_path: Path, is_training: bool = False):
    dataset = WaymoMotionDataset(
        root_path=root_path,
        is_training=is_training,
    )

    num_samples = len(dataset)

    for i in tqdm(range(8090, num_samples)):
        dataset[i]

def main():
    root_path = Path("data/waymo_open_dataset_motion_v_1_1_0/preprocessed")

    check_dataset(root_path / "training", is_training=True)
    check_dataset(root_path / "validation", is_training=False)


if __name__ == "__main__":
    main()
