from pathlib import Path


root_path = Path("/data/datasets/training")

for i in range(1000):
    file_path = root_path / f"training_tfexample.tfrecord-{i:05d}-of-01000"
    if not file_path.exists():
        print(file_path)