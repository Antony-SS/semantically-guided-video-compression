import os
from dataset_stream.dataset_stream import DatasetStream, DatasetInstance
import cv2


class DatasetWriter:
    def __init__(self, target_dataset_path: str):
        self.target_dataset_path = target_dataset_path
        self.setup_dataset_directory(target_dataset_path)

    def setup_dataset_directory(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        if not os.path.exists(os.path.join(dataset_path, "frames")):
            os.makedirs(os.path.join(dataset_path, "frames"))
        else:
            raise ValueError(f"Dataset already exists at {dataset_path}, will not overwrite it.")
        with open(os.path.join(dataset_path, "metadata.txt"), "w") as f:
            f.write("frame_path x y yaw timestamp\n")
        return dataset_path

    def write_instance(self, instance: DatasetInstance):
        with open(os.path.join(self.target_dataset_path, "metadata.txt"), "a") as f:
            new_frame_path = os.path.basename(instance.frame_path)
            timestamp = instance.metadata.timestamp
            pose = instance.pose
            f.write(f"{new_frame_path} {pose[0]} {pose[1]} {pose[2]} {timestamp}\n")
            cv2.imwrite(os.path.join(self.target_dataset_path, "frames", new_frame_path), cv2.imread(instance.frame_path))

def write_dataset(dataset_stream: DatasetStream, target_dataset_path: str) -> None:
    """
    Write a dataset stream to a directory.
    Parameters
    ----------
    dataset_stream: DatasetStream
    target_dataset_path: str
    """
    if not os.path.exists(target_dataset_path):
        os.makedirs(target_dataset_path)
    if not os.path.exists(os.path.join(target_dataset_path, "frames")):
        os.makedirs(os.path.join(target_dataset_path, "frames"))
    else:
        raise ValueError(f"Dataset already exists at {target_dataset_path}, will not overwrite it.")

    with open(os.path.join(target_dataset_path, "metadata.txt"), "w") as f:
        f.write("frame_path x y yaw timestamp\n")
        for instance in dataset_stream.iterate():
            f.write(f"{instance.frame_path} {instance.pose[0]} {instance.pose[1]} {instance.pose[2]} {instance.timestamp}\n")
            cv2.imwrite(os.path.join(target_dataset_path, "frames", instance.frame_path), cv2.imread(instance.frame_path))