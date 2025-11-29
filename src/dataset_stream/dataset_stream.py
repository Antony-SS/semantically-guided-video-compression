from data_streams.core.data_stream import DataStream
import os
from typing import Optional
import numpy as np
from data_models.core.base_model import BaseInstance
from data_models.core.base_metadata import BaseMetadata
from typing import List

class DatasetInstance(BaseInstance):

    class Config:
        arbitrary_types_allowed = True

    frame_path: str
    pose: np.ndarray # x, y, yaw
    metadata: BaseMetadata


class DatasetStream(DataStream):

    class Config:
        arbitrary_types_allowed = True

    dataset_path: Optional[str] = None
    frames_dir_path: Optional[str] = None

    timestamps_: Optional[List[float]] = None
    poses_: Optional[List[np.ndarray]] = None
    image_paths_: Optional[List[str]] = None

    
    def __init__(self, dataset_path: str):
        frames_dir_path = os.path.join(dataset_path, "frames")
        super().__init__(
            dataset_path=dataset_path,
            frames_dir_path=frames_dir_path
        )
    
    def __len__(self):
        return len(os.listdir(self.frames_dir_path))

    @property
    def timestamps(self) -> List[float]:    
        if not self.timestamps_:
            with open(os.path.join(self.dataset_path, "metadata.txt"), "r") as f:
                self.timestamps_ = [float(line.strip().split(" ")[-1]) for line in f.readlines()[1:]] # last column is timestamp, first line is header
        return self.timestamps_
    @property
    def poses(self) -> List[np.ndarray]:
        if not self.poses_:
            with open(os.path.join(self.dataset_path, "metadata.txt"), "r") as f:
                self.poses_ = [np.array([float(x) for x in line.strip().split(" ")[1:4]]) for line in f.readlines()[1:]] # last column is timestamp, first line is header
        return self.poses_
    
    @property
    def image_paths(self) -> List[str]:
        if not self.image_paths_:
            with open(os.path.join(self.dataset_path, "metadata.txt"), "r") as f:
                self.image_paths_ = [line.strip().split(" ")[0] for line in f.readlines()[1:]] # first column is frame_path, first line is header
        return self.image_paths_
    
    def make_instance(self, instance_metadata: BaseMetadata) -> DatasetInstance:
        index = instance_metadata.index
        
        return DatasetInstance(
            frame_path=os.path.join(self.frames_dir_path, self.image_paths[index]),
            pose=self.poses[index],
            metadata=BaseMetadata(index=index, timestamp=self.timestamps[index])
        )