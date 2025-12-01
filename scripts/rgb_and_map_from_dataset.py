from mapping.impl.tsrb_map import TSRBMap
from dataset_stream.dataset_stream import DatasetStream
from visualization.image_vis_utils import concatenate_images

from tqdm import tqdm
import argparse
import os
import cv2
import numpy as np
from datetime import datetime

def rgb_and_map_from_dataset(dataset_path: str, 
                             output_path: str = "analysis_outputs/", 
                             padding_x: float = 1.0, 
                             padding_y: float = 1.0, 
                             resolution: float = 0.1, 
                             add_overlay: bool = False) -> None:

    dataset_stream = DatasetStream(dataset_path)
    output_path = os.path.join(output_path, os.path.basename(dataset_path), "rgb_and_map")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_poses = np.array(dataset_stream.poses)
    min_x = np.min(all_poses[:, 0])
    min_y = np.min(all_poses[:, 1])
    max_x = np.max(all_poses[:, 0])
    max_y = np.max(all_poses[:, 1])

    poses = []
    map = TSRBMap(np.array([min_x, min_y, max_x, max_y]), padding_x=padding_x, padding_y=padding_y, resolution=resolution, odometry_data=poses)
    rgb_image = cv2.imread(dataset_stream.get_instance(0).frame_path)

    print(map.gridmap_coords.gridmap_shape)
    print(rgb_image.shape)

    output_image_height = max(map.gridmap_coords.gridmap_shape[0], rgb_image.shape[0])
    output_image_width = map.gridmap_coords.gridmap_shape[1] + rgb_image.shape[1]

    for instance in tqdm(dataset_stream.iterate(), total=len(dataset_stream)):
        index = instance.metadata.index
        rgb_image = cv2.imread(instance.frame_path)
        pose = instance.pose
        poses.append(pose)
        timestamp = instance.metadata.timestamp
        map = TSRBMap(np.array([min_x, min_y, max_x, max_y]), padding_x=padding_x, padding_y=padding_y, resolution=resolution, odometry_data=poses)
        map_vis = map.visualize(binary=True, exponential_scaling=False, visualize_origin=True, visualize_odometry=True)
        output_image = np.zeros((output_image_height, output_image_width, 3), dtype=rgb_image.dtype)

        if add_overlay:
            rgb_image = draw_overlay(rgb_image, pose, timestamp)

        if map_vis.shape[0] > rgb_image.shape[0]: # pad rgb_image
            rgb_image = np.pad(rgb_image, ((0, map_vis.shape[0] - rgb_image.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif rgb_image.shape[0] > map_vis.shape[0]: # pad map_vis
            map_vis = np.pad(map_vis, ((0, rgb_image.shape[0] - map_vis.shape[0]), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
        # fill in output image
        output_image[:, :rgb_image.shape[1]] = rgb_image
        output_image[:, rgb_image.shape[1]:] = map_vis

        cv2.imwrite(os.path.join(output_path, f"{index:06d}.png"), output_image)

        # concatenate images
        concatenated_image = concatenate_images(rgb_image, map_vis, grid_shape=(1, 2))
        cv2.imwrite(os.path.join(output_path, f"{index:06d}.png"), concatenated_image)

def draw_overlay(rgb_image: np.ndarray, pose: np.ndarray, timestamp: float) -> np.ndarray:
    timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
    cv2.putText(rgb_image, f"Timestamp: {timestamp_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    next_text_y = 60
    translation = pose[:2]
    yaw = pose[2]
    cv2.putText(rgb_image, f"Translation: {translation}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(rgb_image, f"Yaw: {yaw}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return rgb_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False, default="analysis_outputs/")
    parser.add_argument("--padding_x", type=float, required=False, default=1.0)
    parser.add_argument("--padding_y", type=float, required=False, default=1.0)
    parser.add_argument("--resolution", type=float, required=False, default=0.05)
    parser.add_argument("--add_overlay", action="store_true", required=False)

    return parser.parse_args()

def main():
    args = parse_args()
    rgb_and_map_from_dataset(args.dataset_path, args.output_path, args.padding_x, args.padding_y, args.resolution, args.add_overlay)

if __name__ == "__main__":
    main()