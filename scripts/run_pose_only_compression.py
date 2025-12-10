from compression.pose_only_compression import pose_only_compression
import argparse

def run_pose_only_compression(original_dataset_path: str, output_path: str, xy_pose_resolution: float = 1.0, yaw_pose_resolution: float = 45.0):
    return pose_only_compression(original_dataset_path, output_path, xy_pose_resolution, yaw_pose_resolution)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--xy_pose_resolution", type=float, required=False, default=1.0)
    parser.add_argument("--yaw_pose_resolution", type=float, required=False, default=45.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pose_only_compression(args.original_dataset_path, args.output_path, args.xy_pose_resolution, args.yaw_pose_resolution)