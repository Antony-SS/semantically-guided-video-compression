from compression.pose_and_semantic_compression import pose_and_semantic_compression
import argparse

def run_pose_and_semantic_compression(original_dataset_path: str, target_path: str, xy_pose_resolution: float = 1.0, yaw_pose_resolution: float = 45.0, similarity_threshold: float = 0.90, model_type: str = "CLIP"):
    return pose_and_semantic_compression(original_dataset_path, target_path, xy_pose_resolution, yaw_pose_resolution, similarity_threshold, model_type)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--xy_pose_resolution", type=float, required=False, default=2.5)
    parser.add_argument("--yaw_pose_resolution", type=float, required=False, default=45.0)
    parser.add_argument("--similarity_threshold", type=float, required=False, default=0.8)
    parser.add_argument("--model_type", type=str, required=False, default="CLIP")   
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pose_and_semantic_compression(args.original_dataset_path, args.target_path, args.xy_pose_resolution, args.yaw_pose_resolution, args.similarity_threshold, args.model_type)