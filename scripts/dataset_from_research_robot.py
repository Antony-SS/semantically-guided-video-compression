from data_streams.collection_streams.research_robot import tf_static_to_pose_stream
from data_streams.ros2_common.camera_streams import make_rgb_image_stream

import cv2
import os
import argparse
from tqdm import tqdm

def dataset_from_research_robot(bag_path : str, image_topic_name : str, tf_topic_name : str, output_dataset_name : str, output_dir : str, skip_every : int = 1) -> None:
    """
    Create a dataset from a ROS2 bag (MCAP file) recorded on Jing Chen's Research Robot.
    """
    image_stream = make_rgb_image_stream(ros2_mcap_path=bag_path, topic_name=image_topic_name)
    pose_stream = tf_static_to_pose_stream(ros2_mcap_path=bag_path, tf_topic_name=tf_topic_name, use_header_timestamps=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extracted_dataset_path = os.path.join(output_dir, output_dataset_name)
    
    if not os.path.exists(extracted_dataset_path):
        os.makedirs(extracted_dataset_path)
    else:
        raise ValueError(f"Extracted dataset already exists at {extracted_dataset_path}, will not overwrite it.")


    # Create metadata file and frames directory
    metadata_file_path = os.path.join(extracted_dataset_path, "metadata.txt")
    frames_dir_path = os.path.join(extracted_dataset_path, "frames")
    if not os.path.exists(frames_dir_path):
        os.makedirs(frames_dir_path)

    # Iterate through bag and populate metadata file and frames directory
    with open(metadata_file_path, "w") as metadata_file:
        metadata_file.write(f"frame_path x y yaw timestamp\n")
        for index, image_instance in enumerate(tqdm(image_stream.iterate(skip_every=skip_every), total=len(image_stream)//skip_every)):
            pose_instance = pose_stream.get_nearest_instance(image_instance.timestamp)
            pose = pose_instance.pose
            translation = pose.translation
            yaw = pose.euler_flu_degrees()[2]
            metadata_file.write(f"{index:06d}.png {translation[0]} {translation[1]} {yaw} {image_instance.timestamp}\n")
            cv2.imwrite(os.path.join(frames_dir_path, f"{index:06d}.png"), image_instance.data)

    print(f"Created dataset at {extracted_dataset_path} with frameskip {skip_every} for total of {len(image_stream)} frames")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", type=str, required=True)
    parser.add_argument("--image_topic_name", type=str, required=False, default="/camera/camera/color/image_raw")
    parser.add_argument("--tf_topic_name", type=str, required=False, default="/tf")
    parser.add_argument("--output_dataset_name", type=str, required=False, default="dataset")
    parser.add_argument("--output_dir", type=str, required=False, default="extracted-datasets")
    parser.add_argument("--skip_every", type=int, required=False, default=3) # every 3rd frame, which goes from 30fps to 10fps, which is fair initial downsampling
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_from_research_robot(args.bag_path, args.image_topic_name, args.tf_topic_name, args.output_dataset_name, args.output_dir, args.skip_every)

if __name__ == "__main__":
    main()