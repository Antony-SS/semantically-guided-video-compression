from data_streams.ros2_common.camera_streams import make_rgb_image_stream
from data_streams.ros2_common.pose_streams import make_odometry_stream

import os
import argparse

def extract_dataset_from_ros2(bag_path : str, image_topic_name : str, pose_topic_name : str, collection_name : str, output_dir : str) -> None:
    """
    Extract a dataset from a ROS2 bag
    """
    image_stream = make_rgb_image_stream(ros2_mcap_path=bag_path, topic_name="/camera/color/image_raw")
    pose_stream = make_odometry_stream(ros2_mcap_path=bag_path, topic_name="/pose")

    for image_instance in image_stream:
        pose_instance = pose_stream.get_nearest_instance(image_instance.timestamp)
        pose = pose_instance.pose
        translation = pose.translation
        rotation = pose.euler_flu_degrees()
        print(f"Timestamp: {image_instance.timestamp}, Translation: {translation}, Rotation: {rotation}")