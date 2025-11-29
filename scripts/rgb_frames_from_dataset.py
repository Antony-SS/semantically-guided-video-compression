from dataset_stream.dataset_stream import DatasetStream

import argparse
import cv2
import os
from datetime import datetime
from tqdm import tqdm


def rgb_frames_from_dataset(dataset_path : str, output_dir : str, overlay_pose : bool = False, overlay_timestamps : bool = False) -> None:
    dataset_stream = DatasetStream(dataset_path)

    output_dir = os.path.join(output_dir, os.path.basename(dataset_path), "rgb_frames")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for instance in tqdm(dataset_stream.iterate(), total=len(dataset_stream)):
        image_path = instance.frame_path
        image = cv2.imread(image_path)

        next_text_y = 30
        if overlay_timestamps:
            timestamp = instance.metadata.timestamp
            timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            cv2.putText(image, f"Timestamp: {timestamp_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            next_text_y = 60

        if overlay_pose:
            pose = instance.pose
            translation = pose[:2]
            yaw = pose[2]
            cv2.putText(image, f"Translation: {translation}", (10, next_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {yaw}", (10, next_text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False, default="analysis_outputs/")
    parser.add_argument("--overlay_pose", action="store_true", required=False)
    parser.add_argument("--overlay_timestamps", action="store_true", required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    rgb_frames_from_dataset(args.dataset_path, args.output_dir, args.overlay_pose, args.overlay_timestamps)

if __name__ == "__main__":
    main()