# Standard dataset format for using the compression tools in this repo

Example extracted dataset structure (see example collection for an example)

extracted-datasets/
└── dataset/
    ├── metadata.txt -> each line is frame_path x y yaw timestamp
    └── frames/
        ├── 000000.png
        ├── 000001.png
        └── ...