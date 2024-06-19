# YOLOv9MIT-ROS

[An MIT rewrite of YOLOv9](https://github.com/WongKinYiu/YOLO) + ROS 2 Humble Object Detection demo

## Supported List
- TensorRT C++

## Requirements
- ROS 2 Humble or rolling
- OpenCV
- TensorRT
- vision_msgs
- v4l2_camera (for Webcam DEMO)

## Build
```bash
cd ros2_ws/src
git clone https://github.com/fateshelled/YOLOv9MIT-ROS
cd ..

colcon build --symlink-install --packages-up-to yolov9mit_ros
```

## Model
### Download
- [Releases](https://github.com/fateshelled/YOLOv9MIT-ROS/releases/tag/v1.0.0)

### Convert to TensorRT Engine
```bash
# S model
wget https://github.com/fateshelled/YOLOv9MIT-ROS/releases/download/v1.0.0/v9-s.vec2box.sim.onnx
trtexec --onnx=v9-s.vec2box.sim.onnx \
        --saveEngine=v9-s.vec2box.sim.engine

# M model
wget https://github.com/fateshelled/YOLOv9MIT-ROS/releases/download/v1.0.0/v9-m.vec2box.sim.onnx
trtexec --onnx=v9-m.vec2box.sim.onnx \
        --saveEngine=v9-m.vec2box.sim.engine

# C model
wget https://github.com/fateshelled/YOLOv9MIT-ROS/releases/download/v1.0.0/v9-c.vec2box.sim.onnx
trtexec --onnx=v9-c.vec2box.sim.onnx \
        --saveEngine=v9-c.vec2box.sim.engine
```

## Webcam Demo
```bash
export YOLOV9_MODEL_PATH="v9-c.vec2box.sim.engine"

ros2 launch yolov9mit_ros launch.xml model_path:=${YOLOV9_MODEL_PATH} video_device:=/dev/video0 imshow:=True
```

## Topic
### Subscribe
- `image_raw`: sensor_msgs/Image
  - Inference input image message

### Publish
- `yolov9mit_ros/detections`: vision_msgs/msg/Detection2DArray
  - BoundingBox output
    - id (`string`): class name
    - bbox (`vision_msgs/msg/BoundingBox2D`): bounding box (center xy + width, height)
    - results (vector of `vision_msgs/msg/ObjectHypothesisWithPose`)
      - class_id (`string`): class index
      - score (`double`): confidence
- `yolov9mit_ros/image_raw`:  sensor_msgs/Image
  - Image with object detection results


