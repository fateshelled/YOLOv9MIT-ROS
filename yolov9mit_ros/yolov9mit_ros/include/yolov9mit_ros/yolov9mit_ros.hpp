#pragma once

#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <cmath>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "yolov9mit/utils.hpp"
#include "yolov9mit/yolov9mit.hpp"

namespace yolov9mit_ros
{

class YOLOV9MIT_Node : public rclcpp::Node
{
public:
    YOLOV9MIT_Node(const rclcpp::NodeOptions &);

private:
    std::unique_ptr<yolov9mit::AbcYOLOV9MIT> yolo_;
    std::vector<std::string> class_names_;
    bool imshow_ = false;

    image_transport::Subscriber sub_image_;
    image_transport::Publisher pub_image_;

    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_bboxes_;

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &);
    vision_msgs::msg::Detection2DArray::SharedPtr objects_to_bboxes(
        const std::vector<yolov9mit::Object> &, const std_msgs::msg::Header &);
};
} // namespace yolov9mit_ros
