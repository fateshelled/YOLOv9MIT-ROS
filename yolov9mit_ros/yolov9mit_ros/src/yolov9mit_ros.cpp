#include "yolov9mit_ros/yolov9mit_ros.hpp"

#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "yolov9mit/utils.hpp"

namespace yolov9mit_ros
{

YOLOV9MIT_Node::YOLOV9MIT_Node(const rclcpp::NodeOptions &options) : Node("yolov9mit_ros", options)
{
    // declare_parameter
    const std::string param_prefix = "yolov9mit_ros/";
    const auto model_path =
        this->declare_parameter(param_prefix + "model_path", "yolov9mit_with_post.sim.engine");
    const auto min_iou = this->declare_parameter(param_prefix + "min_iou", 0.5f);
    const auto min_confidence = this->declare_parameter(param_prefix + "min_confidence", 0.6f);
    const auto class_label_path = this->declare_parameter(param_prefix + "class_label_path", "");
    const auto model_type = this->declare_parameter(param_prefix + "model_type", "tensorrt");
    const auto tensorrt_device = this->declare_parameter(param_prefix + "tensorrt_device", 0);
    const auto input_image_topic =
        this->declare_parameter(param_prefix + "input_image_topic", "image_raw");
    const auto output_image_topic =
        this->declare_parameter(param_prefix + "output_image_topic", "yolov9mit_ros/image_raw");
    const auto output_boundingbox_topic = this->declare_parameter(
        param_prefix + "output_boundingbox_topic", "yolov9mit_ros/detections");
    this->imshow_ = this->declare_parameter(param_prefix + "imshow", true);

    // initialize pub/sub
    {
        if (class_label_path != "")
        {
            this->class_names_ = yolov9mit::utils::read_class_labels(class_label_path);
        }
        else
        {
            this->class_names_ = yolov9mit::COCO_CLASSES;
        }

        this->pub_bboxes_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            output_boundingbox_topic, 10);
        this->pub_image_ = image_transport::create_publisher(this, output_image_topic);
        this->sub_image_ = image_transport::create_subscription(
            this, input_image_topic,
            std::bind(&YOLOV9MIT_Node::image_callback, this, std::placeholders::_1), "raw");

        if (this->imshow_)
        {
            cv::namedWindow("yolov9mit_ros", cv::WINDOW_AUTOSIZE);
        }
    }

    // initialize inference model
    {
        if (model_type == "tensorrt")
        {
#ifdef ENABLE_TENSORRT
            RCLCPP_INFO(this->get_logger(), "Model Type is TensorRT");
            this->yolo_ = std::make_unique<yolov9mit::YOLOV9MIT_TensorRT>(
                model_path, tensorrt_device, min_iou, min_confidence, this->class_names_.size());
#else
            RCLCPP_ERROR(this->get_logger(), "yolov9mit is not built with TensorRT");
            rclcpp::shutdown();
#endif
        }
        if (!this->yolo_)
        {
            RCLCPP_ERROR(this->get_logger(), "yolov9mit is not initialized.");
            rclcpp::shutdown();
        }
    }

    RCLCPP_INFO(this->get_logger(), "initialized.");
}

void YOLOV9MIT_Node::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    const auto image = cv_bridge::toCvCopy(msg, "bgr8")->image;

    auto t0_inf = std::chrono::system_clock::now();
    const auto objects = this->yolo_->inference(image);
    auto t1_inf = std::chrono::system_clock::now();

    auto t0_bboxes = std::chrono::system_clock::now();
    const auto bboxes = objects_to_bboxes(objects, msg->header);
    auto t1_bboxes = std::chrono::system_clock::now();
    this->pub_bboxes_->publish(*bboxes);

    auto t0_draw = std::chrono::system_clock::now();
    yolov9mit::utils::draw_objects(image, objects);
    auto t1_draw = std::chrono::system_clock::now();
    const auto pub_img_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
    this->pub_image_.publish(pub_img_msg);

    // time log
    {
        RCLCPP_INFO(this->get_logger(), "Elapsed");
        auto inf_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1_inf - t0_inf);
        RCLCPP_INFO(this->get_logger(), " - Inference: %ld ms", inf_elapsed.count());

        auto bboxes_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1_bboxes - t0_bboxes);
        RCLCPP_INFO(this->get_logger(), " - to Detection2DArray: %ld ms", bboxes_elapsed.count());

        auto draw_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1_draw - t0_draw);
        RCLCPP_INFO(this->get_logger(), " - Draw objects: %ld ms", draw_elapsed.count());

        RCLCPP_INFO(this->get_logger(), "Detections: %ld count", objects.size());
        RCLCPP_INFO(this->get_logger(), " ");
    }

    if (this->imshow_)
    {
        cv::imshow("yolov9mit_ros", image);
    }
}

vision_msgs::msg::Detection2DArray::SharedPtr YOLOV9MIT_Node::objects_to_bboxes(
    const std::vector<yolov9mit::Object> &objects, const std_msgs::msg::Header &header)
{
    vision_msgs::msg::Detection2DArray::SharedPtr msg(new vision_msgs::msg::Detection2DArray);
    msg->header = header;

    const auto objects_size = objects.size();
    msg->detections.resize(objects_size);
    for (size_t i = 0; i < objects_size; ++i)
    {
        const auto &obj = objects[i];
        vision_msgs::msg::Detection2D det;
        det.header = header;
        det.id = class_names_[obj.class_id];
        det.bbox.center.position.x = obj.rect.x + obj.rect.width * 0.5;
        det.bbox.center.position.y = obj.rect.y + obj.rect.height * 0.5;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        hypothesis.hypothesis.class_id = class_names_[obj.class_id];
        det.results.push_back(hypothesis);

        msg->detections[i] = det;
    }
    return msg;
}
} // namespace yolov9mit_ros

RCLCPP_COMPONENTS_REGISTER_NODE(yolov9mit_ros::YOLOV9MIT_Node)
