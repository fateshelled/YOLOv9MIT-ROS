#include "yolov9mit_ros/yolov9mit_ros.hpp"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include "yolov9mit/utils.hpp"
#include "yolov9mit_ros/cv_bridge_include.hpp"

namespace yolov9mit_ros
{

YOLOV9MIT_Node::YOLOV9MIT_Node(const rclcpp::NodeOptions &options) : Node("yolov9mit_ros", options)
{
    // declare_parameter
    const auto model_path = this->declare_parameter("model_path", "v9-s.vec2box.sim.engine");
    const auto min_iou = this->declare_parameter("min_iou", 0.5f);
    const auto min_confidence = this->declare_parameter("min_confidence", 0.5f);
    const auto class_label_path = this->declare_parameter("class_label_path", "");
    const auto model_type = this->declare_parameter("model_type", "tensorrt");
    const auto tensorrt_device = this->declare_parameter("tensorrt_device", 0);
    const auto input_image_topic = this->declare_parameter("input_image_topic", "image_raw");
    const auto output_image_topic =
        this->declare_parameter("output_image_topic", "yolov9mit_ros/image_raw");
    const auto output_boundingbox_topic =
        this->declare_parameter("output_boundingbox_topic", "yolov9mit_ros/detections");
    this->imshow_ = this->declare_parameter("imshow", false);

    {
        RCLCPP_INFO(this->get_logger(), "Params: ");
        RCLCPP_INFO(this->get_logger(), " - model_path: %s", model_path.c_str());
        RCLCPP_INFO(this->get_logger(), " - min_iou: %f", min_iou);
        RCLCPP_INFO(this->get_logger(), " - min_confidence: %f", min_confidence);
        RCLCPP_INFO(this->get_logger(), " - class_label_path: %s", class_label_path.c_str());
        RCLCPP_INFO(this->get_logger(), " - model_type: %s", model_type.c_str());
        RCLCPP_INFO(this->get_logger(), " - tensorrt_device: %ld", tensorrt_device);
        RCLCPP_INFO(this->get_logger(), " - input_image_topic: %s", input_image_topic.c_str());
        RCLCPP_INFO(this->get_logger(), " - output_image_topic: %s", output_image_topic.c_str());
        RCLCPP_INFO(this->get_logger(), " - output_boundingbox_topic: %s",
                    output_boundingbox_topic.c_str());
        RCLCPP_INFO(this->get_logger(), " - imshow: %s", imshow_ ? "true" : "false");
    }

    // initialize pub/sub
    {
        if (model_path == "")
        {
            std::string msg = "model_path is not set.";
            throw std::runtime_error(msg);
        }
        if (!std::filesystem::exists(model_path))
        {
            std::string msg = "model_path[" + model_path + "] is not exist.";
            throw std::runtime_error(msg);
        }

        if (class_label_path == "")
        {
            std::string msg = "class_label_path is not set.";
            throw std::runtime_error(msg);
        }
        if (!std::filesystem::exists(class_label_path))
        {
            std::string msg = "class_label_path[" + class_label_path + "] is not exist.";
            throw std::runtime_error(msg);
        }

        this->class_names_ = yolov9mit::utils::read_class_labels(class_label_path);

        this->pub_bboxes_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            output_boundingbox_topic, 10);
        this->pub_image_ = image_transport::create_publisher(this, output_image_topic);
        this->sub_image_ = image_transport::create_subscription(
            this, input_image_topic,
            std::bind(&YOLOV9MIT_Node::image_callback, this, std::placeholders::_1), "raw");

        if (this->imshow_)
        {
            cv::namedWindow(this->window_name_, cv::WINDOW_NORMAL);
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
    yolov9mit::utils::draw_objects(image, objects, this->class_names_);
    auto t1_draw = std::chrono::system_clock::now();
    const auto pub_img_msg = cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg();
    this->pub_image_.publish(pub_img_msg);

    // time log
    {
        RCLCPP_INFO(this->get_logger(), "Elapsed");
        auto inf_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1_inf - t0_inf);
        RCLCPP_INFO(this->get_logger(), " - Inference: %.3f ms",
                    (float)inf_elapsed.count() * 0.001);

        auto bboxes_elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t1_bboxes - t0_bboxes);
        RCLCPP_INFO(this->get_logger(), " - to Detection2DArray: %.3f ms",
                    (float)bboxes_elapsed.count() * 0.001);

        auto draw_elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(t1_draw - t0_draw);
        RCLCPP_INFO(this->get_logger(), " - Draw objects: %.3f ms",
                    (float)draw_elapsed.count() * 0.001);

        RCLCPP_INFO(this->get_logger(), "Detections: %ld count", objects.size());
        RCLCPP_INFO(this->get_logger(), " ");
    }

    if (this->imshow_)
    {
        cv::imshow(this->window_name_, image);
        const auto key = cv::waitKey(1);
        if (key == 113)
        {
            cv::destroyWindow(this->window_name_);
            rclcpp::shutdown();
        }
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
        hypothesis.hypothesis.class_id = std::to_string(obj.class_id);
        hypothesis.hypothesis.score = (double)obj.confidence;
        det.results.push_back(hypothesis);

        msg->detections[i] = det;
    }
    return msg;
}
} // namespace yolov9mit_ros

RCLCPP_COMPONENTS_REGISTER_NODE(yolov9mit_ros::YOLOV9MIT_Node)
