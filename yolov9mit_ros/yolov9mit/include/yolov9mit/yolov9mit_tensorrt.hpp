#ifndef _YOLOV9MIT_TENSORRT_HPP_
#define _YOLOV9MIT_TENSORRT_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "core.hpp"
#include "tensorrt_logger.h"

namespace yolov9mit
{
using namespace nvinfer1;

class YOLOV9MIT_TensorRT : public AbcYOLOV9MIT
{
public:
    YOLOV9MIT_TensorRT(const std::string &engine_path, const int32_t device = 0,
                       const float min_iou = 0.5f, const float min_confidence = 0.5f,
                       const size_t num_classes = 80);
    ~YOLOV9MIT_TensorRT();
    std::vector<Object> inference(const cv::Mat &frame) override;

private:
    void doInference(std::vector<float> &output0, std::vector<float> &output1);

    int32_t device_ = 0;
    MyTRTLogger trt_logger_;
    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;
    int32_t output0_size_;
    int32_t output1_size_;
    const int32_t input_index_ = 0;
    const int32_t output0_index_ = 1;
    const int32_t output1_index_ = 2;
    void *inference_buffers_[3];
    std::vector<float> output_blob_classes_;
    std::vector<float> output_blob_bbox_;
};
} // namespace yolov9mit

#endif
