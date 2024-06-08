#ifndef _YOLOV9MIT_TENSORRT_HPP_
#define _YOLOV9MIT_TENSORRT_HPP_

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "coco_names.hpp"
#include "core.hpp"
#include "cuda_runtime_api.h"
#include "tensorrt_logger.h"

namespace yolov9mit
{
using namespace nvinfer1;

class YOLOV9MIT_TensorRT : public AbcYOLOV9MIT
{
public:
    YOLOV9MIT_TensorRT(file_name_t engine_path, int32_t device = 0, float min_iou = 0.5f,
                       float min_confidence = 0.5f, size_t num_classes = 80);
    std::vector<Object> inference(const cv::Mat &frame) override;

private:
    void doInference(std::vector<float> input, std::vector<float> &output0,
                     std::vector<float> &output1);

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
};
} // namespace yolov9mit

#endif
