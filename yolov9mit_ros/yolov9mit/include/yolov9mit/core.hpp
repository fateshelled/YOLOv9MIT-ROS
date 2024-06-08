#ifndef _YOLOV9MIT_CORE_HPP_
#define _YOLOV9MIT_CORE_HPP_

#include <opencv2/core/types.hpp>

namespace yolov9mit
{

#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread

struct Object
{
    cv::Rect_<float> rect;
    int class_id;
    float confidence;
};

class AbcYOLOV9MIT
{
public:
    AbcYOLOV9MIT() {}
    AbcYOLOV9MIT(float min_iou = 0.45, float min_confidence = 0.3, size_t num_classes = 80)
        : min_iou_(min_iou), min_confidence_(min_confidence), num_classes_(num_classes)
    {
    }
    virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

protected:
    size_t input_w_;
    size_t input_h_;
    float min_iou_;
    float min_confidence_;
    size_t num_classes_;

    cv::Mat preprocess(const cv::Mat &img)
    {
        cv::Mat output(input_h_, input_w_, CV_8UC3);
        cv::resize(img, output, output.size());
        cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
        return output;
    }

    // HWC -> NCHW
    std::vector<float> blobFromImage(const cv::Mat &img)
    {
        static const float scale = 1.0f / 255.0f;
        static const size_t channels = 3;
        const size_t img_h = img.rows;
        const size_t img_w = img.cols;
        std::vector<float> blob_data(channels * img_h * img_w);

        for (size_t c = 0; c < channels; ++c)
        {
            const size_t chw = c * img_w * img_h;
            for (size_t h = 0; h < img_h; ++h)
            {
                const size_t chw_hh = chw + h * img_w;
                for (size_t w = 0; w < img_w; ++w)
                {
                    // blob_data[c * img_w * img_h + h * img_w + w] =
                    //     (float)img.ptr<cv::Vec3b>(h)[w][c] * scale;
                    blob_data[chw_hh + w] = (float)img.ptr<cv::Vec3b>(h)[w][c] * scale;
                }
            }
        }
        return blob_data;
    }

    // HWC -> NHWC
    std::vector<float> blobFromImage_nhwc(const cv::Mat &img)
    {
        static const float scale = 1.0f / 255.0f;
        static const size_t channels = 3;
        const size_t img_hw = img.rows * img.cols;

        std::vector<float> blob_data(channels * img_hw);

        for (size_t i = 0; i < img_hw; ++i)
        {
            for (size_t c = 0; c < channels; ++c)
            {
                blob_data[i * channels + c] = (float)img.data[i * channels + c] * scale;
            }
        }
        return blob_data;
    }

    std::vector<Object> outputs_to_objects(const std::vector<float> &prob_classes,
                                           const std::vector<float> &prob_bboxes,
                                           const int org_img_w, const int org_img_h)
    {
        std::vector<Object> objects;

        const float w_scale = (float)org_img_w / (float)input_w_;
        const float h_scale = (float)org_img_h / (float)input_h_;
        const float x_max = org_img_w - 1;
        const float y_max = org_img_h - 1;

        const size_t length = prob_classes.size() / this->num_classes_;

        for (size_t i = 0; i < length; ++i)
        {
            const size_t idx = i * num_classes_;
            int class_id = -1;
            float max_confidence = -1.0;

            for (size_t class_idx = 0; class_idx < num_classes_; ++class_idx)
            {
                const float conf = prob_classes[idx + class_idx];
                if (conf > max_confidence)
                {
                    class_id = class_idx;
                    max_confidence = conf;
                }
            }
            if (max_confidence > this->min_confidence_)
            {
                const size_t bbox_idx = i * 4;
                float x0 = prob_bboxes[bbox_idx + 0] * w_scale;
                float y0 = prob_bboxes[bbox_idx + 1] * h_scale;
                float x1 = prob_bboxes[bbox_idx + 2] * w_scale;
                float y1 = prob_bboxes[bbox_idx + 3] * h_scale;

                // clip
                x0 = std::max(std::min(x0, x_max), 0.f);
                y0 = std::max(std::min(y0, y_max), 0.f);
                x1 = std::max(std::min(x1, x_max), 0.f);
                y1 = std::max(std::min(y1, y_max), 0.f);

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.class_id = class_id;
                obj.confidence = max_confidence;
                objects.push_back(obj);
            }
        }
        return objects;
    }

    std::vector<Object> decode_outputs(const std::vector<float> &prob_classes,
                                       const std::vector<float> &prob_bboxes, const int org_img_w,
                                       const int org_img_h)
    {
        auto objects = outputs_to_objects(prob_classes, prob_bboxes, org_img_w, org_img_h);
        return objects;

        // TODO add nms
    }
};
} // namespace yolov9mit
#endif
