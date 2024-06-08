#ifndef _YOLOV9MIT_UTILS_HPP_
#define _YOLOV9MIT_UTILS_HPP_

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "coco_names.hpp"
#include "core.hpp"

namespace yolov9mit
{
namespace utils
{

inline std::vector<std::string> read_class_labels(std::string file_name)
{
    std::vector<std::string> class_names;
    std::ifstream ifs(file_name);
    if (ifs.fail())
    {
        return class_names;
    }
    std::string buff;
    while (getline(ifs, buff))
    {
        if (buff == "") continue;
        class_names.push_back(buff);
    }
    return class_names;
}

inline void draw_objects(cv::Mat bgr_image, const std::vector<Object>& objects,
                         const std::vector<std::string>& class_names = COCO_CLASSES)
{
    const size_t COLOR_NUM = COLOR_LIST.size();
    for (const auto& obj : objects)
    {
        const int idx = obj.class_id % COLOR_NUM;
        const cv::Scalar color =
            cv::Scalar(COLOR_LIST[idx].b, COLOR_LIST[idx].g, COLOR_LIST[idx].r);
        const cv::Scalar txt_color =
            cv::Scalar(TEXT_COLOR_LIST[idx].b, TEXT_COLOR_LIST[idx].g, TEXT_COLOR_LIST[idx].r);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.class_id].c_str(), obj.confidence * 100);

        int baseLine = 0;
        const cv::Size label_size =
            cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        const cv::Scalar txt_bg_color = color * 0.7;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        if (y > bgr_image.rows) y = bgr_image.rows;

        cv::rectangle(bgr_image, obj.rect, color, 2);
        cv::rectangle(
            bgr_image,
            cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bg_color, -1);

        cv::putText(bgr_image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX,
                    0.4, txt_color, 1);
    }
}
} // namespace utils
} // namespace yolov9mit
#endif
