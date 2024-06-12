#include "yolov9mit/yolov9mit_tensorrt.hpp"

#include <fstream>

#include "cuda_runtime_api.h"
#include "yolov9mit/coco_names.hpp"

namespace yolov9mit
{

static inline void print_dims(nvinfer1::Dims dims)
{
    std::cout << "[ ";
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        std::cout << dims.d[i];
        if (i < dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << " ]";
}

static inline void cuda_check(cudaError_t status)
{
    if (status != 0)
    {
        std::cerr << "Cuda Error: " << status << std::endl;
        abort();
    }
}

YOLOV9MIT_TensorRT::YOLOV9MIT_TensorRT(file_name_t engine_path, int device, float min_iou,
                                       float min_confidence, size_t num_classes)
    : AbcYOLOV9MIT(min_iou, min_confidence, num_classes), device_(device)
{
    cudaSetDevice(this->device_);

    this->runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(this->trt_logger_));
    assert(this->runtime_ != nullptr);

    std::ifstream file(engine_path, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        const size_t size = file.tellg();
        file.seekg(0, file.beg);
        char* trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        this->engine_ = std::unique_ptr<ICudaEngine>(
            this->runtime_->deserializeCudaEngine(trtModelStream, size));
        assert(this->engine_ != nullptr);

        delete[] trtModelStream;
    }
    else
    {
        std::cerr << "invalid arguments engine_path: " << engine_path << std::endl;
        return;
    }

    this->context_ = std::unique_ptr<IExecutionContext>(this->engine_->createExecutionContext());
    assert(this->context_ != nullptr);

    const auto input_name = this->engine_->getIOTensorName(this->input_index_);
    const auto output0_name = this->engine_->getIOTensorName(this->output0_index_);
    const auto output1_name = this->engine_->getIOTensorName(this->output1_index_);

    assert(this->engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    assert(this->engine_->getTensorDataType(output0_name) == nvinfer1::DataType::kFLOAT);
    assert(this->engine_->getTensorDataType(output1_name) == nvinfer1::DataType::kFLOAT);

    {
        const auto input_dims = this->engine_->getTensorShape(input_name);
        this->input_h_ = input_dims.d[2];
        this->input_w_ = input_dims.d[3];

        std::cout << "MODEL Input:" << std::endl;
        std::cout << "  name:  " << input_name << std::endl;
        std::cout << "  shape: ";
        print_dims(input_dims);
        std::cout << std::endl;
    }

    {
        auto output0_dims = this->engine_->getTensorShape(output0_name);
        this->output0_size_ = 1;
        for (int j = 0; j < output0_dims.nbDims; ++j)
        {
            this->output0_size_ *= output0_dims.d[j];
        }
        std::cout << "MODEL Output1:" << std::endl;
        std::cout << "  name:  " << output0_name << std::endl;
        std::cout << "  shape: ";
        print_dims(output0_dims);
        std::cout << std::endl;
    }

    {
        auto output1_dims = this->engine_->getTensorShape(output1_name);
        this->output1_size_ = 1;
        for (int j = 0; j < output1_dims.nbDims; ++j)
        {
            this->output1_size_ *= output1_dims.d[j];
        }
        std::cout << "MODEL Output2:" << std::endl;
        std::cout << "  name:  " << output1_name << std::endl;
        std::cout << "  shape: ";
        print_dims(output1_dims);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

std::vector<Object> YOLOV9MIT_TensorRT::inference(const cv::Mat& frame)
{
    // preprocess
    const auto pr_img = preprocess(frame);

    // HWC -> NCHW
    blobFromImage(pr_img);

    // inference
    std::vector<float> output_blob_classes;
    std::vector<float> output_blob_bbox;
    this->doInference(output_blob_classes, output_blob_bbox);

    const auto objects =
        decode_outputs(output_blob_classes, output_blob_bbox, frame.cols, frame.rows);

    return objects;
}

void YOLOV9MIT_TensorRT::doInference(std::vector<float>& output0, std::vector<float>& output1)
{
    void* buffers[3];
    output0.resize(this->output0_size_);
    output1.resize(this->output1_size_);

    // Create GPU buffers on device
    cuda_check(cudaMalloc(&buffers[this->input_index_],
                          3 * this->input_h_ * this->input_w_ * sizeof(float)));
    cuda_check(cudaMalloc(&buffers[this->output0_index_], this->output0_size_ * sizeof(float)));
    cuda_check(cudaMalloc(&buffers[this->output1_index_], this->output1_size_ * sizeof(float)));

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    // cudaMemcpyAsync(dist, src, size, type, stream)
    cuda_check(cudaMemcpyAsync(buffers[this->input_index_], blob_data_.data(),
                               3 * this->input_h_ * this->input_w_ * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    context_->enqueueV2(buffers, stream, nullptr);
    // context_->enqueueV3(stream);
    cuda_check(cudaMemcpyAsync(output0.data(), buffers[this->output0_index_],
                               this->output0_size_ * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaMemcpyAsync(output1.data(), buffers[this->output1_index_],
                               this->output1_size_ * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    cuda_check(cudaFree(buffers[0]));
    cuda_check(cudaFree(buffers[1]));
    cuda_check(cudaFree(buffers[2]));
}

} // namespace yolov9mit
