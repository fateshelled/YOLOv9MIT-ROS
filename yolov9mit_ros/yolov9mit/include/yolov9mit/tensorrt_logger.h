#ifndef _MY_TENSORRT_LOGGER_H_
#define _MY_TENSORRT_LOGGER_H_

class MyTRTLogger : public nvinfer1::ILogger
{
public:
    MyTRTLogger(nvinfer1::ILogger::Severity defaultLevel = nvinfer1::ILogger::Severity::kWARNING)
        : m_defaultLevel(defaultLevel)
    {
    }
    virtual void log(Severity severity, const char* msg) noexcept override
    {
        if (severity >= m_defaultLevel)
        {
            switch (severity)
            {
                case Severity::kINTERNAL_ERROR:
                case Severity::kERROR:
                    std::cerr << "[TensorRT Error] " << msg << std::endl;
                    break;
                case Severity::kWARNING:
                    std::cout << "[TensorRT Warning] " << msg << std::endl;
                    break;
                case Severity::kINFO:
                    std::cout << "[TensorRT Info] " << msg << std::endl;
                    break;
                case Severity::kVERBOSE:
                    std::cout << "[TensorRT Verbose] " << msg << std::endl;
                    break;
            }
        }
    }

private:
    nvinfer1::ILogger::Severity m_defaultLevel;
};

#endif
