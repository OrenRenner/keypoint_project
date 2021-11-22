#pragma once

// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>


namespace keypoint {
    class KeypointRCNN {
    public:
        KeypointRCNN(const std::string& model, const std::string device = "GPU");
        ~KeypointRCNN();

        bool initialize(const void* userdata = nullptr);
        void* calculate(const cv::Mat& mat, size_t& count);
        bool deinitialize() const;

    private:
        std::string model;
        std::string device;

        Ort::Env env;
        Ort::SessionOptions sessionOptions;
        std::string instanceName{ "keypoint_rcnn" };
        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes;
        size_t numOutputNodes;
        const char* inputName;
        Ort::Session* session;
        ONNXTensorElementDataType inputType;
        std::vector<int64_t> inputDims;
        std::vector<const char*> outputNames;
        size_t output_names_count;
        ONNXTensorElementDataType outputType;
        std::vector<int64_t> outputDims;

    };
};