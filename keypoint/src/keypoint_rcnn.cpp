#include "keypoint_rcnn.hpp"
#include <numeric>

template<typename T>
inline T vectorProduct(const std::vector<T>& v)
{
	return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}


keypoint::KeypointRCNN::KeypointRCNN(const std::string& model, const std::string device)
{
	this->model = model;
	this->device = device;
}

keypoint::KeypointRCNN::~KeypointRCNN()
{
}

bool keypoint::KeypointRCNN::initialize(const void* userdata)
{
	try {
		bool useCUDA{ false };
		if (this->device == "GPU") {
			useCUDA = true;
		}

		this->env = Ort::Env{ ORT_LOGGING_LEVEL_WARNING, "keypoint_rcnn" };
		this->sessionOptions.SetIntraOpNumThreads(1);
		if (useCUDA)
		{
			// Using CUDA backend
			// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
			OrtCUDAProviderOptions cuda_options;
			this->sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
		}

		this->sessionOptions.SetGraphOptimizationLevel(
			GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

		std::wstring widestr = std::wstring(this->model.begin(),
			this->model.end());
		const wchar_t* widecstr = widestr.c_str();

		this->session = new Ort::Session(this->env, widecstr, this->sessionOptions);

		this->numInputNodes = this->session->GetInputCount();
		this->numOutputNodes = this->session->GetOutputCount();

		this->inputName = this->session->GetInputName(0, this->allocator);
		Ort::TypeInfo inputTypeInfo = this->session->GetInputTypeInfo(0);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		this->inputType = inputTensorInfo.GetElementType();
		this->inputDims = inputTensorInfo.GetShape();

		this->output_names_count = this->session->GetOutputCount();
		for (int i = 0; i < this->session->GetOutputCount(); i++) {
			const char* outputName = this->session->GetOutputName(i, this->allocator);
			this->outputNames.push_back(outputName);
		}

		Ort::TypeInfo outputTypeInfo = this->session->GetOutputTypeInfo(0);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		this->outputType = outputTensorInfo.GetElementType();
		this->outputDims = outputTensorInfo.GetShape();

		return true;
	}
	catch (std::exception) {
		return false;
	}
}

void* keypoint::KeypointRCNN::calculate(const cv::Mat& mat, size_t& count)
{
	cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
	cv::resize(mat, resizedImageBGR,
		cv::Size(this->inputDims.at(2), this->inputDims.at(3)),
		cv::InterpolationFlags::INTER_CUBIC);
	cv::cvtColor(resizedImageBGR, resizedImageRGB,
		cv::ColorConversionCodes::COLOR_BGR2RGB);
	resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

	cv::dnn::blobFromImage(resizedImage, preprocessedImage);

	size_t inputTensorSize = vectorProduct(this->inputDims);
	std::vector<float> inputTensorValues(inputTensorSize);
	inputTensorValues.assign(preprocessedImage.begin<float>(),
		preprocessedImage.end<float>());

	size_t outputTensorSize = this->outputDims[1]; // vectorProduct(outputDims);
	std::vector<float> outputTensorValues(outputTensorSize);

	std::vector<const char*> inputNames{ inputName };
	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;

	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	inputTensors.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, inputTensorValues.data(), inputTensorSize,
		this->inputDims.data(),
		this->inputDims.size()));

	std::vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{ nullptr }, inputNames.data(),
		inputTensors.data(), 1, this->outputNames.data(), this->output_names_count);

	float* keys_ret = output_tensors[3].GetTensorMutableData<float>();

	float x_coef = (1.0 * resizedImageBGR.cols) / (1.0 * mat.cols);
	float y_coef = (1.0 * resizedImageBGR.rows) / (1.0 * mat.rows);

	std::vector<cv::Point> points;
	int i = 0;
	while (true) {
		cv::Point tmp;
		tmp.x = keys_ret[i] / x_coef;
		i = i + 1;
		tmp.y = keys_ret[i] / y_coef;
		i = i + 1;

		if (keys_ret[i] != 0) {
			i = i + 1;
			points.push_back(tmp);
		}
		else break;
	}

	cv::Point* result = new cv::Point[points.size()];
	for (int i = 0; i < points.size(); i++) result[i] = points[i];
	count = points.size();
	return static_cast<void*>(result);
}

bool keypoint::KeypointRCNN::deinitialize() const
{
	return true;
}


