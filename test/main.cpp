#include"../keypoint/src/keypoint_rcnn.hpp"
#include<iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>


int main() {
	// Creation model
	keypoint::KeypointRCNN skeleton("tmp.onnx");
	cv::Mat image;

	//Initialize model
	if (skeleton.initialize()) {
		image = cv::imread("image2.jpg");

		// Run model
		size_t count = 0;
		cv::Point* result = static_cast<cv::Point*>(skeleton.calculate(image, count));

		// Draw result
		for (int i = 0; i < count; i++) {
			cv::circle(image, result[i], 2, cv::Scalar(255, 0, 0));
		}

		// Show result
		cv::namedWindow("image", 0);
		cv::imshow("image", image);
		cv::waitKey(0);
	}

	return 0;
}