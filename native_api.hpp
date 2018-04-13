#ifndef OCEAN_AI_NATIVE_API_HPP_
#define OCEAN_AI_NATIVE_API_HPP_

// gencode sm
// Titan sm52/sm50
// 10x0 sm61

#include "common.hpp"

namespace ocean_ai {

	// Init caffe context
	bool InitEngine(const char* config_path);

	// Face detection
	std::vector<FaceInfo> FaceDetect(const cv::Mat& image);

	// Face alignments
	cv::Mat FaceAlign(const cv::Mat& image, const FPoints& fpts);
	std::vector<cv::Mat> FaceAlign(const cv::Mat& image, std::vector<FaceInfo> infos);

	// Extract face feature
	cv::Mat FaceExtract(const cv::Mat& image);
	cv::Mat FaceExtract(const std::vector<cv::Mat>& faces);

	// Verify two faces
	float FaceVerify(const cv::Mat& image1, const cv::Mat& image2);
	float FaceVerify(const cv::Mat& image1, const FPoints& fpts1,
	                 const cv::Mat& image2, const FPoints& fpts2);

} // ocean_ai


#endif // OCEAN_AI_NATIVE_API_HPP_