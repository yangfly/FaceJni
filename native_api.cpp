#include <caffe/caffe.hpp>  //? Why must include guy!
#include "native_api.hpp"
#include "face_context.hpp"

#include <iostream>
using namespace std;

namespace ocean_ai {

ContextPool<FaceContext> pool;

	bool InitEngine(const char* config_path) {
		try {
			Config config = Config(config_path);
			// config logging
			FLAGS_logtostderr = 0;
			FLAGS_minloglevel = config.settings.glog.level;
			FLAGS_log_dir = config.settings.glog.dir;
			::google::InitGoogleLogging("api");
			::google::InstallFailureSignalHandler();

			int device_count;
			cudaError_t st = cudaGetDeviceCount(&device_count);
			if (st != cudaSuccess)
				throw std::invalid_argument("could not list CUDA devices");

			for (int dev = 0; dev < device_count; ++dev) {
				if (!FaceContext::IsCompatible(dev)) {
					LOG(ERROR) << "Skipping device: " << dev;
					continue;
				}

				for (int i = 0; i < config.settings.K_ctx_per_GPU; ++i){
					std::unique_ptr<FaceContext> context(new FaceContext(config, dev));
					LOG(WARNING) << "Initialize face context " << i << " on GPU " << dev;
					pool.Push(std::move(context));
				}
			 }

			if (pool.Size() == 0)
				throw std::invalid_argument("no suitable CUDA device");
			return true;

		}
		catch (const std::invalid_argument& ex) {
				LOG(ERROR) << "exception: " << ex.what();
				return false;
		}
	} 

	cv::Mat format(const cv::Mat& image) {
		cv::Mat sample;
		// change image format
		if (image.channels() == 1)
			cv::cvtColor(image, sample, cv::COLOR_GRAY2BGR);
		else if (image.channels() == 4)
			cv::cvtColor(image, sample, cv::COLOR_RGBA2BGR);
		else
			sample = image;
		sample.convertTo(sample, CV_32FC3);

		return R(sample);
	}

	std::vector<FaceInfo> FaceDetect(const cv::Mat& image) {
		try {
			cv::Mat sample = format(image);

			{
				/* In this scope an execution context is acquired for inference and it
				 * will be automatically released back to the context pool when
				 * exiting this scope. 
				 */
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_detect_)
					throw std::invalid_argument("detection option is disable when call face detection.");

				return R(context->mtcnn()->detect(sample));
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return R(std::vector<FaceInfo>());
		}
	}

	cv::Mat FaceAlign(const cv::Mat& image, const FPoints& fpts) {
		try {
			cv::Mat sample = format(image);

			{
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face alignment.");

				return R(context->center()->align(sample, fpts));
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return R(cv::Mat());
		}
	}

	std::vector<cv::Mat> FaceAlign(const cv::Mat& image, std::vector<FaceInfo> infos) {
		try {
			cv::Mat sample = format(image);

			{
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face alignment.");

				std::vector<cv::Mat> faces;
				Center* center = context->center();
				for (auto info : infos) {
					faces.push_back(R(center->align(sample, info.fpts)));
				}
				return R(faces);
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return R(std::vector<cv::Mat>());
		}
	}

	cv::Mat FaceExtract(const cv::Mat& image) {
		try {
			cv::Mat sample = format(image);

			{
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face extraction.");

				std::vector<FaceInfo> infos = context->mtcnn()->detect(sample);
				Center* center = context->center();
				std::vector<cv::Mat> faces;
				for (auto info : infos) {
					faces.push_back(R(center->align(sample, info.fpts)));
				}
				cv::Mat features = center->forward(faces);
				return R(features);
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return R(cv::Mat());
		}
	}

	cv::Mat FaceExtract(const std::vector<cv::Mat>& faces) {
		try {
			{
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face extraction.");

				cv::Mat features = context->center()->forward(faces);
				return R(features);
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return R(cv::Mat());
		}
	}

	float FaceVerify(const cv::Mat& image1, const cv::Mat& image2) {
		try {
			{
				cv::Mat sample1 = format(image1);
				cv::Mat sample2 = format(image2);

				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face verification.");
				Mtcnn* mtcnn = context->mtcnn();
				FPoints fpts1 = mtcnn->detect(sample1)[0].fpts;
				FPoints fpts2 = mtcnn->detect(sample2)[0].fpts;

				return context->center()->verify(sample1, fpts1, sample2, fpts2);
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return -1;
		}
	}

	float FaceVerify(const cv::Mat& image1, const FPoints& fpts1,
									 const cv::Mat& image2, const FPoints& fpts2) {
		try {
			{
				cv::Mat sample1 = format(image1);
				cv::Mat sample2 = format(image2);
				ScopedContext<FaceContext> context(pool);
				if (!context->enable_recog_)
					throw std::invalid_argument("recognition option is disable when call face verification.");

				return context->center()->verify(sample1, fpts1, sample2, fpts2);
			}
		}
		catch (const std::invalid_argument& ex)
		{
			LOG(ERROR) << "exception: " << ex.what();
			return -1;}
	}

} // ocean_ai