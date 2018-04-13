#ifndef OCEAN_AI_FACE_CONTEXT_HPP_
#define OCEAN_AI_FACE_CONTEXT_HPP_

#include "context.hpp"
#include "mtcnn.hpp"
#include "center.hpp"

#include <cuda_runtime.h>

namespace ocean_ai {

	class FaceContext
	{
	public:
		friend ScopedContext<FaceContext>;

		static bool IsCompatible(int device)
		{
			cudaError_t st = cudaSetDevice(device);
			if (st != cudaSuccess)
				return false;

			cv::cuda::DeviceInfo info;
			if (!info.isCompatible())
				return false;

			return true;
		}

		FaceContext(const Config& config, int device) :
			device_(device),
			enable_detect_(config.options.detection),
			enable_recog_(config.options.recognition) {

			cudaError_t st = cudaSetDevice(device_);
			if (st != cudaSuccess)
				throw std::invalid_argument("could not set CUDA device");

			caffe_context_.reset(new caffe::Caffe);
			caffe::Caffe::Set(caffe_context_.get());

			if (enable_detect_)
				mtcnn_.reset(new Mtcnn(config.settings.mtcnn));
			if (enable_recog_)
				center_.reset(new Center(config.settings.center));

			caffe::Caffe::Set(nullptr);
		}

		Mtcnn* mtcnn()
		{
			return mtcnn_.get();
		}

		Center* center() {
			return center_.get();
		}

	private:
		void Activate()
		{
			cudaError_t st = cudaSetDevice(device_);
			if (st != cudaSuccess)
				throw std::invalid_argument("could not set CUDA device");
			caffe::Caffe::Set(caffe_context_.get());
		}

		void Deactivate()
		{
			caffe::Caffe::Set(nullptr);
		}

	public:
		bool enable_detect_;
		bool enable_recog_;

	private:
		int device_;
		std::unique_ptr<caffe::Caffe> caffe_context_;
		std::unique_ptr<Mtcnn> mtcnn_;
		std::unique_ptr<Center> center_;
	};

}	// ocean_ai


#endif // OCEAN_AI_FACE_CONTEXT_HPP_