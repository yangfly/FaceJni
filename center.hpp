#ifndef OCEAN_AI_CENTER_HPP_
#define OCEAN_AI_CENTER_HPP_

#include <vector>
#include <caffe/caffe.hpp>

#include "config.hpp"

namespace ocean_ai {
	// #define USE_OPENMP
	#ifdef USE_OPENMP
	#include <omp.h>
	#define _NUM_THREADS 4
	#endif

	class Center {
	 public:
		using C_Center = Config::Settings::Center;
		using C_Mirror = Config::Settings::Center::Mirror;
		using C_Pca = Config::Settings::Center::Pca;

		// Default constructor.
		Center() {}
		// Real constructor
		Center(const C_Center& c_center);
		// Cos similarity between two features.
		float similar(const cv::Mat& features);
		// Feed: wapper of warpInputLayer.
		// feed image into Net's inpub batch blob with certain id .
		void feed(const cv::Mat& face, int id);
		// Forward multi-faces and get features.
		cv::Mat forward(const std::vector<cv::Mat>& faces);
		// Align image with facial points.
		cv::Mat align(const cv::Mat& image, const FPoints& fpts);
		// Verify between two images
		float verify(const cv::Mat& image1, const FPoints& fpts1,
			const cv::Mat& image2, const FPoints& fpts2);

	 private:
		static std::function<cv::Mat(caffe::Blob<float>*)> factory(const C_Mirror& mirror);
		// Set batch size of network.
		void setBatchSize(const int batch_size);

		// config variables
		std::shared_ptr<caffe::Net<float> > net;
		struct Mirror {
			bool enable;
			std::function<cv::Mat(caffe::Blob<float>*)> merge;
			Mirror() {}
			Mirror(const C_Mirror& c_mirror) :
				enable(c_mirror.enable) {
				merge = factory(c_mirror);
			}
		} mirror;
		struct Pca {
			bool enable;
			cv::PCA model;
			Pca() {}
			Pca(const C_Pca& c_pca) :
				enable(c_pca.enable) {
				/* Code from: www.bytefish.de/blog/pca_in_opencv */
				if (enable) {
					cv::FileStorage fs(c_pca.model, cv::FileStorage::READ);
					model.read(fs.root());
					fs.release();
				}
			}
		} pca;
		FPoints ref_points;
		// tool variables
		cv::Size face_size;
	};

} // ocean_ai

#endif // OCEAN_AI_CENTER_HPP_