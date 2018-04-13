#ifndef OCEAN_AI_MTCNN_HPP_
#define OCEAN_AI_MTCNN_HPP_

#include <vector>
#include <caffe/caffe.hpp>

#include "config.hpp"

namespace ocean_ai {

	// Only do normalization once
	// before any image processing to save time.
	// #define NORM_FARST

	// face regression: x, y, size
	typedef cv::Vec4f Reg;

	class Proposal : public FaceInfo {
	 public:
		Reg reg;	// face regression
		Proposal(BBox&& bbox, float score, Reg&& reg)
			: FaceInfo(R(bbox), R(score)), 
				reg(R(reg)) {
		}
		Proposal(BBox&& bbox, float score, FPoints&& fpts, Reg&& reg)
			: FaceInfo(R(bbox), R(score), R(fpts)),
				reg(R(reg)) {
		}
	};

	class Mtcnn {
	 public:
		enum NMS_TYPE {
			IoM,	// Intersection over Union
			IoU		// Intersection over Minimum
		};

		using C_Mtcnn = Config::Settings::Mtcnn;
		using Limitation = C_Mtcnn::Limitation;

		// Default constructor.
		Mtcnn() {};
		// Real constructor
		Mtcnn(const C_Mtcnn& c_mtcnn);
		// Init four networks and load trained weights.
		void loadModels(const std::string& model_dir);
		// Set batch size of network.
		void setBatchSize(std::shared_ptr<caffe::Net<float> > net, const int batch_size);
		// Warp whole input layer into cv::Mat channels.
		std::vector<std::vector<cv::Mat> > warpInputLayer(std::shared_ptr<caffe::Net<float> > net);
		// Create scale pyramid: down order
		std::vector<float> scalePyramid(const int height, const int width);
		// Get bboxes from maps of confidences and regressions.
		std::vector<Proposal> getCandidates(const float scale,
			const caffe::Blob<float>* regs, const caffe::Blob<float>* scores);
		// Non Maximum Supression with type 'IoU' or 'IoM'.
		std::vector<Proposal> NonMaximumSuppression(std::vector<Proposal>& pros,
			const float threshold, const NMS_TYPE type);
		// Refine bounding box with regression.
		void boxRegression(std::vector<Proposal>& pros);
		// Convert bbox from float bbox to square
		void square(std::vector<BBox> & bboxes);
		void square(BBox & bbox);

		// Crop proposals with padding 0.
		cv::Mat cropPadding(const cv::Mat& sample, const BBox& bbox);

		// Stage 1: Pnet get proposal bounding boxes
		std::vector<BBox> ProposalNetwork(const cv::Mat& sample);
		// Stage 2: Rnet refine and reject proposals
		std::vector<BBox> RefineNetwork(const cv::Mat& sample, std::vector<BBox>& bboxes);
		// Stage 3: Onet refine and reject proposals and regress facial landmarks.
		std::vector<FaceInfo> OutputNetwork(const cv::Mat& sample, std::vector<BBox>& bboxes);
		// Stage 4: Lnet refine facial landmarks
		void LandmarkNetwork(const cv::Mat& sample, std::vector<FaceInfo>& infos);

		// Detect faces from images
		std::vector<FaceInfo> detect(const cv::Mat & sample);
	
	 private:
		// configures 
		std::string model_dir;
		float factor;
		int min_size;
		cv::Vec3f thresholds;
		bool precise_landmark;
		Limitation limitation;
		// networks
		std::shared_ptr<caffe::Net<float> > Pnet;
		std::shared_ptr<caffe::Net<float> > Rnet;
		std::shared_ptr<caffe::Net<float> > Onet;
		std::shared_ptr<caffe::Net<float> > Lnet;
	};

} // ocean_ai

#endif // OCEAN_AI_MTCNN_HPP_