#include "center.hpp"

namespace ocean_ai {

	namespace Merge {
		/* direct as features without merge */
		cv::Mat direct(caffe::Blob<float>* out) {
			int num = out->num();
			int len = out->channels();
			const float* data = out->cpu_data();
			return R(cv::Mat(num, len, CV_32FC1, const_cast<float*>(data)));
		}
		/* concatenate mirror features */
		cv::Mat concat(caffe::Blob<float>* out) {
			int num = out->num() / 2;
			int len = out->channels() * 2;
			const float* data = out->cpu_data();
			return R(cv::Mat(num, len, CV_32FC1, const_cast<float*>(data)));
		}
		/* elem-wise add */
		cv::Mat add(caffe::Blob<float>* out) {
			int num = out->num() / 2;
			int len = out->channels();
			const float* data = out->cpu_data();
			cv::Mat merged(num, len, CV_32FC1);
			for (int i = 0; i < num; ++i) {
				float* row = merged.ptr<float>(i);
				for (int j = 0; j < len; ++j)
					row[j] = data[2*i*len + j] + data[(2*i+1)*len + j];
			}
			return R(merged);
		}
		/* elem-wise max */
		cv::Mat max(caffe::Blob<float>* out) {
			int num = out->num() / 2;
			int len = out->channels();
			const float* data = out->cpu_data();
			cv::Mat merged(num, len, CV_32FC1);
			for (int i = 0; i < num; ++i) {
				float* row = merged.ptr<float>(i);
				for (int j = 0; j < len; ++j)
					row[j] = std::max(data[2*i*len + j], data[(2*i+1)*len + j]);
			}
			return R(merged);
		}
		/* elem-wise min */
		cv::Mat min(caffe::Blob<float>* out) {
			int num = out->num() / 2;
			int len = out->channels();
			const float* data = out->cpu_data();
			cv::Mat merged(num, len, CV_32FC1);
			for (int i = 0; i < num; ++i) {
				float* row = merged.ptr<float>(i);
				for (int j = 0; j < len; ++j)
					row[j] = std::min(data[2*i*len + j], data[(2*i+1)*len + j]);
			}
			return R(merged);
		}
	}

	std::function<cv::Mat(caffe::Blob<float>*)> Center::factory(const Center::C_Mirror& mirror) {
		if (mirror.enable) {
			if (mirror.mode == "concat")
				return Merge::concat;
			else if (mirror.mode == "add")
				return Merge::add;
			else if (mirror.mode == "max")
				return Merge::max;
			else // min
				return Merge::min;
		}
		else
			return Merge::direct;
	}

	void Center::setBatchSize(const int batch_size) {
		caffe::Blob<float>* input_layer = net->input_blobs()[0];
		std::vector<int> input_shape = input_layer->shape();
		input_shape[0] = batch_size;
		input_layer->Reshape(input_shape);
		net->Reshape();
	}

	Center::Center(const Center::C_Center& c_center) :
		mirror(c_center.mirror),
		pca(c_center.pca),
		ref_points(c_center.ref_points) {

		/* Load the Net and Model. */
		net = std::make_shared<caffe::Net<float>>(c_center.deploy, caffe::TEST);
		net->CopyTrainedLayersFrom(c_center.model);
		caffe::Blob<float>* input_layer = net->input_blobs()[0];
		face_size.height = input_layer->shape(2);
		face_size.width = input_layer->shape(3);
	}

	float Center::similar(const cv::Mat& features) {
		int len = features.cols;
		float* data = reinterpret_cast<float*>(features.data);
		float norm_a = caffe::caffe_cpu_dot<float>(len, data, data);
		float norm_b = caffe::caffe_cpu_dot<float>(len, data + len, data + len);
		float inp_ab = caffe::caffe_cpu_dot<float>(len, data, data + len);
		return 0.5 + 0.5 * inp_ab / (sqrt(norm_a) * sqrt(norm_b));
	}

	void Center::feed(const cv::Mat &face, int id) {
		/* Feed image into input batch blob. */
		caffe::Blob<float>* input_layer = net->input_blobs()[0];
		int channels = input_layer->channels();
		int data_size = face_size.area() * channels;
		/* head pointer */
		float* input_data = input_layer->mutable_cpu_data() + id * data_size;
		std::vector<cv::Mat> input_channels;
		for (int i = 0; i < channels; ++i) {
			cv::Mat channel(face_size.height, face_size.width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += face_size.area();
		}
		/* Normalization: [0,255] -> [-1, 1] */
		cv::Mat normed;
		face.convertTo(normed, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
		cv::split(normed, input_channels);
	}

	cv::Mat Center::forward(const std::vector<cv::Mat>& faces) {
		int num = faces.size();
		if (num == 0)
			return R(cv::Mat());
		
		if (mirror.enable) {
			setBatchSize(num * 2);
			cv::Mat mirror_face;
			for (int i = 0; i < num; i++) {
				feed(faces[i], 2 * i);
				cv::flip(faces[i], mirror_face, 1);
				feed(mirror_face, 2 * i + 1);
			}
		}
		else { // mirror disable
			setBatchSize(num);
			for (int i = 0; i < num; i++)
				feed(faces[i], i);
		}

		caffe::Blob<float>* out = net->Forward()[0];

		cv::Mat features = mirror.merge(out);
		
		if (pca.enable) {
			features = pca.model.project(features);
		}

		return R(features);
	}

	cv::Mat Center::align(const cv::Mat& image, const FPoints& fpts) {
		cv::Mat face;
		cv::Mat tform = cv::estimateRigidTransform(fpts, ref_points, true);
		if (tform.empty())
			tform = cv::estimateRigidTransform(fpts, ref_points, false);
		cv::warpAffine(image, face, tform, face_size);
		return face;
	}

	float Center::verify(const cv::Mat& image1, const FPoints& fpts1,
	                     const cv::Mat& image2, const FPoints& fpts2) {
		std::vector<cv::Mat> faces;
		faces.push_back(R(align(image1, fpts1)));
		faces.push_back(R(align(image2, fpts2)));
		cv::Mat features = forward(faces);
		return similar(features);
	}

} // ocean_ai