#include "mtcnn.hpp"

namespace ocean_ai {

	Mtcnn::Mtcnn(const Mtcnn::C_Mtcnn& c_mtcnn) :
		model_dir(c_mtcnn.model_dir),
		factor(c_mtcnn.factor),
		min_size(c_mtcnn.min_size),
		thresholds(c_mtcnn.thresholds),
		precise_landmark(c_mtcnn.precise_landmark),
		limitation(c_mtcnn.limitation) {
		loadModels(model_dir);
	}

	void Mtcnn::loadModels(const std::string& model_dir)
	{
		Pnet = std::make_shared<caffe::Net<float>>(model_dir + "/det1.prototxt", caffe::TEST);
		Pnet->CopyTrainedLayersFrom(model_dir + "/det1.caffemodel");
		Rnet = std::make_shared<caffe::Net<float>>(model_dir + "/det2.prototxt", caffe::TEST);
		Rnet->CopyTrainedLayersFrom(model_dir + "/det2.caffemodel");
		Onet = std::make_shared<caffe::Net<float>>(model_dir + "/det3.prototxt", caffe::TEST);
		Onet->CopyTrainedLayersFrom(model_dir + "/det3.caffemodel");
		if (precise_landmark) {
			Lnet = std::make_shared<caffe::Net<float>>(model_dir + "/det4.prototxt", caffe::TEST);
			Lnet->CopyTrainedLayersFrom(model_dir + "/det4.caffemodel");
		}
	}

	void Mtcnn::setBatchSize(std::shared_ptr<caffe::Net<float>> net, const int batch_size)
	{
		caffe::Blob<float>* input_layer = net->input_blobs()[0];
		std::vector<int> input_shape = input_layer->shape();
		input_shape[0] = batch_size;
		input_layer->Reshape(input_shape);
		net->Reshape();
	}

	std::vector<std::vector<cv::Mat> > Mtcnn::warpInputLayer(std::shared_ptr<caffe::Net<float>> net)
	{
		std::vector<std::vector<cv::Mat> > input_channals;
		caffe::Blob<float>* input_layer = net->input_blobs()[0];
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();

		for (int i = 0; i < input_layer->num(); ++i)
		{
			std::vector<cv::Mat> channals;
			for (int j = 0; j < input_layer->channels(); ++j)
			{
				channals.emplace_back(height, width, CV_32FC1, input_data);
				input_data += width * height;
			}
			input_channals.push_back(std::move(channals));
		}

		return R(input_channals);
	}

	std::vector<float> Mtcnn::scalePyramid(const int height, const int width)
	{
		std::vector<float> scales;
		int min_len = std::min(height, width);
		int max_len = std::max(height, width);
		float max_scale = 12.0f / min_size;
		float min_scale = 12.0f / min_len;
		if (limitation.enable && limitation.size < max_len)
			max_scale *= limitation.size * 1.0f / max_len;
		for (float scale = max_scale; scale >= min_scale; scale *= factor)
			scales.push_back(scale);

		return R(scales);
	}

	std::vector<Proposal> Mtcnn::getCandidates(
		const float scale, const caffe::Blob<float>* scores, const caffe::Blob<float>* regs)
	{
		int stride = 2;
		int cell_size = 12;
		int output_width = regs->width();
		int output_height = regs->height();
		std::vector<Proposal> pros;

		for (int i = 0; i < output_height; ++i)
			for (int j = 0; j < output_width; ++j)
				if (scores->data_at(0, 1, i, j) >= thresholds[0])
				{
					// bounding box
					BBox bbox;
					bbox[0] = j * stride / scale;	// x1
					bbox[1] = i * stride / scale;	// y1
					bbox[2] = (j * stride + cell_size - 1) / scale + 1;	// x2
					bbox[3] = (i * stride + cell_size - 1) / scale + 1;	// y2
					// bbox regression
					Reg reg(
						regs->data_at(0, 0, i, j),	// reg_x1
						regs->data_at(0, 1, i, j),	// reg_y1
						regs->data_at(0, 2, i, j),	// reg_x2
						regs->data_at(0, 3, i, j));	// reg_y2
					// face confidence
					float score = scores->data_at(0, 1, i, j);
					pros.emplace_back(R(bbox), score, R(reg));
				}

		return R(pros);
	}

	std::vector<Proposal> Mtcnn::NonMaximumSuppression(std::vector<Proposal>& pros,
		const float threshold, const NMS_TYPE type)
	{
		if (pros.size() <= 1)
			return pros;

		std::sort(pros.begin(), pros.end(),
			// Lambda function: descending order by score.
			[](const Proposal& x, const Proposal& y) -> bool { return x.score > y.score; });


		std::vector<Proposal> nms_pros;
		while (!pros.empty()) {
			// select maximun candidates.
			Proposal max = R(pros[0]);
			pros.erase(pros.begin());
			float max_area = (max.bbox[2] - max.bbox[0])
				* (max.bbox[3] - max.bbox[1]);
			// filter out overlapped candidates in the rest.
			int idx = 0;
			while (idx < pros.size()) {
				// computer intersection.
				float x1 = std::max(max.bbox[0], pros[idx].bbox[0]);
				float y1 = std::max(max.bbox[1], pros[idx].bbox[1]);
				float x2 = std::min(max.bbox[2], pros[idx].bbox[2]);
				float y2 = std::min(max.bbox[3], pros[idx].bbox[3]);
				float overlap = 0;
				if (x1 < x2 && y1 < y2)
				{
					float inter = (x2 - x1) * (y2 - y1);
					// computer denominator.
					float outer;
					float area = (pros[idx].bbox[2] - pros[idx].bbox[0])
						* (pros[idx].bbox[3] - pros[idx].bbox[1]);
					if (type == IoM)	// Intersection over Minimum
						outer = std::min(max_area, area);
					else	// Intersection over Union
						outer = max_area + area - inter;
					overlap = inter / outer;
				}

				if (overlap > threshold)	// erase overlapped candidate
					pros.erase(pros.begin() + idx);
				else
					idx++;	// check next candidate
			}
			nms_pros.push_back(R(max));
		}

		return R(nms_pros);
	}

	void Mtcnn::boxRegression(std::vector<Proposal>& pros)
	{
		for (auto& pro : pros) {
			float width = pro.bbox[2] - pro.bbox[0];
			float height = pro.bbox[3] - pro.bbox[1];
			pro.bbox[0] += pro.reg[0] * width;	// x1
			pro.bbox[1] += pro.reg[1] * height;	// y1
			pro.bbox[2] += pro.reg[2] * height;	// x2
			pro.bbox[3] += pro.reg[3] * height;	// y2
		}
	}

	void Mtcnn::square(std::vector<BBox> & bboxes) {
		for (auto& bbox : bboxes)
			square(bbox);
	}

	void Mtcnn::square(BBox & bbox) {
		float x1 = std::floor(bbox[0]);
		float y1 = std::floor(bbox[1]);
		float x2 = std::ceil(bbox[2]);
		float y2 = std::ceil(bbox[3]);
		int diff = static_cast<int>((x2 - x1) - (y2 - y1));
		if (diff > 0) {	// width > height
			y1 -= diff / 2;
			y2 += diff / 2;
			if (diff % 2 != 0) {
				if ((bbox[1] - y1) < (y2 - bbox[3]))
					y1 -= 1;
				else
					y2 += 1;
			}
		}
		else if (diff < 0) {	// height < width
			diff = -diff;
			x1 -= diff / 2;
			x2 += diff / 2;
			if (diff % 2 != 0) {
				if ((bbox[0] - x1) < (x2 - bbox[2]))
					x1 -= 1;
				else
					x2 += 1;
			}
		}
		bbox[0] = x1;
		bbox[1] = y1;
		bbox[2] = x2;
		bbox[3] = y2;
	}

	cv::Mat Mtcnn::cropPadding(const cv::Mat& sample, const BBox& bbox)
	{
		cv::Rect img_rect(0, 0, sample.cols, sample.rows);
		cv::Rect crop_rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
		cv::Rect inter_on_sample = crop_rect & img_rect;
		// shifting inter from image CS (coordinate system) to crop CS.
		cv::Rect inter_on_crop = inter_on_sample - crop_rect.tl();

		cv::Mat crop(crop_rect.size(), CV_32FC3, cv::Scalar(0.0));
		sample(inter_on_sample).copyTo(crop(inter_on_crop));

		return R(crop);
	}

	std::vector<BBox> Mtcnn::ProposalNetwork(const cv::Mat & sample)
	{
		std::vector<float> scales = scalePyramid(sample.rows, sample.cols);
		std::vector<Proposal> total_pros;

		caffe::Blob<float>* input_layer = Pnet->input_blobs()[0];
		for (float scale : scales)
		{
			int height = static_cast<int>(std::ceil(sample.rows * scale));
			int width = static_cast<int>(std::ceil(sample.cols * scale));
			cv::Mat img;
			cv::resize(sample, img, cv::Size(width, height));
#ifndef NORM_FARST
			img.convertTo(img, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
#endif // !NORM_FARST

			// Reshape Net.
			input_layer->Reshape(1, 3, height, width);
			Pnet->Reshape();

			std::vector<cv::Mat> channals = warpInputLayer(Pnet)[0];
			cv::split(img, channals);
			const std::vector<caffe::Blob<float>*> out = Pnet->Forward();
			std::vector<Proposal> pros = getCandidates(scale, out[0], out[1]);

			// intra scale nms
			pros = NonMaximumSuppression(pros, 0.5f, IoU);

			if (!pros.empty()) {
				total_pros.insert(total_pros.end(), pros.begin(), pros.end());
			}
		}
		// inter scale nms
		total_pros = NonMaximumSuppression(total_pros, 0.7f, IoU);	
		boxRegression(total_pros);

		std::vector<BBox> bboxes;
		for (auto& pro : total_pros)
			bboxes.push_back(R(pro.bbox));

		return R(bboxes);
	}

	std::vector<BBox> Mtcnn::RefineNetwork(const cv::Mat & sample, std::vector<BBox> & bboxes)
	{
		if (bboxes.empty())
			return R(bboxes);

		size_t num = bboxes.size();
		square(bboxes);	// convert bbox to square
		setBatchSize(Rnet, num);
		std::vector<std::vector<cv::Mat> > input_channals = warpInputLayer(Rnet);
		for (int i = 0; i < num; ++i)
		{
			cv::Mat crop = cropPadding(sample, bboxes[i]);
			cv::resize(crop, crop, cv::Size(24, 24));
#ifndef NORM_FARST
			crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
#endif // NORM_FARST
			cv::split(crop, input_channals[i]);
		}

		const std::vector<caffe::Blob<float>*> out = Rnet->Forward();
		caffe::Blob<float>* scores = out[0];
		caffe::Blob<float>* regs = out[1];

		std::vector<Proposal> pros;
		for (int i = 0; i < num; ++i)
		{
			if (scores->data_at(i, 1, 0, 0) >= thresholds[1]) {
				Reg reg(regs->data_at(i, 0, 0, 0),	// x1
					regs->data_at(i, 1, 0, 0),	// y1
					regs->data_at(i, 2, 0, 0),	// x2
					regs->data_at(i, 3, 0, 0));	// y2
				float score = scores->data_at(i, 1, 0, 0);
				pros.emplace_back(R(bboxes[i]), score, R(reg));
			}
		}

		pros = NonMaximumSuppression(pros, 0.7f, IoU);
		boxRegression(pros);

		bboxes.clear();
		for (auto& pro : pros)
			bboxes.push_back(R(pro.bbox));

		return R(bboxes);
	}

	std::vector<FaceInfo> Mtcnn::OutputNetwork(const cv::Mat & sample, std::vector<BBox> & bboxes)
	{
		std::vector<FaceInfo> infos;
		if (bboxes.empty())
			return infos;

		size_t num = bboxes.size();
		square(bboxes);	// convert bbox to square

		setBatchSize(Onet, num);
		std::vector<std::vector<cv::Mat> > input_channals = warpInputLayer(Onet);
		for (int i = 0; i < num; ++i)
		{
			cv::Mat crop = cropPadding(sample, bboxes[i]);
			cv::resize(crop, crop, cv::Size(48, 48));
#ifndef NORM_FARST
			crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
#endif // NORM_FARST
			cv::split(crop, input_channals[i]);
		}

		const std::vector<caffe::Blob<float>*> out = Onet->Forward();
		caffe::Blob<float>* fpts = out[0];
		caffe::Blob<float>* scores = out[1];
		caffe::Blob<float>* regs = out[2];

		std::vector<Proposal> pros;
		for (int i = 0; i < num; ++i)
		{
			BBox& bbox = bboxes[i];
			if (scores->data_at(i, 1, 0, 0) >= thresholds[2]) {
				Reg reg(regs->data_at(i, 0, 0, 0),	// x1
					regs->data_at(i, 1, 0, 0),	// y1
					regs->data_at(i, 2, 0, 0),	// x2
					regs->data_at(i, 3, 0, 0));	//y2
				float score = scores->data_at(i, 1, 0, 0);
				// facial landmarks
				FPoints fpt;
				float width = bbox[2] - bbox[0];
				float height = bbox[3] - bbox[1];
				for (int j = 0; j < 5; j++)
					fpt.emplace_back(
					fpts->data_at(i, 2*j, 0, 0) * width + bbox[0],
					fpts->data_at(i, 2*j+1, 0, 0) * height + bbox[1]);
				pros.emplace_back(R(bbox), score, R(fpt), R(reg));
			}
		}

		pros = NonMaximumSuppression(pros, 0.7f, IoM);
		boxRegression(pros);

		for (auto& pro : pros)
			infos.emplace_back(R(pro.bbox), pro.score, R(pro.fpts));

		return R(infos);
	}

	void Mtcnn::LandmarkNetwork(const cv::Mat & sample,
		std::vector<FaceInfo>& infos)
	{
		if (infos.empty())
			return;

		size_t num = infos.size();
		setBatchSize(Lnet, num);
		cv::Rect img_rect(0, 0, sample.cols, sample.rows);
		std::vector<std::vector<cv::Mat> > input_channals = warpInputLayer(Lnet);
		cv::Mat patch_sizes(num, 5, CV_32S);
		for (int i = 0; i < num; ++i)
		{
			FaceInfo& info = infos[i];
			float patchw = std::max(info.bbox[2] - info.bbox[0],
				info.bbox[3] - info.bbox[1]);
			float patchs = patchw * 0.25f;
			for (int j = 0; j < 5; ++j)
			{
				BBox patch;
				patch[0] = info.fpts[j].x - patchs * 0.5f;
				patch[1] = info.fpts[j].y - patchs * 0.5f;
				patch[2] = info.fpts[j].x + patchs * 0.5f;
				patch[3] = info.fpts[j].y + patchs * 0.5f;
				square(patch);
				patch_sizes.at<int>(i, j) = patch[2] - patch[0];
				
				cv::Mat crop = cropPadding(sample, patch);
				cv::resize(crop, crop, cv::Size(24, 24));
#ifndef NORM_FARST
				crop.convertTo(crop, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
#endif // NORM_FARST

				// extract channels of certain patch
				std::vector<cv::Mat> patch_channels;
				patch_channels.push_back(input_channals[i][3 * j + 0]);	// B
				patch_channels.push_back(input_channals[i][3 * j + 1]);	// G
				patch_channels.push_back(input_channals[i][3 * j + 2]);	// R
				cv::split(crop, patch_channels);
			}
		}

		const std::vector<caffe::Blob<float>*> out = Lnet->Forward();

		// for every facial landmark
		// Caffe output blobs are disordered.
		std::vector<int> fpt_order = {0, 3, 2, 1, 4};
		for (int j = 0; j < 5; ++j)
		{
			caffe::Blob<float>* offs = out[fpt_order[j]];
			// for every face
			for (int i = 0; i < num; ++i)
			{
				int patch_size = patch_sizes.at<int>(i,j);
				float off_x = offs->data_at(i, 0, 0, 0) - 0.5;
				float off_y = offs->data_at(i, 1, 0, 0) - 0.5;
				// Dot not makeoffrge movement with relative offset > 0.35
				if (std::fabs(off_x) <= 0.35 && std::fabs(off_y) <= 0.35)
				{
					infos[i].fpts[j].x += off_x * patch_size;
					infos[i].fpts[j].y += off_y * patch_size;
				}
			}
		}
	}

	std::vector<FaceInfo> Mtcnn::detect(const cv::Mat & sample)
	{
	#ifdef NORM_FARST
		cv::Mat normed_sample;
		sample.convertTo(normed_sample, CV_32FC3, 0.0078125, -127.5 * 0.0078125);
	#else
		const cv::Mat& normed_sample = sample;
	#endif // NORM_FARST	

		std::vector<BBox> bboxes = ProposalNetwork(normed_sample);
		bboxes = RefineNetwork(normed_sample, bboxes);
		std::vector<FaceInfo> infos = OutputNetwork(normed_sample, bboxes);
		if (precise_landmark)
			LandmarkNetwork(normed_sample, infos);
		return R(infos);
	}

}