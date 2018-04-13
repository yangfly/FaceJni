#include "mtcnn.hpp"

using namespace std;
using namespace ocean_ai;
using namespace cv;

Rect rect(const BBox& bbox) {
	Rect r;
	r.x = bbox[0];
	r.y = bbox[1];
	r.width = bbox[2] - bbox[0];
	r.height = bbox[3] - bbox[1];
	return R(r);
}

int main() {
	try {
		Config config("config.json");
		// config caffe
		int device = 0;
		if (device < 0) {
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}
		else {
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
			caffe::Caffe::SetDevice(device);
		}

		FLAGS_logtostderr = 1;
		FLAGS_minloglevel = 2;
		::google::InitGoogleLogging("");
		::google::InstallFailureSignalHandler();
		// config logging
		// FLAGS_minloglevel = config.settings.log_level;
		// ::google::InitGoogleLogging("mtcnn");
		// ::google::SetLogDestination(::google::ERROR, "error_");
		// ::google::InstallFailureSignalHandler();

		Mtcnn mtcnn(config.settings.mtcnn);
		Timer timer;

		Mat img = imread("test/test.jpg");
		cv::Mat sample;
		// change image format
		if (img.channels() == 1)
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		else if (img.channels() == 4)
			cv::cvtColor(img, sample, cv::COLOR_RGBA2BGR);
		else
			sample = img;

		sample.convertTo(sample, CV_32FC3);

		vector<FaceInfo> infos;

		for (int i = 0; i < 20; i++) {
			timer.Tic();
			infos = mtcnn.detect(sample);
			timer.Toc();
			cout << "detect " << infos.size() << " use: " << timer.Elasped() << "ms" << endl;
		}
		
		// write detect results
		for (auto info : infos) {
			// cout << info.bbox[0] << " "
			// 		 << info.bbox[1] << " "
			// 		 << info.bbox[2] << " "
			// 		 << info.bbox[3] << " "
			// 		 << info.score   << " "
			// 		 << info.fpts[0].x << " " << info.fpts[0].y << " "
			// 		 << info.fpts[1].x << " " << info.fpts[1].y << " "
			// 		 << info.fpts[2].x << " " << info.fpts[2].y << " "
			// 		 << info.fpts[3].x << " " << info.fpts[3].y << " "
			// 		 << info.fpts[4].x << " " << info.fpts[4].y << endl;
			rectangle(img, rect(info.bbox), Scalar(0, 255, 0));
			for (auto fp : info.fpts) {
				circle(img, fp, 1, Scalar(0, 0, 255));
			}
		}

		imwrite("build/out.jpg", img);
	}
	catch (const std::invalid_argument& ex) {
		cout << "exception: " << ex.what();
	}

	return 0;
}