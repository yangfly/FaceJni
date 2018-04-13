#include "mtcnn.hpp"
#include "center.hpp"

using namespace std;
using namespace ocean_ai;
using namespace cv;

Mat imread(const char* filename) {
	Mat img = cv::imread(filename);
	cv::Mat sample;
	// change image format
	if (img.channels() == 1)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else if (img.channels() == 4)
		cv::cvtColor(img, sample, cv::COLOR_RGBA2BGR);
	else
		sample = img;
	sample.convertTo(sample, CV_32FC3);

	return R(sample);
}

#include <fstream>
void dumps(const char* filename, const Mat& feature) {
	int num = feature.rows;
	int len = feature.cols;
	ofstream out(filename);
	for (int i = 0; i < num; i++) {
		const float* row = feature.ptr<float>(i);
		for (int j = 0; j < len; j++)
			out << row[j] << " ";
		out << endl;
	}
	out.close();
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

		Mtcnn mtcnn(config.settings.mtcnn);
		Center center(config.settings.center);
		Timer timer;

		if (config.settings.center.mode == "verify") {
			Mat sample1 = imread("test/cdy_cdy_0_01.jpg");
			Mat sample2 = imread("test/cdy_cdy_0_02.jpg");
			float score1, score2;
			for (int i = 0; i < 10; i++) {
				timer.Tic();
				FPoints fpts1 = mtcnn.detect(sample1)[0].fpts;
				FPoints fpts2 = mtcnn.detect(sample2)[0].fpts;
				score1 = center.verify(sample1, fpts1, sample2, fpts2);
				timer.Toc();
				cout << "verify use: " << timer.Elasped() << "ms" << endl;
				Mat face1 = center.align(sample1, fpts1);
				Mat face2 = center.align(sample2, fpts2);
				score2 = center.verify(face1, face2);
			}
			cout << "similarity: " << score1 << " " << score2 << endl;
		}
		else {  // search
			Mat sample = imread("test/test2.jpg");
			Mat features;
			for (int i = 0; i < 10; i++) {
				timer.Tic();
				vector<FaceInfo> infos = mtcnn.detect(sample);
				timer.Toc();
				cout << "detect " << infos.size() << " use: " << timer.Elasped() << "ms" << endl;
				timer.Tic();
				vector<Mat> faces;
				for (auto info : infos)
					faces.push_back(center.align(sample, info.fpts));
				timer.Toc();
				cout << "align " << infos.size() << " use: " << timer.Elasped() << "ms" << endl;
				timer.Tic();
				features = center.forward(faces);
				timer.Toc();
				cout << "forward " << infos.size() << " use: " << timer.Elasped() << "ms" << endl;
			}
			dumps("build/cpp.txt", features);
		}
	}
	catch (const std::invalid_argument& ex) {
		cout << "exception: " << ex.what();
	}

	return 0;
}