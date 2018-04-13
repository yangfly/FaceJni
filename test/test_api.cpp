#include "native_api.hpp"

using namespace std;
using namespace cv;
using namespace ocean_ai;

Rect rect(const BBox& bbox) {
  Rect r;
  r.x = bbox[0];
  r.y = bbox[1];
  r.width = bbox[2] - bbox[0];
  r.height = bbox[3] - bbox[1];
  return R(r);
}

void imdraw(Mat& img, const vector<FaceInfo>& infos) {
  for (auto info : infos) {
    rectangle(img, rect(info.bbox), Scalar(0, 255, 0));
    for (auto fp : info.fpts) {
      circle(img, fp, 1, Scalar(0, 0, 255));
    }
  }
}

int main() {
  if(InitEngine("config.json"))
    cout << "Init inference engine successfully." << endl;
  else
    cout << "Failed to init inference engine." << endl;
  Timer timer;
  // face detect
  Mat image0 = imread("test/test2.jpg");
  timer.Tic();
  vector<FaceInfo> infos0 = FaceDetect(image0);
  timer.Toc();
  cout << "detect " << infos0.size() << " use: " << timer.Elasped() << "ms" << endl;
  imdraw(image0, infos0);
  imwrite("build/detect.jpg", image0);
  // face verify
  Mat image1 = imread("test/cdy_cdy_0_01.jpg");
  Mat image2 = imread("test/cdy_cdy_0_02.jpg");
  // Mat image2 = imread("test/test2.jpg");
  for (int i = 0; i < 3; i++) {
    timer.Tic();
    FPoints fpts1 = FaceDetect(image1)[0].fpts;
    FPoints fpts2 = FaceDetect(image2)[0].fpts;
    cout << "similarity: " << FaceVerify(image1, fpts1, image2, fpts2) << endl;
    timer.Toc();
    cout << "verify1 use: " << timer.Elasped() << "ms" << endl;
    timer.Tic();
    cout << "similarity: " << FaceVerify(image1, image2) << endl;
    timer.Toc();
    cout << "verify2 use: " << timer.Elasped() << "ms" << endl;
  }
  
  // face search
  Mat image3 = imread("test/test2.jpg");
  timer.Tic();
  vector<FaceInfo> infos = FaceDetect(image3);
  vector<Mat> faces = FaceAlign(image3, infos);
  Mat features = FaceExtract(faces);
  timer.Toc();
  cout << "extract use: " << timer.Elasped() << "ms" << endl;
  cout << "features shape: " << features.rows << " x " << features.cols << endl; 

  return 0;
}