#include "com_neptune_api_FaceTool.h"
#include "jni_utils.hpp"
#include "native_api.hpp"

/*
 * Class:     com_neptune_api_FaceTool
 * Method:    init
 * Signature: (Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_neptune_api_FaceTool_init
  (JNIEnv *env, jclass, jstring jstr) {

  return InitEngine(toStr(env, jstr).c_str());
}

/*
 * Class:     com_neptune_api_FaceTool
 * Method:    detect
 * Signature: (Lcom/persist/util/tool/Face/ImageInfo;)Ljava/util/ArrayList;
 */
JNIEXPORT jobject JNICALL Java_com_neptune_api_FaceTool_detect
  (JNIEnv *env, jclass, jobject jimg) {

  return toJava(env, FaceDetect(toMat(env, jimg)));
}

/*
 * Class:     com_neptune_api_FaceTool
 * Method:    extract
 * Signature: (Lcom/persist/util/tool/Face/ImageInfo;)Ljava/util/ArrayList;
 */
JNIEXPORT jobject JNICALL Java_com_neptune_api_FaceTool_extract
  (JNIEnv *env, jclass, jobject jimg) {

  cv::Mat img = toMat(env, jimg);
  std::vector<FaceInfo> infos = FaceDetect(img);
  std::vector<cv::Mat> faces = FaceAlign(img, infos);
  cv::Mat features = FaceExtract(faces);

  return toJava(env, infos, features);
}

/*
 * Class:     com_neptune_api_FaceTool
 * Method:    verify
 * Signature: (Lcom/persist/util/tool/Face/ImageInfo;Lcom/persist/util/tool/Face/ImageInfo;)F
 */
JNIEXPORT jfloat JNICALL Java_com_neptune_api_FaceTool_verify
  (JNIEnv *env, jclass, jobject jimg1, jobject jimg2) {

  return FaceVerify(toMat(env, jimg1), toMat(env, jimg2));
}
