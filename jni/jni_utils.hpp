#ifndef JNI_UTILS_HPP_
#define JNI_UTILS_HPP_

#include <jni.h>
#include "common.hpp"

using namespace ocean_ai;

std::string toStr(JNIEnv* env, const jstring& jstr) {
  char* cstr = nullptr;
  jclass jstr_class = env->FindClass("java/lang/String");
  jstring jstr_encode = env->NewStringUTF("utf-8");
  jmethodID mid = env->GetMethodID(jstr_class, "getBytes", "(Ljava/lang/String;)[B");
  jbyteArray jba = (jbyteArray)env->CallObjectMethod(jstr, mid, jstr_encode);
  jsize len = env->GetArrayLength(jba);
  jbyte* jbp = env->GetByteArrayElements(jba, JNI_FALSE);
  if (len > 0) {
    cstr = new char[len + 1];
    memcpy(cstr, jbp, len);
    cstr[len] = '\0';
  }
  env->ReleaseByteArrayElements(jba, jbp, 0);
  std::string str(cstr);
  delete[] cstr;
  return R(str);
}

cv::Mat toMat(JNIEnv* env, const jobject& jimg) {
  jclass img_class = env->GetObjectClass(jimg);
  jfieldID img_fid;
  img_fid = env->GetFieldID(img_class, "pixels", "[B");
  jbyteArray data = (jbyteArray)env->GetObjectField(jimg, img_fid);
  img_fid = env->GetFieldID(img_class, "width", "I");
  jint width = env->GetIntField(jimg, img_fid);
  img_fid = env->GetFieldID(img_class, "height", "I");
  jint height = env->GetIntField(jimg, img_fid);

  jsize len = env->GetArrayLength(data);
  if (len == 0)
    return R(cv::Mat());

  jbyte* jbp = env->GetByteArrayElements(data, JNI_FALSE);
  cv::Mat img = cv::Mat(height, width, CV_8UC3, jbp);

  cv::Mat new_img = img.clone();

  env->ReleaseByteArrayElements(data, jbp, 0);

  return R(new_img);
}

jobject toJava(JNIEnv* env, const FaceInfo& info) {
  jfloatArray jf_array = env->NewFloatArray(15);

  float* temp = new float[15];
  for (int i = 0; i < 4; i++)
    temp[i] = info.bbox[i];
  temp[4] = info.score;
  for (int i = 0; i < 5; i++) {
    temp[2*i+5] = info.fpts[i].x;
    temp[2*i+6] = info.fpts[i].y;
  }

  env->SetFloatArrayRegion(jf_array, 0, 15, temp);
  delete[] temp;

  jclass info_class = env->FindClass("Lcom/neptune/utils/FaceInfo;");
  jmethodID info_construct = env->GetMethodID(info_class, "<init>", "([F)V");
  jobject j_info = env->NewObject(info_class, info_construct, jf_array);
  
  return R(j_info);
}

jobject toJava(JNIEnv* env, const FaceInfo& info, const cv::Mat& features, const int id) {
  int len = features.cols;
  jfloatArray jf_array = env->NewFloatArray(15 + len);

  float* temp = new float[15];
  for (int i = 0; i < 4; i++)
    temp[i] = info.bbox[i];
  temp[4] = info.score;
  for (int i = 0; i < 5; i++) {
    temp[2*i+5] = info.fpts[i].x;
    temp[2*i+6] = info.fpts[i].y;
  }

  env->SetFloatArrayRegion(jf_array, 0, 15, temp);
  env->SetFloatArrayRegion(jf_array, 15, len, features.ptr<float>(id));
  jfloat* jfp = env->GetFloatArrayElements(jf_array, JNI_FALSE);

  delete[] temp;

  jclass feat_class = env->FindClass("Lcom/neptune/utils/FaceFeature;");
  jmethodID feat_construct = env->GetMethodID(feat_class, "<init>", "([F)V");
  jobject j_feat = env->NewObject(feat_class, feat_construct, jf_array);
  
  return R(j_feat);
}

jobject toJava(JNIEnv* env, const std::vector<FaceInfo>& infos) {
  jclass list_class = env->FindClass("Ljava/util/ArrayList;");
  jmethodID list_construct = env->GetMethodID(list_class, "<init>","()V");
  jobject list_obj = env->NewObject(list_class , list_construct);
  // ArrayList.add: boolean add(Object obj)
  jmethodID list_add = env->GetMethodID(list_class, "add", "(Ljava/lang/Object;)Z");

  for (auto info : infos) {
    jobject j_info = toJava(env, info);
    env->CallBooleanMethod(list_obj, list_add, j_info);
  }

  return list_obj;
}

jobject toJava(JNIEnv* env, const std::vector<FaceInfo>& infos, const cv::Mat& features) {
  jclass list_class = env->FindClass("Ljava/util/ArrayList;");
  jmethodID list_construct = env->GetMethodID(list_class, "<init>","()V");
  jobject list_obj = env->NewObject(list_class , list_construct);
  // ArrayList.add: boolean add(Object obj)
  jmethodID list_add = env->GetMethodID(list_class, "add", "(Ljava/lang/Object;)Z");

  int num = infos.size();
  for (int i = 0; i < num; i++) {
    jobject j_feat = toJava(env, infos[i], features, i);
    env->CallBooleanMethod(list_obj, list_add, j_feat);
  }

  return list_obj;
}

#endif // JNI_UTILS_HPP_