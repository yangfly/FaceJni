# FaceApi
Face Api for Detection and Recognition with multi-context support on Caffe.

### 1. 目录结构

```shell
FaceJni
├── additions # mtcnn and centor models
├── build
│   ├── test_api	   # cpp 单元测试
│   └── libJniFace.so  # Jni 动态链接库
├── CMakeLists.txt	# cmake 配置文件
├── com	 # Java 目录
│   ├── neptune
│   │   ├── api/FaceTool.java	# java 接口类
│   │   ├── test/TestFaceTool.java	# java 单元测试
│   │   └── utils
│   │       ├── FaceFeature.java	# 数据格式类
│   │       └── FaceInfo.java		# 数据格式类
│   └── persist/util/tool/Face.java # Image IO
├── config.json	  # 配置文件
├── jni  # Jni 目录
│   ├── com_neptune_api_FaceTool.h  # Jni 头文件(接口类生成)
│   ├── FaceTool.cpp	# Jni 接口实现
│   └── jni_utils.hpp	# Jni 数据转换工具
├── log	# 日志输出目录(*注意配置的glog_dir必须存在)
├── python	# 测试 mtcnn 中间结果正确性，忽略。
├── rapidjson	# json解析头文件库
└── test	# 测试目录：代码和图片
   ├── test_api.cpp	# cpp 单元测试
   └── test_xxx.cpp # 过期测试文件。
```

### 2. 编译流程

- 项目编译
  ```shell
  cd ~/share/TestApi/newapi
  mkdir build && cd build
  cmake .. && make -j 16
  cd .. && ls build  # 看到 test_api 和 libJniFace.so 说明编译成功
  javac com/neptune/test/TestFaceTool.java
  java com.neptune.test.TestFaceTool  # java 单元测试
  build/test_api   # cpp 单元测试
  display build/detect.jpg   # 可以看到人脸检测框和关键点
  ```

- JNI编译(适用于加新功能)
  ```shell
  vim com/neptune/api/FaceTool.java	# 添加 native static 方法
  javac com/neptune/api/FaceTool.java	# 编译出 .class
  javah -d jni com.neptune.api.FaceTool	# 生成并输出 jni 头文件到 jni 目录
  vim jni/FaceTool.cpp	# 实现 jni 接口函数
  # 项目编译....
  ```


### 3. Api 改进
- GpuMat: https://github.com/NVIDIA/gpu-rest-engine/tree/master/caffe
- TensorRT: https://github.com/NVIDIA/gpu-rest-engine/tree/master/tensorrt

### 4. MTCNN
- MTCNN 版本高于 `FaceJpy` 但落后于 `FaceDeploy`。
