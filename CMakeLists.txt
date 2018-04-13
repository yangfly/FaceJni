cmake_minimum_required(VERSION 2.8.8)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message (STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
    message (STATUS "OpenCV_LIBS=${OpenCV_LIBS}")
endif()

find_package(JNI REQUIRED)
if (JNI_FOUND)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	message (STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
endif()

find_package(Caffe REQUIRED)
if (Caffe_FOUND)
	message (STATUS "Caffe_INCLUDE_DIRS=${Caffe_INCLUDE_DIRS}")
    message (STATUS "Caffe_DEFINITIONS=${Caffe_DEFINITIONS}")
 	message (STATUS "Caffe_LIBRARIES=${Caffe_LIBRARIES}")
endif()

include_directories(
	${PROJECT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS})
	
file(GLOB PROJECT_INCLUDE ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.hpp)
file(GLOB PROJECT_SRC ${PROJECT_SOURCE_DIR}/*.cpp)
message (STATUS "PROJECT_INCLUDE=${PROJECT_INCLUDE}")
message (STATUS "PROJECT_SRC=${PROJECT_SRC}")

set(srcs "test/test_api.cpp" "jni/FaceTool.cpp")
message (STATUS "srcs=${srcs}")


foreach(src ${srcs})
    # get file name
    get_filename_component(name ${src} NAME_WE)
    # get folder name
    get_filename_component(path ${src} PATH)
	get_filename_component(folder ${path} NAME_WE)
	
	if (${folder} STREQUAL "test")
		add_executable(${name} ${src} ${PROJECT_INCLUDE} ${PROJECT_SRC})
		target_link_libraries(${name} ${OpenCV_LIBS}
			"-Wl,--whole-archive" ${Caffe_LIBRARIES} "-Wl,--no-whole-archive" )
	else()  #(${folder} STREQUAL "jni")
		set(name "JniFace")
		include_directories(${JNI_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/jni)
		add_library(${name} SHARED ${src} ${PROJECT_INCLUDE} ${PROJECT_SRC} ${JNI_INCLUDE})
		target_link_libraries(${name} ${OpenCV_LIBS} ${JNI_LIBRARIES}
			"-Wl,--whole-archive" ${Caffe_LIBRARIES} "-Wl,--no-whole-archive" )
	endif()	
endforeach()
