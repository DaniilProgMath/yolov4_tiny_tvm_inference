cmake_minimum_required(VERSION 3.18)
project(test_yolo_detection)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


set(OPENCV_DIR "" CACHE STRING "opencv installation directory")
set(TVM_LIBRARY_PATH "" CACHE STRING "apache tvm installation directory")


find_package(
		OpenCV REQUIRED
		COMPONENTS opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_imgcodecs opencv_dnn
		CONFIG 
		PATHS ${OPENCV_DIR})

set( NAME_SRC
    src/test_yolo_detection.cpp
    src/yolo_inference.cpp
    src/tvm_runtime_pack.cc
)

set( NAME_HEADERS       
     include/yolo_inference.h
     include/json.hpp
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(${TVM_LIBRARY_PATH}/include)
INCLUDE_DIRECTORIES(${TVM_LIBRARY_PATH}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${TVM_LIBRARY_PATH}/3rdparty/dlpack/include)

link_directories(${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(test_yolo_detection ${NAME_SRC} ${NAME_HEADERS})

target_link_libraries(test_yolo_detection ${OpenCV_LIBS} ${CMAKE_DL_LIBS})

install(DIRECTORY weights/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/weights)
install(DIRECTORY config/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/config)
install(DIRECTORY test_data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/test_data)

add_custom_target(run
    COMMAND test_yolo_detection --file-path ${CMAKE_CURRENT_BINARY_DIR}/test_data/frame_15.jpg
    DEPENDS test_yolo_detection
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
