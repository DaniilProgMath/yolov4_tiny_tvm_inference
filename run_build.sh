export TVM_LIBRARY_PATH="/home/daniil/Desktop/flussonic_test_task/tvm/"
export OPENCV_LIBRARY_PATH="/media/daniil/684422184421EA10/cpp_packages/opencv/build"

cd build
cmake	\
	-DTVM_LIBRARY_PATH=$TVM_LIBRARY_PATH \
	-DOPENCV_DIR=$OPENCV_LIBRARY_PATH \
	..
