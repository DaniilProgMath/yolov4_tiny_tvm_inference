export TVM_LIBRARY_PATH="../tvm/"
export OPENCV_LIBRARY_PATH="../opencv/build/"

if [ -d "build" ]; then
	echo "build is exist"
else
	mkdir build
fi

cd build
cmake	\
	-DTVM_LIBRARY_PATH=$TVM_LIBRARY_PATH \
	-DOPENCV_DIR=$OPENCV_LIBRARY_PATH \
	..

make install
