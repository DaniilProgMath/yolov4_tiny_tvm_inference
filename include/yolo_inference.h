#ifndef YOLO_INFERENCE_H
#define YOLO_INFERENCE_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <json.hpp>
using json = nlohmann::json;

constexpr float CONFIDENCE_THRESHOLD = 0.4;
constexpr float NMS_THRESHOLD = 0.6;
constexpr int NUM_CLASSES = 80;


class Yolov4_tiny
{

public:
	Yolov4_tiny(std::string model_path, std::string config_path)
	{
		DLDevice dev{kDLCPU, 0};
		mod_factory = tvm::runtime::Module::LoadFromFile(model_path);
		gmod = mod_factory.GetFunction("default")(dev);
		set_input = gmod.GetFunction("set_input");
		get_output = gmod.GetFunction("get_output");
		run = gmod.GetFunction("run");
		class_vector = load_class_names(config_path);

		x = tvm::runtime::NDArray::Empty({1, 3, 416, 416}, DLDataType{kDLFloat, 32, 1}, dev);
		y0 = tvm::runtime::NDArray::Empty({1, 2535, 1, 4}, DLDataType{kDLFloat, 32, 1}, dev);
  		y1 = tvm::runtime::NDArray::Empty({1, 2535, 80}, DLDataType{kDLFloat, 32, 1}, dev);

	}

	void load_and_preprocessing_data(cv::Mat);
	void run_inference();
	json dump_output_to_json();


private:

	std::vector<std::vector<float>> inference_bboxes;
    std::vector<std::vector<float>> inference_confs;

	std::vector<std::vector<cv::Rect>> result_bboxes;
    std::vector<std::vector<float>> result_scores;
    std::vector<std::vector<int>> result_ClassIds;

    std::vector<std::string> class_vector;

    cv::Mat image, frame, input;
    tvm::runtime::NDArray x, y0, y1;
    tvm::runtime::PackedFunc run, get_output, set_input;
    tvm::runtime::Module mod_factory, gmod;

    void output_postprocess();
    std::vector<std::string> load_class_names(std::string);
    void collect_stream_to_vect(tvm::runtime::NDArray, 
    							std::vector<std::vector<float>>&, 
    							int, int);
	void Mat_to_CHW(float*, cv::Mat&);

};


#endif
