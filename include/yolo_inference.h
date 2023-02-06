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

// Данные переменные в лучшем случае должны загружаться через 
// метод вида get_config в классе Yolov4_tiny, но в 
//тестовом задании дешевле объявить их глобально.

constexpr float CONFIDENCE_THRESHOLD = 0.4;
constexpr float NMS_THRESHOLD = 0.6;
constexpr int NUM_CLASSES = 80;
constexpr int TOP_DETECTION_COUNT = 10;


class Yolov4_tiny
{

	/*
		ru:
			Основной класс Yolov4_tiny для управления работой сети.
		eng:
			The main Yolov4_tiny class for network management.
	*/
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

		input = tvm::runtime::NDArray::Empty({1, 3, 416, 416}, DLDataType{kDLFloat, 32, 1}, dev);
		output_0 = tvm::runtime::NDArray::Empty({1, 2535, 1, 4}, DLDataType{kDLFloat, 32, 1}, dev);
  		output_1 = tvm::runtime::NDArray::Empty({1, 2535, 80}, DLDataType{kDLFloat, 32, 1}, dev);

	}

	void load_and_preprocessing_data(cv::Mat);
	void run_inference();
	json dump_output_to_json(int, int, float);


private:

	std::vector<std::vector<float>> inference_bboxes; // Вектор bbox-ов на выходе из сети.
    std::vector<std::vector<float>> inference_confs;

	std::vector<std::vector<cv::Rect2d>> result_bboxes; // Вектор bbox-ов полученных после постпроцессинга.
    std::vector<std::vector<float>> result_scores;
    std::vector<std::vector<int>> result_ClassIds;

    std::vector<std::string> class_vector;

    tvm::runtime::NDArray input, output_0, output_1;
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
