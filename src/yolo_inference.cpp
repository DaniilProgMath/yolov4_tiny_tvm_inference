#include <fstream>

#include "yolo_inference.h"


std::vector<std::string> Yolov4_tiny::load_class_names(std::string config_path)
{
	/*
		ru:
        	Функция для загрузки списка классов.
    	eng:
        	Function to load the list of classes.
    */

	std::vector<std::string> class_vector;
	std::fstream class_file;
	class_file.open(config_path, std::ios::in);

	if (class_file.is_open())
	{
		std::string tp;
		while(getline(class_file, tp))
			class_vector.push_back(tp);
		class_file.close();
	}

	return class_vector;
}


void Yolov4_tiny::collect_stream_to_vect(
	tvm::runtime::NDArray stream, std::vector<std::vector<float>>& vectors,
	int stream_size, int step)
{
	/*
		ru:
			Функция выполняющая сбор данных из выходного тензора сети в
			контейнер с векторами из числа step значений. 
		eng:
			A function that collects data from the output tensor of the network 
			into a container with vectors of step values.
	*/

	auto stream_ptr = static_cast<float*>(stream->data);

    for(size_t i = 0; i < stream_size; i+=step)
    {
        std::vector<float> vector;
        for(size_t j=i; j < i+step; ++j)
            vector.push_back(stream_ptr[j]);
        vectors.push_back(vector);
    }
}


void Yolov4_tiny::Mat_to_CHW(float *data, cv::Mat &frame)
{
	/*
		ru:
			Функция выполняющая преобразование cv::Mat в линейный массив 
			из нормализованных значений пикселей.
		eng:
			A function that converts cv::Mat into a linear array 
			of normalized pixel values.
	*/
    assert(data && !frame.empty());
    unsigned int volChl = 416 * 416;

    for(int c = 0; c < 3; ++c)
    {
        for (unsigned j = 0; j < volChl; ++j)
            data[c*volChl + j] = static_cast<float>(
            	float(frame.data[j * 3 + c]) / 255.0);
    }
}


void Yolov4_tiny::load_and_preprocessing_data(cv::Mat image)
{
	/*
		ru:
        	Функция выполняющая препроцессинг изображения и загрузку его во
        	входной тензор модели.

    	eng:
        	A function that preprocesses an image and loads it into the model's input tensor.
	*/
	cv::Mat resized_image, RGB_image;
   	cv::cvtColor(image, RGB_image, cv::COLOR_BGR2RGB);
    cv::resize(RGB_image, resized_image,  cv::Size(416,416));
    float data[416 * 416 * 3];
    Mat_to_CHW(data, resized_image);
    memcpy(this->input->data, &data, 3 * 416 * 416 * sizeof(float));
    this->set_input("input", this->input);
}


void Yolov4_tiny::run_inference()
{
	/*
		ru:
			Функция выполняющая инференс модели.
		eng:
			The function that performs the inference of the model.
			
	*/
	this->inference_bboxes.clear();
    this->inference_confs.clear();

	this->run();
	
	this->get_output(0, output_0);
    this->get_output(1, output_1);

    collect_stream_to_vect(output_0, this->inference_bboxes, 2535*4, 4);
    collect_stream_to_vect(output_1, this->inference_confs, 2535*80, 80);

    this->output_postprocess();

}


void Yolov4_tiny::output_postprocess()
{
	/*
		ru:
			Функция для обработки результатов выхода сети.
		eng:
			A function to process the results of a network output.
	*/
	std::vector<float> max_confs;
    std::vector<int> max_inds;
    std::vector<std::vector<float>> max_bboxes;

    this->result_bboxes.clear();
    this->result_scores.clear();
    this->result_ClassIds.clear();

    for(size_t i = 0; i < this->inference_confs.size(); i++)
    {
    	std::vector<float>::iterator max_conf;
    	max_conf = std::max_element(this->inference_confs.at(i).begin(), 
    								this->inference_confs.at(i).end());

    	if(*max_conf > CONFIDENCE_THRESHOLD)
    	{
			max_inds.push_back(std::distance(	this->inference_confs.at(i).begin(), 
												max_conf));
			max_confs.push_back(*max_conf);
			max_bboxes.push_back(this->inference_bboxes.at(i));
		}
    }


    for(size_t cls = 0; cls < NUM_CLASSES; cls++)
    {
    	std::vector<cv::Rect2d> cls_bboxes;
    	std::vector<float> cls_scores;
    	std::vector<int> indices;

    	for(size_t i = 0; i < max_inds.size(); i++)
    	{
 			if(max_inds.at(i) == cls)
 			{
 				float x1 = max_bboxes.at(i).at(0);
    			float y1 = max_bboxes.at(i).at(1);
    			float x2 = max_bboxes.at(i).at(2);
    			float y2 = max_bboxes.at(i).at(3);

    			cls_bboxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
    			cls_scores.push_back(max_confs.at(i));
 			}
    	}

    	cv::dnn::NMSBoxes(cls_bboxes, cls_scores, CONFIDENCE_THRESHOLD, 
    					  NMS_THRESHOLD, indices);

    	std::vector<cv::Rect2d> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;

    	for(size_t i = 0; i < indices.size(); i++)
    	{
    		if(i < TOP_DETECTION_COUNT)
    		{
	    		size_t idx = indices.at(i);
	    		nmsBoxes.push_back(cls_bboxes.at(idx));
	    		nmsConfidences.push_back(cls_scores.at(idx));
	    		nmsClassIds.push_back(cls);
	    	}
    	}

    	this->result_bboxes.push_back(nmsBoxes);
    	this->result_scores.push_back(nmsConfidences);
    	this->result_ClassIds.push_back(nmsClassIds);
    }
}


json Yolov4_tiny::dump_output_to_json(int img_width, int img_height, float timestamp)
{
	/*
		ru:
	        Функция для сохранения результатов детекций в объект json.

	    eng:
	        A function to collect discovery results in a json object.
    */
	json frame_info;
	json json_detected_objects = json::array();

	for(size_t i = 0; i < this->result_bboxes.size(); i++)
		for(size_t j = 0; j < this->result_bboxes.at(i).size(); j++)
		{
			std::string class_name = this->class_vector.at(this->result_ClassIds.at(i).at(j));

			if (class_name == "person" or class_name == "car")
			{
				json json_object;

				int x1 = (int) (this->result_bboxes.at(i).at(j).x * img_width);
				int y1 = (int) (this->result_bboxes.at(i).at(j).y * img_height);
				int box_width = (int) (this->result_bboxes.at(i).at(j).width * img_width);
				int box_height = (int) (this->result_bboxes.at(i).at(j).height * img_height);

				json_object["class"] = class_name;
				json_object["coords"]["x1"] = x1;
				json_object["coords"]["y1"] = y1;
				json_object["coords"]["x2"] = x1 + box_width;
				json_object["coords"]["y2"] = y1 + box_height;
				json_object["score"] = this->result_scores.at(i).at(j);

				json_detected_objects.push_back(json_object);
			}
		}

	frame_info["timestamp"] = timestamp;
	frame_info["objects"] = json_detected_objects;

	return frame_info;
}
