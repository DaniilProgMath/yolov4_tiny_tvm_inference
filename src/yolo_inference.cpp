#include <fstream>

#include "yolo_inference.h"


std::vector<std::string> Yolov4_tiny::load_class_names(std::string config_path)
{
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
	this->image = image.clone();
   	cv::cvtColor(image, this->frame, cv::COLOR_BGR2RGB);
    cv::resize(this->frame, this->input,  cv::Size(416,416));
    float data[416 * 416 * 3];
    Mat_to_CHW(data, this->input);
    memcpy(this->x->data, &data, 3 * 416 * 416 * sizeof(float));
    this->set_input("input", this->x);
}


void Yolov4_tiny::run_inference()
{
	this->inference_bboxes.clear();
    this->inference_confs.clear();

	this->run();
	
	this->get_output(0, y0);
    this->get_output(1, y1);

    collect_stream_to_vect(y0, this->inference_bboxes, 2535*4, 4);
    collect_stream_to_vect(y1, this->inference_confs, 2535*80, 80);

    this->output_postprocess();

}


void Yolov4_tiny::output_postprocess()
{
	std::vector<float> max_confs;
    std::vector<int> max_inds;

    this->result_bboxes.clear();
    this->result_scores.clear();
    this->result_ClassIds.clear();

    for(size_t i = 0; i < this->inference_confs.size(); i++)
    {
    	std::vector<float>::iterator max_conf;
    	max_conf = std::max_element(this->inference_confs.at(i).begin(), 
    								this->inference_confs.at(i).end());
		max_inds.push_back(std::distance(this->inference_confs.at(i).begin(), max_conf));
		max_confs.push_back(*max_conf);
    }

    for(size_t cls = 0; cls < NUM_CLASSES; cls++)
    {
    	std::vector<cv::Rect> cls_bboxes;
    	std::vector<float> cls_scores;
    	std::vector<int> indices;

    	for(size_t i = 0; i < max_inds.size(); i++)
    	{
 			if(max_inds.at(i) == cls)
 			{
 				float x1 = this->inference_bboxes.at(i).at(0) * this->image.cols;
    			float y1 = this->inference_bboxes.at(i).at(1) * this->image.rows;
    			float x2 = this->inference_bboxes.at(i).at(2) * this->image.cols;
    			float y2 = this->inference_bboxes.at(i).at(3) * this->image.rows;

    			cls_bboxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    			cls_scores.push_back(max_confs.at(i));
 			}
    	}

    	cv::dnn::NMSBoxes(cls_bboxes, cls_scores, CONFIDENCE_THRESHOLD, 
    					  NMS_THRESHOLD, indices, 1.f, 10);

    	std::vector<cv::Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;

    	for(size_t i = 0; i < indices.size(); i++)
    	{
    		size_t idx = indices.at(i);
    		nmsBoxes.push_back(cls_bboxes.at(idx));
    		nmsConfidences.push_back(cls_scores.at(idx));
    		nmsClassIds.push_back(cls);
    	}

    	this->result_bboxes.push_back(nmsBoxes);
    	this->result_scores.push_back(nmsConfidences);
    	this->result_ClassIds.push_back(nmsClassIds);
    }
}


json Yolov4_tiny::dump_output_to_json()
{
	json json_detected_objects = json::array();

	for(size_t i = 0; i < this->result_bboxes.size(); i++)
		for(size_t j = 0; j < this->result_bboxes.at(i).size(); j++)
		{
			std::string class_name = this->class_vector.at(this->result_ClassIds.at(i).at(j));

			if (class_name == "person" or class_name == "car")
			{
				json json_object;

				json_object["class"] = class_name;
				json_object["coords"]["x1"] = this->result_bboxes.at(i).at(j).x;
				json_object["coords"]["y1"] = this->result_bboxes.at(i).at(j).y;
				json_object["coords"]["x2"] = this->result_bboxes.at(i).at(j).x + this->result_bboxes.at(i).at(j).width;
				json_object["coords"]["y2"] = this->result_bboxes.at(i).at(j).y + this->result_bboxes.at(i).at(j).height;
				json_object["timestamp"] = 0;
				json_object["score"] = this->result_scores.at(i).at(j);

				json_detected_objects.push_back(json_object);
			}
		}
	return json_detected_objects;
}
