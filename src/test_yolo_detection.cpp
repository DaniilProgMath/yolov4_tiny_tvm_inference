#include <iostream>
#include "yolo_inference.h"


void draw_predictions(cv::Mat& image, json frame_info)
{

	for (json::iterator it = frame_info["objects"].begin(); it != frame_info["objects"].end(); ++it) {

  		float x1 = (*it)["coords"]["x1"];
  		float y1 = (*it)["coords"]["y1"];
  		float x2 = (*it)["coords"]["x2"];
  		float y2 = (*it)["coords"]["y2"];

  		cv::Rect bbox_rect(x1, y1, x2 - x1, y2 - y1);

  		cv::rectangle(image, bbox_rect, CV_RGB(0, 255, 0), 2);

  		cv::putText(image,
	            	(*it)["class"],
	            	cv::Point(bbox_rect.x, bbox_rect.y),
	            	cv::FONT_HERSHEY_DUPLEX,
	            	1.0,
	            	CV_RGB(255, 0, 0),
	            	2);
	}
}


std::string getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return std::string(*itr);
    }
    return "";
}


bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}


json run_yolov4_tiny_model(Yolov4_tiny yolov4_instance, cv::Mat image, float timestamp)
{
	yolov4_instance.load_and_preprocessing_data(image);
	yolov4_instance.run_inference();
	return yolov4_instance.dump_output_to_json(timestamp);
}


int main(int argc, char * argv[])
{	

	std::string file_path = getCmdOption(argv, argv + argc, "--file-path");
	std::string file_extention = file_path.substr(file_path.find_last_of(".") + 1);
	bool visualize_detection = cmdOptionExists(argv, argv+argc, "--visualize");

	Yolov4_tiny yolov4_instance = Yolov4_tiny(
											"weights/yolo4-416x416f32.so",
											"config/coco_classes.txt"
										);

	if(file_extention == "jpg" or file_extention == "jpeg" 
		or file_extention == "png")
	{
    	cv::Mat image = cv::imread(file_path);
    	json frame_info = run_yolov4_tiny_model(yolov4_instance, image, 0);
    	std::cout << frame_info.dump(4) << std::endl;

    	if(visualize_detection)
    	{
    		draw_predictions(image, frame_info);
			cv::imshow("Detected Objects", image);
			cv::waitKey(0);
			cv::destroyAllWindows();
    	}

	}
	else if (file_extention == "mp4")
	{
		cv::VideoCapture cap;
		cap.open(file_path);

		json all_frames_detections = json::array();

		if(!cap.isOpened())
		{
    		std::cout << "Error opening video stream or file" << std::endl;
    		return -1;
  		}

  		for(int frame = 0; frame < cap.get(cv::CAP_PROP_FRAME_COUNT); frame++)
	    {
	    	cv::Mat image;
		    cap >> image;

		    if (image.empty()) {
		        std::cerr << "ERROR! blank frame grabbed\n";
		        break;
		    }

		    float timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
		    json frame_info = run_yolov4_tiny_model(yolov4_instance, image, timestamp);
		    all_frames_detections.push_back(frame_info);

		    if(visualize_detection)
    		{
    			draw_predictions(image, frame_info);
				cv::imshow("Detected Objects", image);
				cv::waitKey(5);
    		}
		}

		cap.release();
		cv::destroyAllWindows();

		std::cout << all_frames_detections.dump(4) << std::endl;
	}
	else
		std::cout<<"Wrong file type"<<std::endl;

	return 0;
}
