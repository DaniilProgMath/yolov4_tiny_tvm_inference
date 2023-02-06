#include <iostream>
#include "yolo_inference.h"


void draw_predictions(cv::Mat& image, json frame_info)
{
	/*
		ru:
        	Функция для отрисовки найденных объектов на изображении.
    	eng:
        	Function for drawing the found objects on the image.

	*/

	int font = cv::FONT_HERSHEY_DUPLEX;
	double fontScale = 0.7;
	int msg_thickness = 1;

	for (json::iterator it = frame_info["objects"].begin(); it != frame_info["objects"].end(); ++it) {

  		float x1 = (*it)["coords"]["x1"];
  		float y1 = (*it)["coords"]["y1"];
  		float x2 = (*it)["coords"]["x2"];
  		float y2 = (*it)["coords"]["y2"];

  		float score = (*it)["score"];
  		std::string message = (*it)["class"];
  		message += " ";
  		message += std::to_string(score);

		cv::Scalar rgb_color;
  		cv::Size m_size = cv::getTextSize(message, font, fontScale, msg_thickness, nullptr);
  		cv::Point pt2 = cv::Point(x1 + m_size.width, y1 - m_size.height - 3);
  		cv::Rect bbox_rect(x1, y1, x2 - x1, y2 - y1);

  		if ((*it)["class"] == "person")
  			rgb_color = CV_RGB(34, 139, 34);
  		else
  			rgb_color = CV_RGB(0, 0, 255);

  		cv::rectangle(image, bbox_rect, rgb_color, 3); // draw rect for detected bbox obj
  		cv::rectangle(image, cv::Point(x1, y1), pt2, rgb_color, -1); // mini filled frame

  		cv::putText(image,
	            	message,
	            	cv::Point(bbox_rect.x, bbox_rect.y),
	            	font,
	            	fontScale,
	            	CV_RGB(255, 255, 255),
	            	msg_thickness);
	}
}


std::string getCmdOption(char ** begin, char ** end, const std::string & option)
{
	/*
		ru:
			Функция выполняющая поиск значения после параметра option
			в аргументах при запуске программы.
		eng:
			A function that searches after the value of the option parameter
			in the arguments when starting the program.
	*/
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return std::string(*itr);
    }
    return "";
}


bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	/*
		ru:
			Функция определяющая существование параметра 
			option в аргументах при запуске программы.
		eng:
			A function that determines the existence of the 
			option parameter in the arguments when starting the program.
	*/
    return std::find(begin, end, option) != end;
}


json run_yolov4_tiny_model(	Yolov4_tiny yolov4_instance, cv::Mat image, float timestamp, 
							bool visualize_detection, int time_delay)
{
	/*
		ru:
			Функция объеденяющая вызовы для препроцессинга, инференса, получения json и визуализации
			задетектированных объектов.
		eng:
			A function that combines calls for preprocessing, inference, 
			getting json and rendering of detected objects.

	*/
	yolov4_instance.load_and_preprocessing_data(image);
	yolov4_instance.run_inference();
	json frame_info = yolov4_instance.dump_output_to_json(image.cols, image.rows, timestamp);

	if(visualize_detection)
	{
		draw_predictions(image, frame_info);
		cv::imshow("Detected Objects", image);
		cv::waitKey(time_delay);
	}

	return frame_info;
}


int main(int argc, char * argv[])
{	

	/*
		ru:
			Главная функция программы, производит парсинг аргументов, загрузку видео
			или изображения, вызывает запуск сети и печатает в stdout json детекции.
		eng:
			The main function of the program is parsing arguments, loading a video or image,
			causing the network to start and printing the detection json to stdout.
	*/

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
    	json frame_info = run_yolov4_tiny_model(yolov4_instance, image, 0, 
    											visualize_detection, 0);
    	std::cout << frame_info.dump(4) << std::endl;

	}
	else if (file_extention == "mp4")
	{
		cv::VideoCapture cap;
		cap.open(file_path);

		json all_frames_detections = json::array();

		if(!cap.isOpened())
		{
    		std::cerr << "ERROR: Opening video stream or file." << std::endl;
    		return -1;
  		}

  		for(int frame = 0; frame < cap.get(cv::CAP_PROP_FRAME_COUNT); frame++)
	    {
	    	cv::Mat image;
		    cap >> image;

		    if (image.empty()) {
		        std::cerr << "ERROR: Blank frame grabbed."<< std::endl;
		        break;
		    }

		    float timestamp = cap.get(cv::CAP_PROP_POS_MSEC);
		    json frame_info = run_yolov4_tiny_model(yolov4_instance, image, 
		    										timestamp, visualize_detection, 5);
		    all_frames_detections.push_back(frame_info);
		}

		cap.release();
		std::cout << all_frames_detections.dump(4) << std::endl;
	}
	else
		std::cerr<<"ERROR: Wrong file type."<<std::endl;

	cv::destroyAllWindows();

	return 0;
}
