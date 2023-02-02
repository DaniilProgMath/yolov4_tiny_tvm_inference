# yolov4_tiny_tvm_inference

Решение тестового задания для компании flussonic/Эрливидео, 
детекция людей и автомобилей на изображении и видео с использованием apache tvm.

## Installation

Инструкция по установке на ubuntu 18.04


Склонируйте необходимые репозитории

```
git clone https://github.com/apache/tvm.git
git clone https://github.com/opencv/opencv.git
git clone https://github.com/DaniilProgMath/yolov4_tiny_tvm_inference.git
```

Установите необходимые для opencv библиотеки и выполните сборку исходников opencv
```
sudo apt-get install pkg-config
sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev libavresample-dev
sudo apt install -y g++ cmake make git libgtk2.0-dev pkg-config
cd opencv
mkdir build
cmake ..
make -j4
```

Для сборки решения отредактируйте файл run_build.sh указав актуальные пути до библиотек tvm и opencv

```
cd ../../
nano run_build.sh
export TVM_LIBRARY_PATH="../tvm/" # Set your path
export OPENCV_LIBRARY_PATH="../opencv/" # Set your path
```
Запустите этот файл:

```
bash run_build.sh
```

## Testing cpp compiled code

Для запуска перейдите в папку build и выполните одну из следующих команд

```
cd build
make run
make test_on_image_with_visualization
make test_on_video_with_visualization
```

Таргет run запускает инференс на тестовой картинке.</br>
Таргет test_on_image_with_visualization запускает инференс на тестовой картинке и дополнительно отрисовывает детекции.</br>
Таргет test_on_video_with_visualization запускает инференс на тестовом видео и дополнительно отрисовывает детекции.</br>

Для запуска на своих данных выполните:

```
./bin/test_yolo_detection --file-path image_path.jpg
./bin/test_yolo_detection --file-path video_path.mp4
```

Визуализация результатов:
```
./bin/test_yolo_detection --file-path image_path.jpg --visualize
```



