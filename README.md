# yolov4_tiny_tvm_inference

Решение тестового задания для компании flussonic/Эрливидео, 
детекция людей и автомобилей на изображении и видео с использованием apache tvm.

## Installation

Минимальные требования к сборке:
- gcc 7.5.0
- g++ 7.5.0
- CMake 3.18 или выше

### Инструкция по установке на ubuntu:
Склонируйте необходимые репозитории

```
git clone --recursive https://github.com/apache/tvm.git
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
cd ../../yolov4_tiny_tvm_inference
nano run_build.sh
export TVM_LIBRARY_PATH="../tvm/" # Set your path
export OPENCV_LIBRARY_PATH="../opencv/" # Set your path
```
Запустите этот файл:

```
./run_build.sh
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

## Installation and runninig in python

Требуется версия python 3.6.9 или выше.

### Для установки необходимых пакетов выполните следующие инструкции:

```
cd python_utils
python3 -m venv my_venv
source my_venv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt
```

### Для запуска решения выполните одну из следующих команд:

Для запуска на тестовой картинке */test_data/frame_15.jpg.*
```
python3 run_tvm_yolo.py
```
Для запуска на любом mp4/avi/mkv видео по указанному пути  с визуализацией детекций.
```
python3 run_tvm_yolo.py --file-path video_path.mp4 --visualize
```
Для запуска на любой jpg/jpeg/png картинке по указанному пути с визуализацией детекций.
```
python3 run_tvm_yolo.py --file-path image_path.jpg --visualize
```




