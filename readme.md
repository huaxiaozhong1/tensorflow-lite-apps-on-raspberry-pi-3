# Two tensorflow-lite applications running on Raspberry Pi 3

##### If you are interested at the further development of this project, please go to "[complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite](https://github.com/huaxiaozhong1/complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite)".

This is an "Off-line machine learning" (Offline AI, artificial intelligence) project, in which 2 tensorflow-lite applications are developed to run on Raspberry Pi 3 board. The applications are built in docker container in Ubuntu 18.04 host, then runs with Raspbian 9 (stretch) on Pi. The Readme describes all steps on how these application are created based on tensorflow tree (version 1.12.0) and etc. The project provides all necessary stuffs to go through these steps, or points out where the other stuffs are. It hopes to propose a generic method on cross-utilizing TfLite docker image and Pi platform, so that developer could fast follow to get your own smart applications started confidently on the areas like Embedded intelligence (AI embedded system), smart object (AI IoT) and etc.

## Setup & Run

### Setup Raspberry Pi 3 board.

* Install Raspbian (https://www.raspberrypi.org/learning/software-guide/quickstart/)

* On Raspberry Pi 3 board, install the tool-chain:  
>>```#sudo apt install build-essential```
* Setup nfs client at the board, to connect host.

* Connect camera to the board.  
>>Configure the camera (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/pi_examples).  
>>Verify if the camera works or not:  
>>```#raspistill -v```

### Setup cross-building tool-chain in docker container of host (on my case - Ubuntu 18.04), put my new files into.

* Setup docker image:
>>```$sudo docker pull tensorflow/tensorflow:1.12.0-devel ```  
##### Note: the image is the latest one whose TF version is 1.12.0. I notice that TfLite has been moved out from /contrib with TF version 1.13+, as announced. But, the version hasn’t been announced as “stable”, and my building for libtensorflow-lite.a (/tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh) is blocked. So I am reporting my progress based on TF 1.12.0. As soon as the building of TfLite lib gets stable, all these procedures will be updated to be based on 1.13+.

* Run the docker image:  
>>```$sudo docker run -it tensorflow/tensorflow:1.12.0-devel ```

* Exit the image running, give a name, demo-1.12, to container based on the image.  
>>Run the docker container:  
>>```$sudo docker container start demo-1.12.0 ```  
>>```$sudo docker container attach demo-1.12.0 ```

* Setup building environment in the container (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/rpi.md):
>>```#cd /tensorflow ```  
>>```#apt update ```    
>>```#apt install crossbuild-essential-armhf ```  
>>```#./tensorflow/contrib/lite/tools/make/download_dependencies.sh ```  

* Make a little modification in the building tree:  
>>Replace /tensorflow/tensorflow/contrib/lite/tools/make with [new Makefile](make/Makefile) (sudo docker cp).  
>>Copy [the folder of "camera"](./camera)  to /tensorflow/tensorflow/contrib/lite/examples.

### Prepare more static libs to build demo apps:

* In the container at host:  
>>```#apt install -y libjpeg-dev ```  
>>```#apt install libv4l-dev ```  

* At Pi 3 board:
>>```$sudo apt install -y libjpeg-dev ```  
>>```$sudo apt install libv4l-dev ```

* Create folder /usr/lib/arm-linux-gnueabihf in host’s container, copy the following libs from /usr/lib/arm-linux-gnueabihf/ of Pi 3 board [or from here](./arm-linux-gnueabihf) to the corresponding folder of the container:  
>>libjpeg.a  librt.a  libv4l1.a  libv4l2.a  libv4l2rds.a  libv4lconvert.a

* In the container of host, generate libtensorflow-lite.a and executable of demo apps:  
>>```/tensorflow/tensorflow/contrib/lite/tools/make/build_rpi_lib.sh ```

### Run demo apps at Pi 3 board:

* At board:  
>>```#mkdir demos ```

* Copy label_image and camera from /tensorflow/tensorflow/contrib/lite/tools/make/gen/rpi_armv7l/bin/ of host’s container into the folder.

* Copy [grace_hopper.bmp](data/grace_hopper.bmp) from /tensorflow/tensorflow/contrib/lite/examples/label_image of host’s container into the folder.

* Copy [labels.txt](data/labels.txt) from /tensorflow/tensorflow/tensorflow/contrib/lite/java/ovic/src/testdata of host’s container into the folder.

* Download [Mobilenet_v2_1.0_224_quant.tflite](data/Mobilenet_v2_1.0_224_quant.tflite) from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models.md into the folder.

* Now you can run the 2 demo apps in the folder "demos".

* When running app “camera”, you may need to well manage the brightness that light illuminates on the object you want to recognize. On my test, when the object is illuminated as [the photo](test/181123.jpg), the recognizing results can be achieved as [the log](test/181123.txt). That is, the “confidence”s of 100 frames are around 0.8.  
