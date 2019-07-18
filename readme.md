<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

**B**ased on  **my other 2 projects**, [tensorflow-lite-apps-on-raspberry-pi-3](https://github.com/huaxiaozhong1/tensorflow-lite-apps-on-raspberry-pi-3) and [complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite](https://github.com/huaxiaozhong1/complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite), you have learned: **1)**, how to develop a ***Tensorflow-lite (Tf-lite)*** app to run an existing Tf-lite model on **Raspbrerry PI (RPI)**; **2)**, how to  re-train an existing Tf model for your own data on RPI.   
**B**ut it remained 1 step within the whole procedure: how to train a Tf model from scratch? Correspondingly, how run the model on RPI?
As a developer for **"AI-on-device"**, you can't be satisfied if unable to write a deep-learning model that can run on a targeted device, such as, RPI.  
After learn through **This Repository**, you will control all the resources to solve an AI problem, such as, recognize an image on RPI.
You will be able to indepedently create and train **your own Tf model**, which is only based on the data you collect, the convolutional neural newwork APIs Tf provides. 
You will be able to convet a TF model to a Tf-lite one so that the latter can run on RPI board. 
At the end, you will see that **your RPI board is able to recognize images**. 

### 1, Create our own Tf model at host.

#### 1.1) Setup Tf docker image.
With GPU: 
```
$sudo docker pull tensorflow/tensorflow:latest-gpu-py3
$sudo docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-py3 bash
```
Or with cpu.
```
$sudo docker run -it tensorflow/tensorflow:latest-py3 bash
```
Now, the latest Tf image has been loaded at your host, and you are inside a container of the image. On my case, it is the version created on June 20th, 2019.

To understand the installation in detail, you could get reference from [Install Tensorflow with Docker](https://www.tensorflow.org/install/docker).

#### 1.2) Setup the docker containre.
 
Exit the container, rename it as a name that is easier for you to remember. On my case, it is called as "my_model".
Type the following command at your host:
```
$sudo docker container start my_model
$sudo docker container attach my_model
```
Now you have entered your Tf docker container. 
In the container, you will make all tasks done to output a tflite mode, with which an image on RPI will be recognized.

#### 1.3) Clone the project respository from github.
Enter the container, execute the following commands.
```
#apt install git
#git clone https://github.com/huaxiaozhong1/Your-Own-TfLite-RaspberryPi-model.git
cd Your-Own-TfLite-RaspberryPi-model
```
Now you could start to utilize my repository.

#### 1.4) Setup necessary environment in the container.
In the container, run some more commands as below.
```
#pip3 install pillow
#apt install python-scipy
#pip3 install scipy

#curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files
#cd flower_photos
```
In directory "flowr_photos", there are 5 folders that contain ~4300 pyotos totally. 
We sugguest you to create 2 sub-folders under "flower_photos": "train" and "validation", move part of the 5 folders into the "train", and the other into "validation".
On my case, the photos under "train" are ~80% in total; the remaining ~20% into "validation".
From [complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite](https://github.com/huaxiaozhong1/complete-procedure-to-train-and-recognize-you-by-raspberry-pi-3-and-tensorflow-lite), you could know the way to take some your own photos from camera on RPI. So, if you want, you could generate photos, for more classes and labels, then split and put them into sub-folders under "train" and "validation".

#### 1.5) Generate our Tf model.

The following command uses of all default input-parameters to generate a Tf model.
```
#python3 generate_my_model.py
```
To get explanation for each input-parameter, use option -h:
```
#python3 generate_my_model.py -h
```
If want to know the control-flow and data-flow of all the commands, please check the comments of generate_my_model.py.
It is just a demo to help you in creating your own Tf model. The model can be improved from a lot of aspects. 
We could open another much bigger topic for these improvements out of this repository :-)

Come back to my report, when running
```
#python3 generate_my_model.py --epoch 1000
```
here are the final lines of the log:
```
Epoch 999/1000
54/55 [============================>.] - ETA: 2s - loss: 0.3163 - acc: 0.9451 === epoch 998, val_loss 0.9640, val_acc 0.7383.
55/55 [==============================] - 141s 3s/step - loss: 0.3178 - acc: 0.9447 - val_loss: 0.9640 - val_acc: 0.7383
Epoch 1000/1000
54/55 [============================>.] - ETA: 2s - loss: 0.3197 - acc: 0.9387 === epoch 999, val_loss 1.1014, val_acc 0.7236.
55/55 [==============================] - 146s 3s/step - loss: 0.3198 - acc: 0.9389 - val_loss: 1.1014 - val_acc: 0.7236
```
That is: the accuracy over train-set is 0.9389, the one over validation-set is 0.7236. 
At the end, the script saves the Tf model into a HDF5 file. Its name is my_model.h5, as default. You could use option -m to set whatever name you like.

#### 1.6) Generate our Tf-lite model.
Based on my_model.h5, you could generate your Tf-lite model by input the following command with default options.
```
#python3 generate_my_tflite_model.py
```
As for the Tf model and Tf-lite model, verification is done respectly. 
Here is an example that the following command was input.
```
python3 generate_my_tflite_model.py -i flower_photos/validation/daisy/5884807222_22f5326ba8_m.jpg -v 1
```
Then the last lines of log are like below.
```
=== input of tflite model:  {'name': 'conv2d_input', 'index': 9, 'shape': array([  1, 128, 128,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}
=== output of tflite model:  {'name': 'dense_1/Softmax', 'index': 15, 'shape': array([ 1, 10], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}
=== Results from recognizing the image with Tf-lite model::  [[9.9985361e-01 4.1775102e-06 1.8509031e-11 6.4527019e-05 5.3741755e-05
  2.3991770e-05 1.5108190e-18 3.9252743e-18 2.4350932e-18 8.3753005e-18]]

=== Label of the image that we recognize with Tf-lite model: 0 
```
Namely, it succeeds in recognizing a photo of "daisy" in "validation-set" as daisy.

### 2 Prepare Python running environment for RPI at host. 

To run Tf "AI-on-device" functinalities on RPI, we could have 2 running environments. The ways to setup them are different.
On Section 2 and 3, the way to run Python script will be introduced.

#### 2.1) Download Tf wheel for use at RPI. 
```
#git clone https://github.com/PINTO0309/Tensorflow-bin.git
```
There will be a tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl at the folder "Tensorflow-bin".

### 3 Run Tf Python script on RPI.

#### 3.1) Prepare basic running system at RPI board. 
Following [Raspberry Pi Software Guide](https://www.raspberrypi.org/learning/software-guide/quickstart), install **Rapibian** onto a Raspberry Pi 3 board.

#### 3.2) Copy necessary files we created at host.
As for the images that we have tested at host, use some tools to convert them to bmp format. 
For example, use [Image converter](https://www.online-convert.com/result/84ef338f-fd8d-4795-939e-ef21bb37cbe5) to play the conversion with flower_photos/validation/daisy/5884807222_22f5326ba8_m.jpg, which we have just tested at host.
Create a connection, nfs or ssh, between RPI and your host.
Via the nfs connection, copy the following files from your host to the RPI: tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl, my_Tflite_model.tflite, recog_image.py and test images. 

#### 3.3) Setup environment for our project on RPI.
Refer to [Install TensorFlow with pip](https://www.tensorflow.org/install/pip#raspberry-pi) and [Benchmarking TensorFlow and TensorFlow Lite on the Raspberry Pi](https://blog.hackster.io/benchmarking-tensorflow-and-tensorflow-lite-on-the-raspberry-pi-43f51b796796), execute the following commands on RPI.
```
$sudo apt update
$sudo apt install python3-dev python3-pip
$sudo pip3 install -U virtualenv  

$virtualenv --system-site-packages -p python3 ./venv
$source ./venv/bin/activate  

$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
$ sudo pip3 install keras_applications==1.0.7 --no-deps
$ sudo pip3 install keras_preprocessing==1.0.9 --no-deps
$ sudo pip3 install h5py==2.9.0
$ sudo apt-get install -y openmpi-bin libopenmpi-dev
$ sudo apt-get install -y libatlas-base-dev
$ pip3 install -U --user six wheel mock

sudo pip3 install tensorflow-1.11.0-cp35-cp35m-linux_armv7l.whl
```
#### 3.4) Recognize an image :-)
Input the following command at RPI, to recognize an image over there.
```
$python3 recog_image.py
```
To get more information for each option of the command, input:
```
$python3 recog_image.py -h
```
On my case, when I ran:
```
python3 recog_image.py -i 5884807222_22f5326ba8_m.bmp -v 1
```
The log printed out is like the below.
```
1.11.0
=== input of tflite model:  {'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'shape': array([  1, 128, 128,   3]), 'index': 9, 'name': 'conv2d_input'}
=== output of tflite model:  {'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'shape': array([ 1, 10]), 'index': 15, 'name': 'dense_1/Softmax'}
=== Results from recognizing the image with Tf-lite model:  [[9.9985361e-01 4.1783310e-06 1.8508254e-11 6.4517357e-05 5.3741856e-05
  2.3988885e-05 1.5108882e-18 3.9253641e-18 2.4352046e-18 8.3755246e-18]]

=== Label of the image that we recognize with Tf-lite model: 0 

```
If you want to exit the environment of image recogniztion, type the command as below.
```
$deactivate
```
Then the virtual environment for running Python is exited.

### 4 Prepare C/C++ developing environment for RPI on host.

If you have gone through my another project [tensorflow-lite-apps-on-raspberry-pi-3](https://github.com/huaxiaozhong1/tensorflow-lite-apps-on-raspberry-pi-3), you have known how to create your own C++ Tf app to run at RPI. 
It's difference from the way to run Phython script is that a static library, which includes all Tf-lite functionalities, will be linked to the app. So no Tf running environment needs to be installed at RPI.
The lib(libtensorflow-lite.a) and our app(label_image) will be built at host. Only the label_image needs to be copied to RPI, since libtensroflow-lite.a has been linked into the app.  
On my experiment, the kind of execuble is faster than Python script.

#### 4.1) Setup Tf docker container for development version.

Similar to 1.1) and 1.2), download and start/attach a **development** version of Tf docker container. On my experiment, it is tensorflow/tensorflow:1.12.0-devel.

#### 4.2) Create a Tf-lite app to recognize image on RPI.
Enter the container , and copy the folder Your-Own-TfLite-RaspberryPi-model into.
```
#mv /tensorflow/tensorflow/contrib/lite/examples/bitmap_helpers_impl.h /tensorflow/tensorflow/contrib/lite/examples/bitmap_helpers_impl.h.ori
#cp -p /Your-Own-TfLite-RaspberryPi-model/bitmap_helpers_impl.h /tensorflow/tensorflow/contrib/lite/examples/label_image/
#mv /tensorflow/tensorflow/contrib/lite/tools/make/Makefile /tensorflow/tensorflow/contrib/lite/tools/make/Makefile
#cp -p /Your-Own-TfLite-RaspberryPi-model/Makefile /tensorflow/tensorflow/contrib/lite/tools/make/
#apt update
#apt install -y crossbuild-essential-armhf
#/tensorflow/tensorflow/contrib/lite/tools/make/download_dependencies.sh
#/tensorflow/tensorflow/lite/tools/make/build_rpi_lib.sh 
```
Copy /tensorflow/tensorflow/contrib/lite/tools/make/gen/rpi_armv7l/bin/label_image from the container to host.
Copy /tensorflow/tensorflow/tensorflow/contrib/lite/java/ovic/src/testdata/label.txt from the contairner to host.

### 5 Run Tf-lite app developed in C++ on RPI

#### 5.1) Setup necessary system components on RPI.

```
$sudo apt install build-essential
```

#### 5.2) Prepare app from host to RPI.

At RPI, stay at the same directory as we are in section 3.
Copy the app having just been created, label_image, from host to the RPI folder.
Copy label.txt from host to RPI.
If you have added your own images when training your model, your need to create your own labels.txt by adding the new labels. Then, use your own label file.

#### 5.3) Run the app to recognize image.
To know how to use opions of the app, please type:
```
$label_image -h
```
One of my examples is like the below.
```
$./label_image -i 5884807222_22f5326ba8_m.bmp -m my_tflite_model.tflite -l labels.txt -v 1
```
Then the last 2 lines of the log printed out is as below.
```
average time: 206.4 ms 
0.999877: 0 daisy
```
Meaning: app recognizes the image from validation-set as a daisy flower. It has confidence in 0.999877.