# COVID19 FACE MASK DETECTION

**Objective**
* To train a custom deep learning model to detect whether a person is or is not wearing a mask using the COVID-19 face mask detection dataset.


**Implemented in two phases**
![plan](https://github.com/satyajeet-rajupali/COVID19_FACE_MASK_DETECTION/blob/main/utils/face_mask_detector_phases.jpg)

`In order to train a custom face mask detector, we need to break our project into two distinct phases, each with its own respective sub-steps(as shown by the image above 👆:`

**Training:** This part mostly focusses on loading our face mask detection dataset from disk, training a model (using Keras/TensorFlow) on the `COVID-19 face mask detection dataset`.

**Deployment:** Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and then classifying each face as `with_mask` or `without_mask`.


## Requirements
- Python 3 (I **highly** recommend using Anaconda as this will save you a TON of time)
- Tensorflow (`pip install tensorflow`)
- Sklearn (`pip install scikit-learn`)
- imutils (`pip install umitils`)
- matplotlib (`pip install matplotlib`)
- numpy (`pip install numpy`)

## About the data
- This dataset consists of `1650` images belonging to two classes: 1. `with_mask` & `without_mask`.

## Create, Compile & Train the model

- To accomplish this task, we’ll be using Transfer Learning and will perfomr fine-tuning on the MobileNet V2 architecture, a highly efficient architecture that can be applied to embedded devices with limited computational capacity (ex., Raspberry Pi, Google Coral, NVIDIA Jetson Nano, etc.)


- `Something about MobileNetV2 model:`
MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

![model](https://github.com/satyajeet-rajupali/COVID19_FACE_MASK_DETECTION/blob/main/utils/mobillenetv2.jpg)
- For better understanding refer to this: https://arxiv.org/abs/1801.04381v4


## Few outputs of model

- The output of model when a person has worn a mask

![model](https://github.com/satyajeet-rajupali/COVID19_FACE_MASK_DETECTION/blob/main/utils/With_mask_prediction.jpg)


- The output of model when a person hasn't worn a mask

![model](https://github.com/satyajeet-rajupali/COVID19_FACE_MASK_DETECTION/blob/main/utils/without_mask_prediction.jpg)



