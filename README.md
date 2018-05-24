# Driver Posture Classification

This is a PyTorch code for **Driver Posture Classification** task. We use the [AUC Distracted Driver Dataset](https://devyhia.github.io/projects/auc-distracted-driver-dataset). The dataset was captured to develop the state-of-the-art in detection of distracted drivers. Here are some samples from the dataset:
<p align='center'>
<img src='https://devyhia.github.io/images/projects/auc-distracted-driver-dataset/AUC-Dataset.png' title='3D-FAN-Full example' style='max-width:600px'></img>
</p>

The task is to classify an image to one of these pre-defined categories, namely "Drive Safe", "Talk Passenger", "Text Right", "Drink", and etc. We use a pretrained resnet34 model to achieve comparable performance with the orignal paper [Real-time Distracted Driver Posture Classification](https://arxiv.org/abs/1706.09498). The classification accuracy is about 97%.


## Usage

### Requirements

* python 3.5+
* pytorch 0.4
* visdom (optional)


### Steps

0. Download the dataset and its training and testing splits (train.csv and test.csv). Put them in a directory together.
1. Clone the repository

	`git clone https://github.com/husencd/DriverPostureClassification.git
	cd DriverPostureClassification`

2. Download the resnet model pretrained on ImageNet from [pytorch official model urls](https://download.pytorch.org/models/).

	`cd pretrained_models
	sh download.sh`

3. Now you can train/fine-tune the model

	`cd ..
	python main.py [--model resnet] [--model_depth 34]`
	
   If you want to monitor the training process, use visdom
   
	`python -m visdom.server`


## Reference

* Our code is partially based on https://github.com/chenyuntc/pytorch-best-practice.
