# Pneumonia Detection
This project uses a CNN (convolutional neural network) model to detect pneumonia from chest X-ray images. The goal is to classify X-ray images based on if pneumonia is present or not.

## Project Overview
The notebook walks through the following steps:

### Data Collection:
The data is obtained from kaggle. Data: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Data Prepocessing:
1. The data is split between training, testing, and validation datasets
2. All images are divided by 255.0 to make each pixel in the image range from 0 to 1
3. Each X dataset is reshaped so that the model can use it

### CNN Model Architecture:
* Data Augmentation
  * Input layer (150 x 150 x 3)
  * Random Flip (mode=horizontal)
  * Random Rotation (factor=0.2)
* 2D Convolution layer (filters=32, kernel=(3,3), activation="relu")
* 2D Max Pooling layer (pool_size=(2,2))
* Dropout layer (rate=0.2)
* 2D Convolution layer (filters=64, kernel=(3,3), activation="relu")
* 2D Max Pooling layer (pool_size=(2,2))
* Dropout layer (rate=0.2)
* 2D Convolution layer (filters=64, kernel=(3,3), activation="relu")
* 2D Max Pooling layer (pool_size=(2,2))
* Flatten
* Dense layer (units=128, activation="relu")
* Dropout layer (rate=0.25)
* Dense layer (units=2, activation="softmax")

### Training and Evaluation:
The model is trained on the training set and tested on the holdout validation set. Early stopping and reducing learning rate when it plateaus are used to help prevent overfitting. Accuracy is used to evalute the model's performance. A heatmap is used to visualize model's accuracy.

## Prerequisites
To run this notebook, you will need:
* Python 3.x

And the following libraries:
* OpenCV
* NumPy
* TensorFlow
* scikit-learn
* Matplotlib
* seaborn
```python
pip install cv2 numpy tensorflow sklearn matplotlib seaborn
```

## Instructions
Clone or download this repository to your local machine. Open the notebook (pneumonia_detection.ipynb) in Jupyter Notebook or Jupyter Lab. You can modify parameters to experiment with different configurations.

## Results
The trained mode achieved an accuracy of 85% on the testing set. Further improvements could be made by improving data augmentation or using different model architecture.

## Acknowledgements
The data was obtained from kaggle using their API (Data: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). This project was inspired by a desire to apply deep learning for classifying medical images.
