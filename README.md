# Resistor Detection & Value Recognition - CNN + OpenCV

A computer vision system developed in **Python** to detect resistors in images and recognise their colour bands to calculate resistance values.

The project combines a **Convolutional Neural Network (CNN)** for resistor detection with **OpenCV-based image processing** for resistor localisation, colour-band recognition and value calculation.

## Features

* **CNN-based resistor detection** using TensorFlow
* **Binary classification** to distinguish between images containing resistors and non-resistor images
* **Resistor localisation** using OpenCV, Canny edge detection and contour analysis
* **HSV colour segmentation** for detecting resistor colour bands
* **Morphological image processing** to reduce noise and improve band detection
* **Geometric filtering and clustering** to identify and order colour bands
* **Four-band and five-band resistor decoding** using standard resistor colour codes
* **Resistance value calculation** with tolerance handling
* **Unseen test set** containing 15 unique resistor images for evaluation
* **Hybrid computer vision approach** combining deep learning with traditional image processing

## Technologies & Concepts

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* Convolutional Neural Networks
* Binary image classification
* Canny edge detection
* Contour detection and geometric filtering
* HSV colour segmentation
* Morphological image processing
* Feature extraction
* Image classification and object localisation
* Rule-based resistor colour decoding

## Dataset

The training dataset was sourced from **Kaggle** and contains a large collection of resistor images covering different resistance values, backgrounds and orientations:
https://www.kaggle.com/datasets/eralpozcan/resistor-dataset

The dataset was organised into two classes:

* **Resistors**
* **no_resistor**

A separate set of **15 unseen resistor images** was used to evaluate how well the system generalised to new images.

## Approach

The project initially considered a **SIFT-based feature matching approach**. However, testing showed that matching features across a large image dataset was computationally expensive.

The approach was therefore changed to a **CNN-based classifier**, allowing the model to learn relevant features directly from the training data.

Once a resistor is detected, OpenCV is used to:

1. Locate the resistor within the image.
2. Extract the resistor region.
3. Convert the image to HSV colour space.
4. Detect potential colour bands using colour masks.
5. Apply morphological operations and geometric filtering.
6. Group detections based on their horizontal position.
7. Order the detected bands.
8. Calculate the resistor value using standard colour-code rules.

## Model

The resistor detection network consists of **three convolutional layers** with increasing filter sizes:

```text
Input Image
     ↓
Convolutional Layer — 32 filters
     ↓
Max Pooling
     ↓
Convolutional Layer — 64 filters
     ↓
Max Pooling
     ↓
Convolutional Layer — 128 filters
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layers
     ↓
Sigmoid Output

```

The model was trained using binary cross-entropy loss and the Adam optimiser. Class weighting was also applied to account for the imbalance between resistor and non-resistor images.



# Evaluation

The trained system was evaluated using previously unseen resistor images.

The CNN provided reliable resistor detection, whilst colour-band recognition presented greater challenges due to lighting variations, reflections and similarities between resistor body colours and band colours.

Geometric filtering, clustering and rule-based constraints were used to improve the reliability of the final resistance value calculation.

<img width="315" height="217" alt="image" src="https://github.com/user-attachments/assets/1449888c-7ed3-4d09-9d18-9e82c8e2d0d5" />

<img width="427" height="286" alt="image" src="https://github.com/user-attachments/assets/a4c23f31-65a2-4dfc-9e11-f548b4b96317" />

<img width="805" height="116" alt="image" src="https://github.com/user-attachments/assets/2c8c2ceb-b7e9-4ac1-ad03-004643b5c46b" />


## Development Environment

The project was developed using Google Colab, providing a convenient environment for Python-based machine learning development and GPU acceleration.

The training dataset was sourced from Kaggle.


### Project Structure:
```text
resistor-detection-cnn/
├── resistor_detection.ipynb
├── README.md
└── .gitignore
```

## Author:
Dev Sakaria
