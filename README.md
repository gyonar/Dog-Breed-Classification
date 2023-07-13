# Dog-Breed-Classification
# End to End Multi-Class Dog Breed Classification

This notebook builds an end to end multiclass dog breed classifier using TensorFlow 2.0 and Tensor flow Hub
## Problem Statement :
Identifying the breed of a dog given an image of a dog

## Data:
This is based on the Kaggle competition dataset -
**DOG BREED IDENTIFICATION**
You are provided with a training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. The goal of the competition is to create a classifier capable of determining a dog's breed from a photo.

## Evaluation:
The competition is evaluated on Multi Class Log Loss between the predicted probability and the observed target.
For each image in the test set, you must predict a probability for each of the different breeds in the following format

id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.

Multi-Class Log Loss is a widely used performance metric for evaluating classification models with multiple classes. It measures the dissimilarity between predicted probabilities and actual class labels.

Formula: log_loss = — (1/N) * ∑(Y_true * log(Y_pred))

## Features
1.   We are dealing with unstructured data ie images of dogs
2.   The data set tells there are over 120 breeds of dogs ie 120 classes
1.   There are over 10000+ images in the training and test sets, where the images in the training set have unique ids which match to the breed in the labels.csv file



