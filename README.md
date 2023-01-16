# Welcome to the Vehicle Model Classifier project!

## Overview

This machine learning project aims to classify vehicle models based on provided images. This is a fine-grained classification task, meaning that the model must distinguish between different classes with subtle differences. The method used for this classification is Prototypical Networks, a few-shot learning approach introduced in the paper "[Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)". While the paper "[Prototype Rectification for Few-Shot Learning](https://arxiv.org/abs/1911.10713)" also presents an extension to Prototypical Networks that improves their performance on certain tasks, we found that the vanilla Prototypical Networks performed better on this particular task.

Prototypical Networks were chosen for this task because there are a vast number of vehicle models and manufacturers frequently release new models every 2-5 years. Traditional approaches would struggle to effectively classify such a large and constantly-evolving dataset. Prototypical Networks, on the other hand, can learn a distance metric from a small set of prototype examples and use it to classify novel examples, making them well-suited for this task.

## Paper References

- [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- [Prototype Rectification for Few-Shot Learning](https://arxiv.org/abs/1911.10713)

## Dataset

The dataset used for this project is the [PKU VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html) dataset, which contains images of various vehicle models and brands.

## Inference Guide

[ONNX with Post Processing](https://colab.research.google.com/drive/1uA9B_ZhRxVxPOBXgrNNnMpB3drYYu52E?usp=sharing)
