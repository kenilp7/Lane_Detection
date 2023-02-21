# Detecting Lanes Lines on the Road

## Overview

- This project involves using OpenCV and Python to detect lane lines in images.
- The objective is to develop a pipeline that can detect road lines in a video stream captured by a roof-mounted camera.

## Project Files

- [Lane_detection_img.py](Lane_detection_img.py): contains the code for detecting lane lines in an image
- [Lane_detection_vid.py](Lane_detection_vid.py): contains the code for detecting lane lines in a video

## Goal of the Project

- The project focuses on detecting road lines in an image using computer vision techniques.
- The aim is to identify the lane lines in different driving situations with varying levels of complexity.

## General Process

The process of detecting lane lines involves several steps, including:
- Filtering colors to eliminate non-line components.
- Applying a region filter to focus on the expected location of the lane lines
- Converting the image to grayscale.
- Computing the gradient with the Canny algorithm.
- Smoothing with a Gaussian filter.
- Finding lines with the Hough Transform.
- Smoothing the results with a moving average filter.

## Results

<img width="637" alt="image" src="https://user-images.githubusercontent.com/108230926/220211536-8f5a7807-76fa-4257-9a97-008619609504.png">
