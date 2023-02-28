Human Tracking Robot Perception Stack
=====================================

This repository contains the code for a perception stack designed and developed for a human tracking robot. The perception stack utilizes deep learning-based systems for person detection and re-identification.

Dependencies
------------

The following dependencies are required to run the code:

-   PyTorch
-   OpenCV
-   NumPy

Installation
------------

To install the dependencies, run the following command:

Copy code

`pip install -r requirements.txt`

Usage
-----

To use the perception stack, run the following command:

Copy code

`python perception_stack.py`

Algorithms used
---------------

The perception stack utilizes the following algorithms:

-   YOLOv5 for human detection and instance segmentation
-   SuperPoint for feature extraction and matching

Results
-------

The implemented perception stack achieves accurate identification of the target individual, allowing for effective human tracking by the robot.

Acknowledgments
---------------

The code in this repository is based on the following research papers:

-   [YOLOv5: A Universal Object Detector](https://arxiv.org/abs/2104.03350)
-   [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
