# Magic Hand Painter 🎨

A real-time hand gesture drawing application built with Python, OpenCV, MediaPipe, and NumPy.

## Features

* Draw using the index finger.
* Switch to red using the index and middle fingers.
* Switch to blue using a thumbs-up gesture.
* Clear the canvas using an open palm.
* Recognize drawn shapes by pressing `D`.
* Save drawings by pressing `S`.
* Press `C` to clear and `Q` to quit.

## Requirements

* Python
* OpenCV
* MediaPipe
* NumPy

## Setup

Install the required libraries:

```bash
pip install opencv-python mediapipe numpy
```

Ensure `hand_landmarker.task` is in the project directory.

## Run

```bash
python magic_hand_painter.py
```

## Controls

* `D`: Recognize the drawn shape
* `C`: Clear the canvas
* `S`: Save the drawing
* `Q`: Quit the application

## Technologies

Python | OpenCV | MediaPipe | NumPy

Developed as a team project for the Computational Perception lab.
