﻿# Violence Detection using LLM
This project combines YOLOv8 for real-time object detection with Google Gemini (1.5 Flash) for semantic understanding of the scene, enabling a powerful AI-based anomaly detection system that can detect violent or suspicious activities in webcam feeds.

# Features
Real-time webcam monitoring with object detection (YOLOv8)

Automatic image analysis via Gemini Pro Vision (Generative AI)

Detects signs of violent behavior (e.g., punching, hitting, fighting)

Overlays Gemini's scene understanding output on the video

Triggers visual alert if violence is detected in any frame

# How It Works
YOLOv8 detects and classifies objects in each webcam frame.

If a frame has no confident detections or looks suspicious:

The frame is saved and passed to Gemini Vision model.

A custom prompt asks Gemini to describe objects and detect violent actions.

The response is scanned for violent keywords (e.g., "fight", "slap").

If any violent action is identified, the system highlights an anomaly alert.
