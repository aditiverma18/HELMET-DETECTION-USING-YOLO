# HELMET-DETECTION-USING-YOLO
A robust helmet detection system using YOLOv8 and YOLOv8-nano models that specifically detect helmets in videos.

Features
✅ Helmet-specific detection in videos using specialized models
✅ Uses YOLOv8 models (standard and helmet-trained)
✅ Pre-trained models (no training required)
✅ High accuracy for helmet identification
✅ Easy to use with simple commands
Installation
Clone or download this repository

Install Python dependencies:

pip install -r requirements.txt
The YOLO models will be automatically downloaded when you first run the script

Usage
Basic Usage
Using helmet-specific model (Recommended):

python helmet_detector.py --video path/to/your/video.mp4 --model sharathhhhh/safetyHelmet-detection-yolov8
Using standard YOLO models:

python helmet_detector.py --video path/to/your/video.mp4
Advanced Usage
Use a larger model for better accuracy:
# YOLOv8 Small (faster)
python helmet_detector.py --video input.mp4 --model yolov8s.pt

# YOLOv8 Medium (balanced)
python helmet_detector.py --video input.mp4 --model yolov8m.pt

# YOLOv8 Large (best accuracy)
python helmet_detector.py --video input.mp4 --model yolov8l.pt

# YOLOv8 Extra Large (maximum accuracy)
python helmet_detector.py --video input.mp4 --model yolov8x.pt
Adjust confidence threshold:
# Lower threshold (detect more objects)
python helmet_detector.py --video input.mp4 --conf 0.20

