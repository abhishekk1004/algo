# Computer Vision (CV)

Overview
- CV focuses on enabling computers to interpret visual information from images and video.

Important subtopics
- Image classification, object detection, segmentation
- Feature extraction, CNNs, transfer learning
- Video analysis, optical flow, tracking

Key notes
- Preprocessing: resizing, normalization, augmentation (flip, rotate, crop).
- For object detection, learn bounding box regression and non-max suppression.

Quick example (object detection)
- Use a pretrained Faster R-CNN or YOLO model and fine-tune on labeled bounding boxes.

Mermaid pipeline
```mermaid
flowchart LR
  A[Raw images] --> B[Augmentation]
  B --> C[Feature extractor (CNN)]
  C --> D[Head: detection/segmentation/classification]
  D --> E[Post-process & visualize]
```

Notes on images
- Add sample detection results in `images/cv_detection_example.png`.
