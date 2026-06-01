# Computer Vision (CV)

## Summary
Computer Vision (CV) enables machines to understand and interpret visual data (images, videos). This document summarizes core concepts, common tasks, model families, datasets, tools, evaluation metrics, and practical tips for applied CV workflows.

## Key Concepts
- Image representation: pixels, color spaces (RGB, HSV), channels
- Feature extraction: hand-crafted (SIFT, HOG) vs learned features (CNNs)
- Convolutional Neural Networks (CNNs): conv, pooling, batch norm, residual blocks
- Transfer learning and fine-tuning: use pretrained backbones to speed convergence
 - Activation functions: prefer ReLU (Rectified Linear Unit) over sigmoid/tanh in deep networks — ReLU mitigates the vanishing-gradient problem, produces sparse (and often more robust) activations, is computationally cheap, and typically yields faster convergence. Consider variants like Leaky ReLU or GELU to avoid "dead" ReLU neurons when needed.
- Data augmentation: flips, rotations, crops, color jitter, MixUp, CutMix

## Common Tasks
- Image classification — assign a single label to an image
- Object detection — localize and classify objects (bounding boxes)
- Semantic segmentation — per-pixel class labels
- Instance segmentation — segment individual object instances
- Keypoint detection / pose estimation — locate landmarks on objects/people
- Image generation & enhancement — super-resolution, denoising, GANs
- Video tasks — tracking, action recognition, temporal segmentation

## Model Families & Examples
- Classification: ResNet, DenseNet, EfficientNet
- Detection: Faster R-CNN, SSD, YOLO (v3/4/5/7), DETR
- Segmentation: U-Net, DeepLab, Mask R-CNN
- Transformers in CV: Vision Transformer (ViT), Swin Transformer, DETR

## Popular Datasets
- ImageNet — image classification
- COCO — detection, segmentation, keypoints
- Pascal VOC — detection/segmentation
- Cityscapes — urban scene segmentation
- Open Images — large-scale object detection

## Tools & Libraries
- Frameworks: PyTorch, TensorFlow, JAX
- High-level libs: Detectron2, MMDetection, torchvision, albumentations
- Labeling: LabelImg, CVAT, LabelMe
- Model hubs: Hugging Face Hub, PyTorch Hub

## Evaluation Metrics
- Classification: accuracy, precision, recall, F1
- Detection: mAP (mean Average Precision), IoU thresholds
- Segmentation: IoU / Jaccard, Dice coefficient
- Tracking: MOTA, ID-switches

## Practical Tips
- Always start from a pretrained backbone for limited data.
- Use strong augmentation and validation splits that reflect deployment data.
- Monitor per-class metrics to catch class imbalance issues.
- For detection, tune IoU and NMS thresholds carefully.
- Keep an image visualization pipeline for qualitative checks.

## Quick Pipeline (example)
1. Collect & label images (COCO/Pascal-style annotations)
2. Preprocess: resize, normalize, augment
3. Choose model & pretraining (e.g., Faster R-CNN with ResNet-50)
4. Train with balanced LR schedule and validation checks
5. Post-process: NMS, confidence thresholding, visualization

## Resources
- Papers: "Deep Residual Learning for Image Recognition" (ResNet), "Faster R-CNN"; "Vision Transformer"
- Libraries: [Detectron2](https://github.com/facebookresearch/detectron2), [MMDetection](https://github.com/open-mmlab/mmdetection)

## Notes
- Add example images to `images/cv_detection_example.png` for demos.


