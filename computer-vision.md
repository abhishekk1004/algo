# Computer Vision (CV)

## Summary
Computer Vision (CV) enables machines to understand and interpret visual data (images, videos). This document summarizes core concepts, common tasks, model families, datasets, tools, evaluation metrics, and practical tips for applied CV workflows.

## Key Concepts
- Image representation: pixels, color spaces (RGB, HSV), channels
- Feature extraction: hand-crafted (SIFT, HOG) vs learned features (CNNs)
 - Convolutional Neural Networks (CNNs): conv, pooling (max pooling reduces spatial size and adds translation invariance but can discard precise spatial information; alternatives include average pooling, strided convolutions, or global pooling), batch norm, residual blocks
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

## Additional Topics

- Classical CV & Preprocessing: common low-level ops (color conversion, histogram equalization, edge detectors like Canny, feature descriptors like SIFT/ORB), geometric transforms, camera models and calibration, lens distortion correction, and stereo correspondence.
- Self-Supervised & Foundation Models: contrastive methods (SimCLR), masked/image-modeling approaches (MAE), and large pretrained vision foundation models that enable few-shot transfer and prompt-style adaptation.
- Robustness & Safety: adversarial examples and defenses, out-of-distribution detection, uncertainty estimation, dataset shift handling, and evaluating model robustness to corruptions (e.g., ImageNet-C).
- Explainability & Visualization: saliency maps, Grad-CAM, feature inversion, and tools for qualitative inspection of model attention and failure modes.
- Data Efficiency & Synthetic Data: active learning, semi-supervised learning, domain adaptation, and using synthetic data (simulators, domain randomization) to augment scarce labels.
- Annotation & Dataset Best Practices: annotation schemas (COCO, Pascal VOC), labeling quality checks, inter-annotator agreement, bounding-box vs polygon tradeoffs, and efficient labeling workflows/tooling.
- Metrics & Evaluation (expanded): FID/IS for generative tasks, mAP/IoU variants for detection/segmentation, precision-recall curves, ROC/AUC for binary tasks, per-class breakdowns, and latency/throughput for runtime evaluation.
- Model Compression & Acceleration: quantization, pruning, knowledge distillation, operator fusion, ONNX export, TensorRT/TVM, and trade-offs between accuracy and latency/size.
- Deployment Considerations: edge vs cloud inference, batching strategies, memory/IO constraints, pipeline orchestration (pre/post-processing), monitoring, A/B testing, and model versioning.
- Hardware & Benchmarking: GPU/TPU profiling, mixed-precision training (FP16/AMP), CPU vectorization, and benchmarking for latency, throughput, and power consumption.
- Reproducibility & Experimentation: seed control, deterministic ops, experiment tracking (Weights & Biases, MLflow), and writing reproducible training scripts and configs.
- Ethics, Privacy & Legal: dataset licenses and consent, privacy-preserving techniques (blurring/facial anonymization), and bias audits for fairness in CV systems.

## Quick Additions — Practical Tips (compact)

- Prefer short validation cycles with representative holdouts; monitor per-class metrics.
- Log visual samples (TP/FP/FN) each epoch for qualitative checks.
- When fine-tuning, start with lower LR for pretrained backbone and higher LR for newly initialized heads.
- Use mixed-precision and gradient accumulation to scale batch size when memory-limited.

## More Resources
- Libraries: OpenCV for fundamental CV ops and visualization; ONNX, TensorRT, TVM for runtime optimization.
- Benchmarks & Tools: ImageNet, COCO, Roboflow, Open Images, and robustness suites like ImageNet-C.




