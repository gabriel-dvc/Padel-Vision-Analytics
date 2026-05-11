# Padel-Vision-Analytics
Padel-Vision-Analytics is a Technical Proof of Concept (PoC) demonstrating a high-performance computer vision pipeline for real-time sports analytics. This project showcases the integration of YOLOv8, BoT-SORT, and Homography transformations to bridge the gap between raw match footage and actionable physical metrics.


Current Technical PoC Limitations
As a foundational prototype, this system currently operates with the following known constraints:

Static Geometry: Calibration relies on a pre-calculated static homography matrix, requiring consistent camera placement.
Inference Latency: The current implementation is CPU-optimized; high-FPS processing is limited by sequential inference overhead.
Rule-Based Events: Event detection (Serves/Smashes) uses deterministic heuristics rather than learned temporal patterns, which may vary across different broadcast angles.
Engineering Roadmap & Scaling
To transition from PoC to a production-grade system, the following enhancements are planned:

Automated Calibration: Integration of a segmentation-based court-line detection module to dynamically update homography matrices in response to camera shifts.
Hardware Acceleration: Optimization of model graphs using TensorRT (FP16/INT8) to enable low-latency edge deployment on NVIDIA Jetson or similar platforms.
3D Spatial Analysis: Implementation of 3D triangulation via multi-camera setups to eliminate 2D projection ambiguities in ball height and trajectory mapping.
Temporal Action Recognition: Implementation of a 3D-CNN or Transformer-based head for more robust, data-driven game event classification.
