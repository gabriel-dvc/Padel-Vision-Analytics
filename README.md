# Padel-Vision-Analytics: Technical Proof of Concept (PoC)

## Project Overview
**Padel-Vision-Analytics** is a Technical Proof of Concept (PoC) demonstrating a computer vision pipeline for real-time sports analytics. This project showcases the integration of state-of-the-art detection and tracking algorithms to bridge the gap between raw match footage and actionable physical metrics.

## Technical Architecture
- **Detection**: YOLOv8 (Ultralytics) for robust player and ball identification.
- **Tracking**: BoT-SORT algorithm for stable identity persistence and Re-Identification during occlusions.
- **Spatial Analysis**: Homography Transformation to map pixel coordinates to real-world metric dimensions (20m x 10m).
- **Small Object Detection**: SAHI (Slicing Aided Hyper Inference) for high-sensitivity ball tracking.

## Technical PoC Constraints
As a foundational prototype, this system currently operates with the following known engineering constraints:
- **Static Geometry**: Calibration relies on a pre-calculated static homography matrix, requiring fixed camera placement.
- **Inference Latency**: The current implementation is CPU-bound; real-time high-FPS processing is limited by sequential inference overhead.
- **Heuristic-Based Events**: Game event detection (Serves/Smashes) uses deterministic proximity and velocity heuristics rather than learned temporal patterns.

## Production Scaling Roadmap
To transition from PoC to a production-grade analytics engine, the following enhancements are planned:
1. **Hardware Acceleration**: Optimization of model graphs using **TensorRT (FP16/INT8)** to enable low-latency edge deployment on NVIDIA hardware.
2. **Dynamic Calibration**: Integration of automated court-line detection to dynamically update homography matrices in response to camera shifts or perspective changes.
3. **3D Triangulation**: Implementation of multi-camera setups to resolve 2D projection ambiguities and map precise 3D ball trajectories.
4. **Temporal Action Recognition**: Adoption of 3D-CNN or Transformer architectures for robust, data-driven game event classification.

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run the main processing script: `python scripts/main.py`
