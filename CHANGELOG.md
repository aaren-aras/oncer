# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

...

---

## [v0.1.0] - 2026-08-16

### Added
- Initial release of ONCER brain tumor segmentation pipeline.
- 2D ResU-Net model trained on BraTS 2021 Task 1 dataset (NCR, ED, ET classes).
- FastAPI backend with `/predict` endpoint and Swagger UI documentation.
- Vue 3 + TypeScript frontend with 4-modality image uploader and interactive segmentation viewer.
- Docker containerization backed by NVIDIA TensorFlow GPU base image.

### Known Limitations
- Requires all 4 MRI modalities (T1, T1CE, T2, FLAIR) per upload.
- Restricted to axial slice inference only.
- Research and educational use only; not clinically validated.