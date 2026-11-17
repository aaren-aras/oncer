All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Unreleased
...

### v0.1.1 (2026-11-17)

#### Added

- **Using the App** and **Data Acknowledgements** sections in `README.md`.
- Root `docs/` folder with papers referenced during development.
- Swagger UI template support and configurable paths/defaults for `api/`.
- `.gitattributes` file.

#### Changed

- Revamped `README.md` with clearer intro, prereqs, setup, and retrospective sections.
- Replaced `requirements.txt` files with single `pyproject.toml` file in `api/`.
- Improved contrast between sub-tumoural regions.
- Standardized modality ordering (`T1`, `T1CE`, `T2`, `FLAIR`) across `app/`, `api/`, and docs.
- Moved static assets into `app/src/assets`.
- Shortened filenames for brevity.

#### Fixed

- Root `.env` variable config across `app/` and `api/`.
- Background and font links in `assets/`.
- `api/` startup errors.

#### Removed

- Placeholder text/files, unused dependencies, dead code, and build artifacts.

### v0.1.0 (2026-08-16)

#### Added

- **ONCER** initial release.
- ResU-Net CNN trained on BraTS 2021 Task 1 dataset.
- FastAPI backend with `/predict` endpoint.
- Vue 3 frontend with image uploader and segmentation viewer.
- Docker containerization backed by NVIDIA NGC TensorFlow Container.