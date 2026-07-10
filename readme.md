# ISU 2022 Car Plate Recognition Project

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Language](https://img.shields.io/badge/language-Python-blue.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)

## Project Overview

This is an autonomous car plate recognition system that uses image processing and machine learning to detect, extract, and recognize vehicle license plates. The project integrates a Raspberry Pi-based edge computing device with cloud infrastructure for real-time plate recognition.

## Features

- **Real-time License Plate Detection** - Uses computer vision to locate plates in images
- **Plate Character Recognition** - Extracts and recognizes characters from detected plates
- **Edge Computing** - Raspberry Pi-based processing for low-latency detection
- **Database Integration** - Stores violation data in SQL database
- **Image Processing Pipeline** - Comprehensive image preprocessing and optimization

## Project Structure

```
├── Rasp_final.py              # Main Raspberry Pi implementation
├── img_processing_func.py     # Image processing functions
├── convert.py                 # Data conversion utilities
├── violationSQL.sql           # Database schema for violations
├── flows.json                 # Node-RED workflow configuration
└── readme.md                  # This file
```

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- SQLite/MySQL for database operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wl03064788/2022_ISU_carplate.git
cd 2022_ISU_carplate
```

2. Install required packages:
```bash
pip install opencv-python numpy
```

## Usage

To run the main plate recognition system on Raspberry Pi:

```bash
python Rasp_final.py
```

## License

MIT License

## Contributors

- Original work: HsuKC (May 2020)
- Updated: 2022
