# Traffic Volume Detection

This repository contains the code for traffic volume detection using Yolox and tracking using DeepSort. 

## Getting Started

Follow the steps below to set up the project on your local machine.

### Prerequisites

- Python 3.8+
- Git
- CUDA 
- Virtualenv 

### Setup Instructions

#### 1. Clone the Repository

'''bash
git clone https://github.com/eiphyusinn-dev/traffic_volume_detection.git
cd traffic_volume_detection
'''

#### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

'''bash
python3 -m venv traffic_env
'''

#### 3. Activate the Virtual Environment

- On **Linux/macOS**:

'''bash
source traffic_env/bin/activate
'''

- On **Windows**:

'''bash
traffic_env\Scripts\activate
'''

#### 4. Install Dependencies

After activating the virtual environment, install the required dependencies:

'''bash
pip install -r requirements.txt
'''

#### 5. Set Up YOLOX

Setting PYTHONPATH
Before running the detector, you need to set the PYTHONPATH to include the path to the yolox directory. Use the following command:

'''bash
export PYTHONPATH=$PYTHONPATH:/path/to/traffic_volume_detection-main/yolox
'''

#### 6. Download Yolox Weights files 

Create a weights/ directory and download the Yolox-x weight file using the following command: 

'''bash
mkdir weights/
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
'''
#### 6. Prepare input video files 

Create a src_videos/ directory and put the video files inside the folder:

### Running the Project

# YOLOX Detector Script

This section explains how to run the YOLOX detector script to process videos.

## Command to Run the Detector

To run the YOLOX detector, use the following command:

'''
python3 scripts/yolox_detector.py --save --json --input_dir src_videos/ --show
'''

# DeepSort Tracker Script

This section explains how to run the DeepSort Tracker script to process videos.

## Command to Run the Detector

To run the YOLOX detector, use the following command:

'''
python3 scripts/yolox_tracker.py --save --json --input_dir src_videos/ --show
'''


