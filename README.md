# 🍎 Fruit Ripeness Classification

Welcome to **Fruit Ripeness Classification**, an AI inference service designed to classify fruit ripeness. This service acts as the "Brain" within our distributed system, processing image data and returning precise classification results via a Docker Bridge Network.


## 🚀 Overview
"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua..."


## 📦 Deployment Instructions

### 1. Prerequisites
Ensure you have the following installed on your system: 
* [Git](https://git-scm.com/downloads) 
* [Docker Engine](https://docs.docker.com/engine/install/)
* [Docker Compose](https://docs.docker.com/compose/install/)

### 2. Clone and Enter Folder 
Open **Terminal** or **Command Prompt** and run the following commands to download the project and enter the directory:
```bash 
git clone https://github.com/j4emuzu/ripe_fruit_classification.git
cd your-repo-name
```
git clone -b ai/model-training https://github.com/j4emuzu/ripe_fruit_classification.git
```
**!!!** This branch is used for testing and let our team access the ai-service feature easily **!!!** 

### 3. Launching the Service
Start the AI service with a single command. Docker will automatically pull the environment, install dependencies, and load the model:
```bash 
docker compose up -d --build
```
**Note:** The first time you run this, it may take a few minutes to build the image. Once finished, the container will stay running in the background.

## ⚡ GPU Acceleration (NVIDIA Users)

By default, the service runs on the **CPU** for maximum compatibility. If you have an NVIDIA GPU and want to enable hardware acceleration:

1.  Open the `docker-compose.yml` file in any text editor.
    
2.  Locate the following line:
	```bash
    # runtime: nvidia
    ```
    
3.  **Remove the `#` symbol** to enable the GPU runtime:
	```bash
    runtime: nvidia
    ```
    
4.  Save the file and restart the service:
	```bash
    docker compose up -d
    ```
    
_Requirement: You must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host machine._