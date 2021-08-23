# End-to-end AIoT w/ SageMaker and Greengrass 2.0 on NVIDIA Jetson Nano

***[Note] This hands-on is for NVIDIA Jetson nano, but with only a few lines of code, it works smoothly on NVIDIA Jetson Xavier and Raspberry Pi.***

This hands-on lab starts with ML steps such as data preparing, model training, and model compiling, and then deals with creating and deploying Greengrass v2 components and recipes from scratch on NVIDIA Jetson nano devices. Each folder can be run independently, and you can omit ML Hands-on if you have already compiled models.

- `sm-model-train-compile`: ML part (Amazon SageMaker)
- `ggv2-deploy-on-device`: IoT part (AWS IoT Greengrass 2.0)

## 1. ML Part: Compile your ML model using Pytorch Framework and Amazon SageMaker
In the ML part, you can freely prepare your own data, organize a folder, and execute the code, so that even ML beginners can easily train/compile your own model. Does your data exceed gigabytes? In consideration of large data, PyTorch DDP-based distributed learning was also implemented.

Let's take an example of raw image folder .

Example 1) Training a dog and cat classification model
```
raw
├── cat
└── dog
```

Example 2) Example of good/defective distinction in the production line
```
raw
├── brown_abnormal_chinese
├── brown_abnormal_korean
├── brown_normal_chinese
├── brown_normal_korean
├── no_box
├── red_abnormal
└── red_normal
```

Please refer to the image folder corresponding to Example 2 for reference.
Note that the image data was taken by the author himself, and images from the Internet were not used at all.


## 2. IoT Part: On-Device ML Inference with AWS IoT Greengrass 2.0’

You can directly copy the compiled model from the ML part to NVIDIA Jetson nano for inference, but in real production environments, you need to register multiple devices at once. At this point, you can register your own Greengrass-v2 component and deploy it to multiple edge devices conveniently. In the future, based on this, it is also possible to publish inference data to MQTT to detect Model/Data Drift.

All the codes work normally, but there are many parts that need to be changed manually, so automation through CDK is recommended in the future.

### 1. Optional: Simple Test
```
$ cd artifacts

# Single image inference
$ python3 test_dlr.py

# Camera w/ real-time inference
$ python3 test_camera_dlr.py

# Camera w/ real-time inference (write captured images)
$ python3 test_camera_dlr_write.py

# Flask Camera
$ export DEVICE_IP='[YOUR IP]'
$ python3 flask_camera.py

# Flask Camera w/ real-time inference
$ python3 flask_camera_dlr.py
```

### 2.2. Optional, but Recommended: Shell Script Test
```
$ cd artifacts
$ chmod +x run.sh run_flask.sh

# No Camera: Real-time inference for samples images
$ ./run.sh -c 0 -g 0

# Camera: Real-time inference 
$ ./run.sh -c 1 

# Flask Camera w/ real-time inference
$ ./run_flask.sh -i [YOUR-DEVICE-IP] -p [YOUR-PORT]
# ./run_flask.sh -i 192.168.200.200 -p 2345
```

### 2.3. Register AWS IoT Greengrass Component 
1. Modify `config.json` first.
2. Run `create_gg_component.sh`.

```
$ ./create_gg_component.sh
```