# License plate Recognition
License plate Recognition (LPR) is the capacity to capture images from license plates and transform the optical data into digital information in real-time. LPR is an important technology that is widely used for vehicle management operations such as illegal parking, stolen vehicles detection and many other applications.
By capturing the license plate digital information, LPR allows operators to immediately gather and associate more data about every vehicle. For example, data about the vehicle like origin, restrictions or security alert and data about the driver like license number, personal public or contact information.
The data capture will be performed using OCR routines to translate characters in an image into digital characters that are machine readable.
## Installation

1. **_Clone the repository_**

```bash
 git clone https://github.com/PeterAyad/ImageProject.git
```

2. **_Go to the directory of the repository_**
```bash
 #example.
 $ cd IMAGEPROJECT
  
```
3. **_install needed packages using the following commands_**
```bash
python -m pip install -U scikit-image 
```
```bash
pip install matplotlib
```
```bash
pip install imutils
```
```bash
#it is assumed that you already have pytorch installed if not please go to https://pytorch.org/ and follow the steps to install it
pip install easyocr
```

4. **_Run the script_**
```bash
python final.py
```

## Additional Information about the packages used
Package | Version  
--- | --- | 
easyocr | 1.3.2
imageio |  2.9.0
importlib-resources | 5.4.0
imutils | 0.5.4
matplotlib | 3.5.1
matplotlib-inline | 0.1.3
numpy | 1.21.2
opencv-python | 4.5.5.62
opencv-python-headless | 4.5.4.60
scikit-image | 0.19.1
scipy | 1.7.3
torch | 1.10.1
torchaudio | 0.10.1
torchvision | 0.11.2 


