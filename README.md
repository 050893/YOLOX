# YOLOX
YOLOX_on_a_Custom_Dataset


# How to Train YOLOX on Custom Objects
# YOLOX repository by the Megvii Team.

# Steps Covered in this Tutorial
# To train our detector we take the following steps:

# Install YOLOX dependencies
# Download and Prepare custom YOLOX object detection data
# Download Pre-Trained Weights for YOLOX
# Run YOLOX training
# Evaluate YOLOX performance
# Run YOLOX inference on test images
# Export saved YOLOX weights for future inference



# Install YOLOX Dependencies
!git clone https://github.com/roboflow-ai/YOLOX.git
%cd YOLOX
!pip3 install -U pip && pip3 install -r requirements.txt
!pip3 install -v -e .  
!pip uninstall -y torch torchvision torchaudio
# May need to change in the future if Colab no longer uses CUDA 11.0
!pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html


## Install Nvidia Apex
%cd /content/
!git clone https://github.com/NVIDIA/apex
%cd apex
!pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


## Install PyCocoTools
!pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# Download your Data

Use the "**Pascal VOC**" export format.

pwd
%cd /content/
!curl -L "https://public.roboflow.com/ds/rH6fPDa3Pl?key=39VjpgdKA8" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

%cd YOLOX/
!ln -s /content/train ./datasets/VOCdevkit

## Format Your Data Appropriately
%mkdir "/content/YOLOX/datasets/VOCdevkit/VOC2007"
!python3 voc_txt.py "/content/YOLOX/datasets/VOCdevkit/"
%mkdir "/content/YOLOX/datasets/VOCdevkit/VOC2012"
!cp -r "/content/YOLOX/datasets/VOCdevkit/VOC2007/." "/content/YOLOX/datasets/VOCdevkit/VOC2012"


## Change the Classes
Make sure you change the classes based on what your dataset. To ensure that the training process will function as intended, write the classes in lowercase with no whitespace.

from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


##REPLACE this cell with your classnames stripped of whitespace and lowercase
%%writetemplate /content/YOLOX/yolox/data/datasets/voc_classes.py

VOC_CLASSES = (
  "rbc",
  "wbc",
  "platelets"
)


##REPLACE this cell with your classnames stripped of whitespace and lowercase
%%writetemplate /content/YOLOX/yolox/data/datasets/coco_classes.py

COCO_CLASSES = (
  "rbc",
  "wbc",
  "platelets"
)


Set the number of classes you have in your dataset in te `NUM_CLASSES` variable

NUM_CLASSES = 3
!sed -i -e 's/self.num_classes = 20/self.num_classes = {NUM_CLASSES}/g' "/content/YOLOX/exps/example/yolox_voc/yolox_voc_s.py"

# Download Pretrained Weights

%cd /content/
!wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.pth
%cd /content/YOLOX/

# Train the Model
!python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 16 --fp16 -o -c /content/yolox_s.pth


# Evaluate the Model
MODEL_PATH = "/content/YOLOX/YOLOX_outputs/yolox_voc_s/best_ckpt.pth.tar"
!python3 tools/eval.py -n  yolox-s -c {MODEL_PATH} -b 64 -d 1 --conf 0.001 -f exps/example/yolox_voc/yolox_voc_s.py


# Test the Model
Make sure you replace the `TEST_IMAGE_PATH` variable with a test image from your dataset

TEST_IMAGE_PATH = "/content/valid/BloodImage_00000_jpg.rf.3aa7a653c80726cbb25447cb697ad7a4.jpg"
!python tools/demo.py image -f /content/YOLOX/exps/example/yolox_voc/yolox_voc_s.py -c {MODEL_PATH} --path {TEST_IMAGE_PATH} --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu



# Visualize the Predictions
Make sure you replace the `OUTPUT_IMAGE_PATH` with the respective path of the image output. This path can be found somewhere in the `YOLOX_outputs` folder


from PIL import Image
OUTPUT_IMAGE_PATH = "/content/YOLOX/YOLOX_outputs/yolox_voc_s/vis_res/2022_03_16_16_31_10/BloodImage_00000_jpg.rf.3aa7a653c80726cbb25447cb697ad7a4.jpg" 
Image.open(OUTPUT_IMAGE_PATH)


# Export Trained Weights for Future Inference

Now that you have trained your custom detector, you can export the trained weights you have made here for inference on your device elsewhere

from google.colab import drive
drive.mount('/content/gdrive')

%cp /content/YOLOX/YOLOX_outputs/yolox_voc_s/best_ckpt.pth.tar /content/gdrive/MyDrive/Research/YOLOX/

