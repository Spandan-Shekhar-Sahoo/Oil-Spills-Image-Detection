<<<<<<< HEAD
import torch
import os
import subprocess

def train(yaml_data="oilspill_data.yaml", epochs=100, batch_size=16, img_size=640):
    # Running the YOLOv5 train script via subprocess
    subprocess.run([
        'python', 'S:/IOE_Project/yolov5/train.py',  # Path to train.py in your YOLOv5 folder
        '--data', yaml_data,                         # YAML file for data
        '--epochs', str(epochs),                     # Number of epochs
        '--batch-size', str(batch_size),             # Batch size
        '--img-size', str(img_size)                  # Image size
    ])
    #training-and validation both happens here...validation-happens after every epoch...

def test():
    # Load the trained model directly from the weights file
    model = torch.load('runs/train/exp/weights/best.pt', weights_only=True)
    model.eval()  # Set the model to evaluation mode

    test_images_folder = 'dataset/images/test'  # Directory with test images
    for filename in os.listdir(test_images_folder):
        img_path = os.path.join(test_images_folder, filename)

        results = model.predict(source=img_path, save=True, project=r'S:\IOE Project\results', name="oilspills_experiment")
# Save each image's detections
    """ results.show()  # Show each image's detections -->have,this for each-image...inn,this casee you'lll havee 1000 pop-ups windows iff you're testing and training on 1000images"""
        

yaml_data='oilspill_data.yaml'
epo=100
batch_size=16   
img_size=640

train(yaml_data,epo,batch_size,img_size)
=======
import torch
import os
import subprocess

def train(yaml_data="oilspill_data.yaml", epochs=100, batch_size=16, img_size=640):
    # Running the YOLOv5 train script via subprocess
    subprocess.run([
        'python', 'S:/IOE_Project/yolov5/train.py',  # Path to train.py in your YOLOv5 folder
        '--data', yaml_data,                         # YAML file for data
        '--epochs', str(epochs),                     # Number of epochs
        '--batch-size', str(batch_size),             # Batch size
        '--img-size', str(img_size)                  # Image size
    ])
    #training-and validation both happens here...validation-happens after every epoch...

def test():
    # Load the trained model directly from the weights file
    model = torch.load('runs/train/exp/weights/best.pt', weights_only=True)
    model.eval()  # Set the model to evaluation mode

    test_images_folder = 'dataset/images/test'  # Directory with test images
    for filename in os.listdir(test_images_folder):
        img_path = os.path.join(test_images_folder, filename)

        results = model.predict(source=img_path, save=True, project=r'S:\IOE Project\results', name="oilspills_experiment")
# Save each image's detections
    """ results.show()  # Show each image's detections -->have,this for each-image...inn,this casee you'lll havee 1000 pop-ups windows iff you're testing and training on 1000images"""
        

yaml_data='oilspill_data.yaml'
epo=100
batch_size=16   
img_size=640

train(yaml_data,epo,batch_size,img_size)
>>>>>>> 60a215889218f5e2fc17543489ae788d1c38dadf
test()