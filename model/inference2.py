import cv2
import torch
from PIL import Image
import numpy as np
from matplotlib import cm
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)#for PIL/cv2/np inputs and NMS

# Images
for f in ['zidane.jpg', 'bus.jpg']:  # download 2 images
    print(f'Downloading {f}...')
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/' + f, f)
img1 = Image.open('zidane.jpg')  # PIL image
img2 = cv2.imread('bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)
imgs = [img1, img2]  # batched list of images

imgs = data = np.load('data/imgs_512.npz')['X'][0:100, :]
img1 = imgs[0, ...]*255
img2 = imgs[50, ...]*255
img3 = imgs[25, ...]*255

imgs = [img1, img2, img3]

# Inference
results = model(imgs, size=640)  # includes NMS

# Results
results.print()  # print results to screen
results.show()  # display results
results.save()


if __name__ == '__main__':
    data = np.load('data/imgs_512.npz')['X'][0:100, :]
    #for i in range(data.shape[0]):
