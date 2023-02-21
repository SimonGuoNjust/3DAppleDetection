import os
import cv2
import random

images_path = '/home/psdz/dataset/apple_dataset/v1.0/train/images'
labels_path = '/home/psdz/dataset/apple_dataset/v1.0/train/labels'

for root, dirs, files in os.walk(images_path):
    img_name = random.choices(files,k=1)[0]
    img = cv2.imread(os.path.join(root,img_name))
    w,h = img.shape[0],img.shape[1]
    with open(os.path.join(labels_path,img_name.split('.')[0]+'.txt'),'r') as f :
        bboxes = f.readlines()
        for bbox in bboxes:
            bbox_ = bbox.split(' ')

            cv2.rectangle(img,)
    while cv2.waitKey(10) != ord('q'):
        cv2.imshow('img',img)
