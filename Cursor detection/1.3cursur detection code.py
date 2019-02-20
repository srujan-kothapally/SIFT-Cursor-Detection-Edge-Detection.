import numpy as np
import cv2
import os
from skimage.feature import match_template
from matplotlib import pyplot as plt

#function to bind all the given images and execute at once
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images
#load the images
image = load_images_from_folder("c:\\task3")

#read the image
template = cv2.imread("c:\\pos_523.JPG",0)
template3 = template
s = template.shape

#converting all the pixels with value below 120 to 0 and above 120 to 140.
b=np.where((template3<120))
template3[b] = 0
a=np.where((template3>120))
template3[a]=140
#crop the image get exact shape of cursor.
template3 = template3[13:24,23:30]

#removing all the rows and colums which are completely zero.
template3 = np.transpose(template3)
template3 = template3[~np.all(template3 == 0, axis=1)]
template3 = np.transpose(template3)
template3 = template3[~np.all(template3 == 0, axis=1)]

#applying gaussian blur and laplacian to template
gautemp = cv2.GaussianBlur(template3, (3,3),0)
laptemp = cv2.Laplacian(gautemp, cv2.CV_64F)

c = 1
for image in image:
    
#applying gaussian blur and laplacian to image 
    gauimage = cv2.GaussianBlur(image, (3,3),0)
    lapimage = cv2.Laplacian(gauimage, cv2.CV_64F)
    
    w, h = template3.shape[::-1]
#applyng template matching to the transformed images
    res = match_template(lapimage,laptemp) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)      

    threshold = 0.726
#extracting the value above the threshold.
    loc = np.where(res>threshold)
    for lo in zip(*loc[::-1]):
        cv2.rectangle(image, (int(lo[0]-w), int(lo[1])-h), (int(lo[0] + w), int(lo[1] + h)),(0,255,0),2)

#showing the mapped image.
    cv2.imwrite('d:\\task2\image'+str(c)+'.png',image)
    c +=1
#    cv2.imshow("mapped ", image)
#    cv2.waitKey(0)
#plotting the template before laplacian and after laplacian.
cv2.imwrite("d:\\task2\template"+"before gau of laplacian"+'.jpg',template3)
cv2.imwrite("d:\\task2\template"+"after gau of laplacian"+'.jpg',laptemp)
#plt.subplot(121),plt.imshow(template3, cmap = 'gray')
#plt.subplot(122),plt.imshow(laptemp, cmap = 'gray')
#plt.show()


