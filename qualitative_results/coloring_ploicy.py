save_pre_path=r'C:\My data\KeenAI\results'
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def blend(image1,gt,pre, ratio=0.5):
    
    #print(image1.shape)
    # print(gt.shape)
    # print(pre.shape)
    

    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha
    theta=beta-0.1

    #coloring yellow.
    gt *= [0, 1, 0] ### Green Color for gt's 1
    pre*=[1,0,0]   ## Red Color for prediciton's 1
    image = image1 * alpha + gt * beta+ pre * theta
    
    return image




img=cv2.imread(r'C:\My data\KeenAI\Data\data\val\img\img_1_216.png')
g=cv2.imread(r'C:\My data\KeenAI\Data\data\val\gt\img_1_227.png',0)
p=cv2.imread(r'C:\My data\KeenAI\Data\data\val\gt\img_4_40.png',0)

g = np.stack((g,)*3, axis=-1)
p = np.stack((p,)*3, axis=-1)

img = np.float32(img)
g = np.float32(g)/255
p = np.float32(p)/255

def normalize(x):
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))
img=normalize(img)   
result=blend(img,g,p)

plt.imsave(os.path.join(save_pre_path,'this'+".png"),result)
