import dlr
import cv2
import numpy as np
from dlr import DLRModel



def softmax(x):
    x_exp = np.exp(x - np.max(x))
    f_x = x_exp / np.sum(x_exp)
    return f_x

def preprocess_image(image):
    cvimage = cv2.resize(image, (224,224))
    #config_utils.logger.info("img shape after resize: '{}'.".format(cvimage.shape))
    img = np.asarray(cvimage, dtype='float32')
    img /= 255.0 # scale 0 to 1
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1)) 
    img = np.expand_dims(img, axis=0) # e.g., [1x3x224x224]
    return img

image_data = cv2.imread('brown_korean_normal.jpeg')
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
image_data = preprocess_image(image_data)
device = 'cpu'            

model = DLRModel('/Users/daekeun/Demo/bioplus-sagemaker-develop/model_mac', device)
output = model.run(image_data)  

probs = softmax(output[0][0])
sort_classes_by_probs = np.argsort(probs)[::-1]


label_map = {0: 'brown_abnormal_korean',
 1: 'brown_normal_chinese',
 2: 'brown_normal_korean',
 3: 'red_abnormal_korean',
 4: 'red_normal'}

max_no_of_results = 1
print(probs)
print(sort_classes_by_probs)
idx = sort_classes_by_probs[0]
print(idx)
msg = f'{label_map[idx]}, {probs[i]*100:.2f}%'