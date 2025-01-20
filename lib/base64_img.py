import base64
import numpy as np
import cv2

def encode_base64(file):
    with open(file,'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data)
        print(type(base64_data))
        #print(base64_data)
        # base64_data = "data:image/jpeg;base64," + base64_data
        # 如果想要在浏览器上访问base64格式图片，需要在前面加上：data:image/jpeg;base64,
        base64_str = str(base64_data, 'utf-8')  
        # print(base64_str)
        return base64_data
 
def decode_base64(base64_data):
    with open('./images/base64.jpg','wb') as file:
        img = base64.b64decode(base64_data)
        file.write(img)
        
# 解码base64字符串为numpy图像
def decode_base64_np_img(base64_data, width, height, channels):
    img = base64.b64decode(base64_data)
    img_array = np.fromstring(img, np.uint8)  # 转换np序列
    # TODO: check the image
    img_array = img_array.reshape(height, width, channels)
    print('DEBUG: decoded base64 image shape: ', img_array.shape)
    # cv2.imshow("img", img_array)
    # cv2.waitKey(0)
    return img_array


# 解码base64字符串为opencv图像
def decode_base64_cv_img(base64_data):
    # base64_data = base64_data.split(',')[1]
    img = base64.b64decode(base64_data)
    img_array = np.fromstring(img, np.uint8)  # 转换np序列
    img_raw = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 转换Opencv格式BGR
    img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)  # 转换灰度图

    return img_raw
