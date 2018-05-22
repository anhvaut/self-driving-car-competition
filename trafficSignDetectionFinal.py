import cv2
import numpy as np
import os
import scipy.misc
import pandas as pd
import tensorflow as tf
import pandas as pd
import mahotas
from imutils.perspective import four_point_transform

LABEL_MAP = {
    "0": 7,
    "1": 7,
    "2": 7,
    "3": 7,
    "4": 7,
    "5": 7,
    "6": 7,
    "7": 7,
    "8": 7,
    "14": 1,
    "34": 2,
    "33": 3,
    "43": 4,
    "44": 5,
    "17": 6,
    "45": 9,
    "46": 9
}
SIGN_NAMES = ["Dung", "Re trai", "Re phai", "Cam re trai", "Cam re phai", "Mot chieu", "Toc do", "Khac", "Khong phai"]

INPUT_SHAPE = (32, 32, 1)

hog = cv2.HOGDescriptor()

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
minW = 15
minH = 15
kernel = np.ones((3,3),np.uint8)


def bgr2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)
def resize_image(image, shape=INPUT_SHAPE[:2]):
    return cv2.resize(image, shape)
def load_image(image_file):
    """
    Read image file into numpy array (RGB)
    """
    return plt.imread(image_file)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('checkpoint/frozen_model.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with tf.Session(graph=detection_graph) as sess:
        for op in sess.graph.get_operations():
            print(op.name)
        image_tensor = detection_graph.get_tensor_by_name('input/x:0')
        is_training = detection_graph.get_tensor_by_name('input/is_training:0')
        output_prediction = detection_graph.get_tensor_by_name('prediction/prediction:0')
        output_probability = detection_graph.get_tensor_by_name('prediction/probability:0')
        
        def process_video(path = "./Videos/MVI_1049.avi"):
            cap = cv2.VideoCapture(path)
            print("Hi")
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    
                    #Turn BGR to HSV
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    #Filter red, blue
                    mask_blue = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

                    mask_red_lower = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
                    mask_red_upper = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))

                    mask = cv2.add(mask_red_lower, mask_red_upper)
                    mask = cv2.add(mask, mask_blue)
                    
                    # Apply Gausian blur
                    mask = cv2.GaussianBlur(mask, (3, 3), 0)

                        
                    T = mahotas.thresholding.otsu(mask)
            
                    # Erode to reduce noise and dilate to focus

                    mask = cv2.Canny(mask, T * 0.5, T)
                    
                    # Erode to reduce noise and dilate to focus
                    #mask = cv2.erode(mask, None, iterations = 1)
                    #mask = cv2.dilate(mask, None, iterations = 3)


                    #Find countour
                    cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
                
                #Draw contours
                    if len (cnts) > 0:
                        mask = cv2.drawContours(mask, cnts, -1, 255, -1)

                        mask = cv2.dilate(mask, kernel, iterations=5)
                        mask = cv2.erode(mask, kernel, iterations=5)

                
                    cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]


                #Extract traffic sign

                    for i in range(0, len(cnts)):
                        cnt = cnts[i]
                        x,y,w,h = cv2.boundingRect(cnt)
                        offset = 5

                        if w > minW and h > minH and float(h)/w > 0.9 and float(h)/w < 1.5:
                            beginY, endY, beginX, endX = y,y+h+offset,x,x+w+offset
                            if beginY > offset:
                                beginY -= offset
                            if beginX > offset:
                                beginX -= offset
                            cropped = frame[beginY:endY, beginX:endX]
                            #resize_img = resize_image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                            #cropped = adjust_gamma(cropped, 2.0)
                            resize_img = resize_image(bgr2gray(cropped))
                            resize_img = np.expand_dims(resize_img, axis=0)
                            resize_img = np.expand_dims(resize_img, axis=3)
                            (prediction, probability) = sess.run(
                                [output_prediction, output_probability],
                                feed_dict={image_tensor: resize_img, is_training: False})
                            print (np.max(probability))
                            if np.max(probability) > 0.9:
                                #cv2.imshow("Cropped", cropped)
                                if str(int(prediction)) in LABEL_MAP:
                                    name_id = LABEL_MAP[str(int(prediction))]
                                else:
                                    name_id = 8
                                    print (SIGN_NAMES[name_id-1])
                                print (SIGN_NAMES[name_id-1])
                                if not name_id == 9:
                                    cv2.imshow(SIGN_NAMES[name_id-1], cropped)
                                    cv2.putText(frame, SIGN_NAMES[name_id-1], (x, y), font, fontScale, fontColor, lineType)
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                        prediction = None
                        probability = None
                    cv2.imshow("mask", mask)
                    cv2.imshow("Origion", frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        process_video("./Videos/MVI_1049.avi")










