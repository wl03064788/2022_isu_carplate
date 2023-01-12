import cv2
import math
import numpy as np
import time
import argparse

from skimage.transform import radon
from skimage import transform
import matplotlib.pyplot as plt
import easyocr

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# LOAD YOLO MODE
net = cv2.dnn.readNetFromONNX(r"C:\Users\yolo_model\best.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('string', type=str)
    return parser

def get_detections(img,net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_supression(input_image,detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT
    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5] 
            if class_score > 0.25:
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index =  cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    if len(index) > 0:
        index = index.flatten()
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    crop_img = []
    confidence = []
    draw_img = image.copy()
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'confi:{:.0f}%'.format(bb_conf*100)
    #    print('w*h =',w*h)
    #    print('confidence =',bb_conf*100)
        confidence.append(bb_conf*100)
        cv2.rectangle(draw_img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(draw_img,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(draw_img,(x,y+h),(x+w,y+h+30),(0,0,0),-1)
        cv2.putText(draw_img,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        crop_img.append(crop_plate(image,boxes_np[ind]))
    return draw_img, crop_img, confidence

def yolo_predictions(img,net):
    input_image, detections = get_detections(img,net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img, crop_img, confidence = drawings(img,boxes_np,confidences_np,index)
    return result_img,crop_img, confidence

def crop_plate(image,bbox):
    x,y,w,h =  bbox
    if x < 0:
        x = 0
    if y < 0 :
        y = 0
    plate = image[y:y+h,x:x+w]
    return plate

def find_plate(crop_img,confidence):
    comp = 0
    temp = 0
    area_temp = 0
    for i in confidence:
        if i > 65 and comp > 65 and area_temp < crop_img[temp].shape[0] * crop_img[temp].shape[1] and abs(i-comp) < 10:
            plate = crop_img[temp]
            area_temp = crop_img[temp].shape[0] * crop_img[temp].shape[1]
        elif i > comp:
            comp = i
            plate = crop_img[temp]
            area_temp = crop_img[temp].shape[0] * crop_img[temp].shape[1]
        #    print('i =',i)
        temp += 1 
    return plate

def degree( x1,  y1,  x2,  y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 90.0
    elif dy == 0:
        return 0.0
    else:
        if dx < 0:
            dx = -dx
        if dy < 0:
            dy = -dy
    return math.degrees(math.atan(dy/dx))

def img_radon(img):
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    I = I - np.mean(I)
    sinogram = radon(I)
    r = np.array([np.sqrt(np.mean(np.abs(line)**2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    # print('Rotation: {:.2f} degrees'.format(90-rotation))
    return rotation


def img_rotation(img,rotation):
    h,w,_ = img.shape
    if 90 - rotation < 25  and rotation != 90:
        M = cv2.getRotationMatrix2D((w/2,h/2), 90 - rotation, 1)
        dst = cv2.warpAffine(img,M,(w,h))
        dst = cv2.resize(dst,(240,int((h/w)*240)))
        return dst,True
    else:
        return img,False


def horizontal_croping(img):
    dst_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    adaptivethreshold_image = cv2.adaptiveThreshold(dst_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    horizontal_histogram = adaptivethreshold_image.copy()
    (h,w)=horizontal_histogram.shape
    horizontal = [0 for z in range(0, h)] 

    for j in range(0,h):  
        for i in range(0,w):  
            if  horizontal_histogram[j,i]==0: 
                horizontal[j]+=1 
                horizontal_histogram[j,i]=255
    for j in range(0,h):  
        for i in range(0,horizontal[j]):
            horizontal_histogram[j,i]=0

    mid_line = h // 2
    upper_max_black_count = 0
    upper_max_black_count_postion = 0
    for i in range(mid_line-h//4-h//10,0,-1):
        if horizontal[i] > upper_max_black_count:
            upper_max_black_count = horizontal[i]
            upper_max_black_count_postion = i

    lower_max_black_count = 0
    lower_max_black_count_postion = 0
    for i in range(mid_line+h//4+h//10,h):
        if horizontal[i] > lower_max_black_count:
            lower_max_black_count = horizontal[i]
            lower_max_black_count_postion = i

    crop_img = img[upper_max_black_count_postion:lower_max_black_count_postion,0:w]
    return crop_img

def hough_translation(img):
    h,w,_ = img.shape
    gaussian = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(gaussian,cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray,np.mean(gray),255,cv2.THRESH_BINARY)
    edges = cv2.Canny(binary,100,200,apertureSize = 3)
    lines = cv2.HoughLinesP(edges,2,np.pi/180,25,minLineLength=10,maxLineGap=8)

    result1 = img.copy()
    for i in range(0,len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(result1,(x1,y1),(x2,y2),(0,255,0),2)

    min_x1,max_x2 = 999,0
    t_x1,t_x2 = 0,0
    for i in range(0,len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            angle = degree(lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3])
            if x1 < min_x1 and angle>=60 and x1 <= w//10 and abs(angle)!=90:
                min_x1 = x1
                t_x1 = i
            if x2 > max_x2 and angle>=60 and x2 >= (w//10)*9 and abs(angle)!=90:
                max_x2 = x2
                t_x2 = i
    return t_x1,t_x2,lines,min_x1,max_x2,img

def shearing_img(t_x1,t_x2,lines,min_x1,max_x2,img):
    result2 = img.copy()
    if min_x1 != 999:
        cv2.line(result2,(lines[t_x1][0][0],lines[t_x1][0][1]),(lines[t_x1][0][2],lines[t_x1][0][3]),(0,255,255),2)
        angle = math.degrees(math.atan((lines[t_x1][0][3]-lines[t_x1][0][1])/(lines[t_x1][0][2]-lines[t_x1][0][0])))
        #print('angle =',angle)
        if angle < 0:
            angle = 90 - abs(angle)
            translate_M = np.float32([[1,0,-abs(lines[t_x1][0][2]-lines[t_x1][0][0])//2],[0,1,0]])
        else:
            angle = -(90 - abs(angle))
            translate_M = np.float32([[1,0,abs(lines[t_x2][0][2]-lines[t_x2][0][0])//2],[0,1,0]])
        shear = transform.AffineTransform(shear=angle * (math.pi/180))
        imshear = cv2.warpAffine(img,translate_M,(img.shape[1],img.shape[0]))
        imshear = transform.warp(imshear,inverse_map=shear)
        return imshear
    elif max_x2 != 0:
        cv2.line(result2,(lines[t_x2][0][0],lines[t_x2][0][1]),(lines[t_x2][0][2],lines[t_x2][0][3]),(0,255,0),2)
        angle = math.degrees(math.atan((lines[t_x2][0][3]-lines[t_x2][0][1])/(lines[t_x2][0][2]-lines[t_x2][0][0])))
        #print('angle =',angle)
        if angle < 0:
            angle = 90 - abs(angle)
            translate_M = np.float32([[1,0,-abs(lines[t_x2][0][2]-lines[t_x2][0][0])//2],[0,1,0]])
        else:
            angle = -(90 - abs(angle))
            translate_M = np.float32([[1,0,abs(lines[t_x2][0][2]-lines[t_x2][0][0])//2],[0,1,0]])
        shear = transform.AffineTransform(shear=angle * (math.pi/180))
        imshear = cv2.warpAffine(img,translate_M,(img.shape[1],img.shape[0]))
        imshear = transform.warp(imshear,inverse_map=shear)
        return imshear

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    #AC109 RP88
    filename = args.string
    img = cv2.imread(filename)
    results, crop_img, confidence = yolo_predictions(img,net)
    plate = find_plate(crop_img,confidence)
    h,w,_ = plate.shape
    if (w<240):
        img = cv2.resize(plate,(240,int((h/w)*240)))
    rotation = img_radon(img)
    img , is_img_skew = img_rotation(img,rotation)
    if is_img_skew == True:
        t_x1,t_x2,lines,min_x1,max_x2,img = hough_translation(horizontal_croping(img))
        img = shearing_img(t_x1,t_x2,lines,min_x1,max_x2,img)
    img = img*255
    t = time.localtime()
    t = time.strftime("%Y-%m-%d,%H-%M-%S", t)
    cv2.imwrite(r"C:\Users\IoT\Desktop\crop_plate"+ "/" + t + ".jpg",img)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(r"C:\Users\IoT\Desktop\crop_plate"+ "/" + t + ".jpg")
    print(result[0][1])