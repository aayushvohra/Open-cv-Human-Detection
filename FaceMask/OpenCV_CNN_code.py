import cv2
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
cv2.imshow('Input image', img)
cv2.waitKey(0)

import cv2
gray_img = cv2.imread('D:/Course slides/Datasets/mandrill.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Grayscale', gray_img)
cv2.waitKey(0)

cv2.imwrite('output.tif', gray_img)
import os
os.getcwd()

import cv2
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
cv2.imwrite('output1.png', img, [cv2.IMWRITE_PNG_COMPRESSION])

import cv2
print([x for x in dir(cv2) if x.startswith('COLOR_')]) # RGB, CMYK, YUV, HSV - Hue, Saturation,Value # cyan, magenta, yellow, and key, YUV, Y is the luminance (brightness) component while U and V are the chrominance (color) components

import cv2
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey(0)
    
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(yuv_img)
cv2.imshow('Y channel', y)
cv2.imshow('U channel', u)
cv2.imshow('V channel', v)
cv2.waitKey(0)

cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey(0)

img = cv2.imread('D:/Course slides/Datasets/mandrill.tif', cv2.IMREAD_COLOR)
g,b,r = cv2.split(img)
gbr_img = cv2.merge((g,b,r))
rbr_img = cv2.merge((r,b,r))
cv2.imshow('Original', img)
cv2.imshow('GRB', gbr_img)
cv2.imshow('RBR', rbr_img)
cv2.waitKey(0)



import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
cv2.imshow('Rotation', img_rotation)
cv2.waitKey(0)



import cv2
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
cv2.imshow('Original Image', img)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey(0)

#Affine Transformations Euclidean Transformation
import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0],[int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)


# Projective transformation
import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/mandrill.tif')
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1],[int(0.66*cols),rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey(0)


#Blurring is called as low pass filter. Why?
import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/plane.bmp')
rows, cols = img.shape[:2]
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32) / 9.0 # Divide by 9 to normalize the kernel
kernel_5x5 = np.ones((5,5), np.float32) / 25.0 # Divide by 25 to normalize the kernel
cv2.imshow('Original', img)
output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('Identity filter', output)
output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3 filter', output)
output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x5 filter', output)
cv2.waitKey(0)
output = cv2.blur(img, (3,3))
cv2.imshow('blur', output)
cv2.waitKey(0)




import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/Median filter.png', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
# It is used to indicate depth of cv2.CV_64F.
sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# Kernel size can be: 1,3,5 or 7.
sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow('Original', img)
cv2.imshow('Sobel horizontal', sobel_horizontal)
cv2.imshow('Sobel vertical', sobel_vertical)
cv2.waitKey(0)



import cv2
import numpy as np
img = cv2.imread('D:/Course slides/Datasets/plane.bmp', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
canny = cv2.Canny(img, 50, 240)
cv2.imshow('Canny', canny)
cv2.waitKey(0)




# how to access the webcam
import cv2
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()


# Video color spaces
import cv2
def print_howto():
    print("""
        Change color space of the input video stream using keyboard controls. The control keys are:
            1. Grayscale - press 'g'
            2. YUV - press 'y'
            3. HSV - press 'h'
    """)
if __name__=='__main__':
    print_howto()
    cap = cv2.VideoCapture(0)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    cur_mode = None
    while True:
        # Read the current frame from webcam
        ret, frame = cap.read()
        # Resize the captured image
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)
        c = cv2.waitKey(1)
        if c == 27:
            break
        # Update cur_mode only in case it is different and key was pressed
        # In case a key was not pressed during the iteration result is -1 or 255, depending on library versions
        if c != -1 and c != 255 and c != cur_mode:
            cur_mode = c
        if cur_mode == ord('g'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif cur_mode == ord('y'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif cur_mode == ord('h'):
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            output = frame
        cv2.imshow('Webcam', output)
    cap.release()
    cv2.destroyAllWindows()



import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()



# Identifying face and Augmented reality
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
face_mask = cv2.imread('D:/Course slides/Datasets/mask.jfif')
h_mask, w_mask = face_mask.shape[:2]
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
cap = cv2.VideoCapture(0)
scaling_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3,minNeighbors=3)
    for (x,y,w,h) in face_rects:
        if h <= 0 or w <= 0: pass
        # Adjust the height and weight parameters depending on the sizes and the locations.
        # You need to play around with these to make sure you get it right.
        h, w = int(1.0*h), int(1.0*w)
        y -= int(-0.2*h)
        x = int(x)
        # Extract the region of interest from the image
        frame_roi = frame[y:y+h, x:x+w]
        face_mask_small = cv2.resize(face_mask, (w, h), interpolation=cv2.INTER_AREA)
        # Convert color image to grayscale and threshold it
        gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
        # Create an inverse mask
        mask_inv = cv2.bitwise_not(mask)
        try:
            
            # Use the mask to extract the face mask region of interest
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
            # Use the inverse mask to get the remaining part of the image
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        except cv2.error as e:
            print('Ignoring arithmentic exceptions: '+ str(e))
        # add the two images to get the final output
        frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()

# Eyeball detection
import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('D:/OPENCV/opencv/sources/data/haarcascades/haarcascade_eye.xml')
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
cap = cv2.VideoCapture(0)
ds_factor = 0.5
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors=1)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)
        cv2.imshow('Eye Detector', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(2000)


# Loading a video 
# (Known issue with opencv_python package, which does not load the video.
# solution: pip3 uninstall opencv_python
#           pip3 install opencv_python --user
# - Restart kernel after package reinstallation    
import cv2
import numpy as np
cap = cv2.VideoCapture("/video/VID_20191019_103117.mp4")
cap.isOpened()
ret, image = cap.read()
while ret:
    cv2.imshow('Video Stream', image)
    cv2.waitKey(20)
    ret, image = cap.read()    
cap.release()
cv2.destroyAllWindows()
print('Video Finished')

# Dense optical flow
import cv2
import numpy as np

def display_flow(img, flow, stride=40): 
    for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
        pt1 = tuple(i*stride for i in index)
        delta = flow[pt1].astype(np.int32)[::-1]
        pt2 = tuple(pt1 + 10*delta)
        if 2 <= cv2.norm(delta) <= 10:
            cv2.arrowedLine(img, pt1[::-1], pt2[::-1], (0,0,255), 5, cv2.LINE_AA, 0, 0.4)

    norm_opt_flow = np.linalg.norm(flow, axis=2)
    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

    cv2.imshow('optical flow', img)
    cv2.imshow('optical flow magnitude', norm_opt_flow)
    k = cv2.waitKey(1)

    if k == 27:
        return 1
    else:
        return 0
    
    
cap = cv2.VideoCapture("D:/Object detection/madurai-traffic-counter-master/madurai-traffic-counter-master/input/highway1.mp4")
_, prev_frame = cap.read()

prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.resize(prev_frame, (0,0), None, 0.5, 0.5)
init_flow = True

while True:
    status_cap, frame = cap.read()
    frame = cv2.resize(frame, (0,0), None, 0.5, 0.5)
    if not status_cap:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if init_flow:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        init_flow = False
    else:
        opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, opt_flow, 
                                                0.5, 5, 13, 10, 5, 1.1, 
                                                cv2.OPTFLOW_USE_INITIAL_FLOW)

    prev_frame = np.copy(gray)

    if display_flow(frame, opt_flow):
        break;

cv2.destroyAllWindows()



#Single Shot Object Detection Model
import cv2
import numpy as np

model = cv2.dnn.readNetFromCaffe('D:/Course slides/Datasets/MobileNet-SSD-master/voc/MobileNetSSD_deploy.prototxt',
                                 'D:/Course slides/Datasets/MobileNet-SSD-master/MobileNetSSD_deploy.caffemodel')

CONF_THR = 0.3
LABELS = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
          14: 'motorbike', 15: 'person', 16: 'pottedplant',
          17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

video = cv2.VideoCapture('D:/Object detection/madurai-traffic-counter-master/madurai-traffic-counter-master/input/highway1.mp4')
while True:
    ret, frame = video.read()
    if not ret: break

    h, w = frame.shape[0:2]
    blob = cv2.dnn.blobFromImage(frame, 1/127.5, (300*w//h,300),(127.5,127.5,127.5), False)
    model.setInput(blob)
    output = model.forward()
    
    for i in range(output.shape[2]):
        conf = output[0,0,i,2]
        if conf > CONF_THR:
            label = output[0,0,i,1]
            x0,y0,x1,y1 = (output[0,0,i,3:7] * [w,h,w,h]).astype(int)
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, '{}: {:.2f}'.format(LABELS[label], conf), 
                        (x0,y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(3)
    if key == 27: break

cv2.destroyAllWindows() 




# Human Detection
import cv2
import numpy as np
import imutils

model = cv2.dnn.readNetFromCaffe('D:/Course slides/Datasets/MobileNet-SSD-master/voc/MobileNetSSD_deploy.prototxt',
                                 'D:/Course slides/Datasets/MobileNet-SSD-master/MobileNetSSD_deploy.caffemodel')

CONF_THR = 0.3
LABELS = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
          5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
          10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
          14: 'motorbike', 15: 'person', 16: 'pottedplant',
          17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

c=0

frame = cv2.imread('C:/Users/SiriRamana/Downloads/people.jpg')
frame = imutils.resize(frame, width=1000)

h, w = frame.shape[0:2]
blob = cv2.dnn.blobFromImage(frame, 1/127.5, (300*w//h, 300),
                             (127.5, 127.5, 127.5), False)
model.setInput(blob)
output = model.forward()

for i in range(output.shape[2]):
    conf = output[0,0,i,2]
    if conf > CONF_THR:
        label = output[0,0,i,1]
        x0, y0, x1, y1 = (output[0,0,i,3:7] * [w,h,w,h]).astype(int)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
        cv2.putText(frame, '{}: {:.2f}'. format(LABELS[label], conf),
                    (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

cv2.imshow('frame', frame)
key = cv2.waitKey()

cv2.destroyAllWindows()


