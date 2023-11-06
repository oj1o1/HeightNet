import numpy as np
import cv2
import time
import os
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(proto,mobile_ssd)
    
cap = cv2.VideoCapture


COLORS = (0,255,0)
while cap.isOpened():
    
    tic = time.time()
    ret,frame = cap.read()
    
    if not ret:
        break
    
    #get frame size, h = 720, w = 1080
    (h, w) = frame.shape[:2]
    
    #input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD implementation)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,size=(300,300),mean=127.5)
    
    # pass the blob through the network and obtain the detections and predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward() # size is 1*1*N*7 [0,class_index,probability,x1, y1, x2, y2]
    
    detections = detections[0,0,:] # size is N*7 [0,class_index,probability,x1, y1, x2, y2]

    #select detecions in person and confidence_threshold > 0.2 , return detections # size is N*5 [probability,x1, y1, x2, y2]
    detections = np.asarray([det[2:] for det in detections if int(det[1]) == CLASSES.index('person') and det[2] > confidence_threshold])
    
    '''height pedestrain analysis part'''
    if detections.size > 0:
        '''
        remove other pedestrain box, just leave full body box using aspect ratio
        '''
        # get x, y coordeiante from detections
        box_coordinate = detections[:,[1,2,3,4]]
        box_coordinate = (box_coordinate * np.array([w, h, w, h])).astype("int")

        #get each pedestrian box aspect ratio, convert (N,) to (N,1)
        aspect_ratio = (box_coordinate[:,2]-box_coordinate[:,0]) / (box_coordinate[:,3]-box_coordinate[:,1])
        New_aspect_ratio = aspect_ratio.reshape((aspect_ratio.size,1))

        #define aspect ratio range 
        aspect_ratio_range_lower = 0.35 
        aspect_ratio_range_upper = 0.45 
        
        #add New_aspect_ratio column to box_coordinate, (N * 5)
        New_box_coordinate = np.append(box_coordinate,New_aspect_ratio, axis = 1)
        
        Final_box_coordinate_list = []

        for idx, val in enumerate(New_box_coordinate):
            
            # if aspect ratio in range(0.35,0.45), processing and ddraw reference plane
            if aspect_ratio_range_lower < val[4] < aspect_ratio_range_upper:
                
                #create a new box coordinate abd convert from list to array
                Final_box_coordinate_list.append(New_box_coordinate[idx])
                Final_box_coordinate_array = np.array(Final_box_coordinate_list)
                
                # get new box x,y corrdinate
                Final_Height_Analysis_range = Final_box_coordinate_array[:,[1,3]]
                Final_Width_Analysis_range = Final_box_coordinate_array[:,[0,2]]
                
                #calculate height and width range
                Final_Height_range = Final_box_coordinate_array[:,3]-Final_box_coordinate_array[:,1]
                Final_Width_range = Final_box_coordinate_array[:,2]-Final_box_coordinate_array[:,0]

                #sort height and return medium one
                Height_median = np.median(Final_box_coordinate_array[:,3]-Final_box_coordinate_array[:,1])
                Height_mean = np.mean(Final_box_coordinate_array[:,3]-Final_box_coordinate_array[:,1])
                
                #get median and mean index 
                median_index = np.nanargmin(np.abs(Final_Height_range-Height_median))
                mean_index = np.nanargmin(np.abs(Final_Height_range-Height_mean))
                
                #from indxe, find its coordinate
                median_index_y1 = Final_Height_Analysis_range[median_index][0]
                median_index_y2 = Final_Height_Analysis_range[median_index][1]
        
                mean_index_y1 = Final_Height_Analysis_range[mean_index][0]
                mean_index_y2 = Final_Height_Analysis_range[mean_index][1] 
        
                #define reference plane scale factor
                reference_plane_scale = 0.2