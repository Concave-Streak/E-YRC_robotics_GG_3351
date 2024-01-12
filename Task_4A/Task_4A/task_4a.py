'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			GG_3351
# Author List:		Raaja Das, ..
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
##############################################################



################# ADD UTILITY FUNCTIONS HERE #################

"""
You are allowed to add any number of functions to this code.
"""

def classify_event(image):
    ''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''
    '''
    ADD YOUR CODE HERE
    '''

    device = "cpu"
    categories = ["combat","destroyedbuilding","fire","humanitarianaid","militaryvehicles"]

    path = "models/model_1.pth"
    model = torch.load(path)
    model.to(device)

    data_image = torchvision.io.read_image(str(image)).type(torch.float32)
    data_image /= 255

    IMG_SIZE = (256, 256)

    data_transform = transforms.Compose([
    transforms.Resize(size=IMG_SIZE),
    ])

    data_image_transformed = data_transform(data_image)

    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to image
        data_image_transformed_with_batch_size = data_image_transformed.unsqueeze(dim=0)
       
        # Make a prediction on image with an extra dimension
        data_image_pred = model(data_image_transformed.unsqueeze(dim=0).to(device))

    data_image_pred_label = torch.argmax(data_image_pred, dim=1)

    data_image_pred_class = categories[data_image_pred_label.cpu()]

    event = data_image_pred_class
    return event


##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable
    
    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can 
    return the dictionary from a user-defined function and just call the 
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """  
    identified_labels = {}  
    
##############	ADD YOUR CODE HERE	##############
    
    cam = cv2.VideoCapture(1)
    result, image = cam.read()


    # path = "test images/image 2.jpg"
    # image = cv2.imread(path)

    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #image = image[120:510,100:400]

    THRESHOLD = 150

    lower_bound = 30
    higher_bound = 50

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (thresh, img_bin) = cv2.threshold(img, THRESHOLD, 255,cv2.THRESH_BINARY)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    #(contours, boundingBoxes) = contours, hierarchy

    image_out = image
    dir_path = "images/"
    idx_to_letter = ["A", "B", "C", "D", "E"]
    idx = 0
    for index, contour in enumerate(contours):
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(contour)
        if (w >= lower_bound and h >= lower_bound) and (w <= higher_bound and h <= higher_bound) and (hierarchy[0,index,2] == -1):
            new_img = image[y:y+h, x:x+w]

            img_pth = dir_path + "img_" + str(idx) + '.png'
            cv2.imwrite(img_pth, new_img)

            event_label = classify_event(img_pth)
            identified_labels[idx_to_letter[idx]] = event_label

            image_out = cv2.rectangle(image_out, (x, y), (x + w, y + h), (0,255,0), 1)
            cv2.putText(image_out, event_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            idx += 1

    cv2.imshow("GG_3351", image_out) 

    cv2.waitKey(0) 

    cv2.destroyAllWindows() 

##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)