from flask.globals import session
from scipy.spatial import distance
import math
import cv2
import numpy as np
import requests
import time
from flask import request as Request
from playsound import playsound


BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (255,0,0)
RED = (0,0,255)
CYAN = (255,255,0)
YELLOW =(0,255,255)
MAGENTA = (255,0,255)
GRAY = (128,128,128)
GREEN = (0,255,0)
PURPLE = (128,0,128)
ORANGE = (0,165,255)
PINK = (147,20,255)
color=(225, 0, 100)

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 


Base_Url= 'http://localhost:8000'

# Eucledian Distance
def eucledianDistance(point,point1):
  x,y  =point
  x1,y1  =point1
  distance = math.sqrt((x1-x)**2 +(y1-y)**2)
  return distance

# Blink Ratio 

def blinkRatio(img,landmarks,right_indices,left_indices):
  # Right Eye Horizontal Plane
  rh_right = landmarks[right_indices[0]]
  rh_left=landmarks[right_indices[8]]

  # Right Eye vertical Plane
  rv_top = landmarks[right_indices[12]]
  rv_bottom=landmarks[right_indices[4]]

   # Left Eye Horizontal Plane
  lh_right = landmarks[left_indices[0]]
  lh_left=landmarks[left_indices[8]]

  # Left Eye vertical Plane
  lv_top = landmarks[left_indices[12]]
  lv_bottom=landmarks[left_indices[4]]

  rhDistance = eucledianDistance(rh_right,rh_left)
  rvDistance = eucledianDistance(rv_top,rv_bottom)

  lhDistance = eucledianDistance(lh_right,lh_left)
  lvDistance = eucledianDistance(lv_top,lv_bottom)

  reRatio = rhDistance/rvDistance
  leRatio = lhDistance/lvDistance

  ratio = (reRatio+leRatio)/2
  return ratio

# EAR
def ear(landmarks,indices):
  p1=landmarks[indices[0]]
  p2=landmarks[indices[11]]
  p3=landmarks[indices[13]]
  p4=landmarks[indices[15]]
  p5=landmarks[indices[3]]
  p6=landmarks[indices[5]]

  ear = (eucledianDistance(p2,p6) + eucledianDistance(p3,p5)) / (2.0 * eucledianDistance(p1,p4))
  return ear

# Eyes Extractor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
  # converting color image to  scale image 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # getting the dimension of image 
  dim = gray.shape

  # creating mask from gray scale dim
  mask = np.zeros(dim, dtype=np.uint8)

  # drawing Eyes Shape on mask with white color 
  cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
  cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

  # showing the mask 
  # cv2.imshow('mask', mask)
  
  # draw eyes image on mask, where the white shapes are the eyes
  eyes = cv2.bitwise_and(gray, gray, mask=mask)
  # change black color to gray 
  # cv2.imshow('eyes draw', eyes)
  eyes[mask==0]=155
  
  # getting minium and maximum x and y  for right and left eyes 
  # For Right Eye 
  r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
  r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
  r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
  r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

  # For LEFT Eye
  l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
  l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
  l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
  l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

  # croping the eyes from mask 
  cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
  cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

  # returning the cropped eyes 
  return cropped_right, cropped_left

# play an alarm sound
def sound_alarm():
	playsound('alarm/alarm.wav')


# PERCLOS
def perclos(frame_closed,frame_open):
  perclos = frame_closed/(frame_open+frame_closed) * 100
  return perclos  

# landmark detection function
def landmarksDetection(img,results,draw=False):
  img_height,img_width =img.shape[:2]
  mesh_coords = [(int(point.x * img_width),(int(point.y*img_height))) for point in results.multi_face_landmarks[0].landmark] 
  if draw:
      [cv2.circle(img,p,2,color,-1) for p in mesh_coords]

  return mesh_coords

def fillPolyTrans(img, points, color, opacity):
  list_to_np_array = np.array(points, dtype=np.int32)
  overlay = img.copy()  # coping the image
  cv2.fillPoly(overlay,[list_to_np_array], color )
  new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
  # print(points_list)
  img = new_img
  cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
  return img

def register(name,email,password,confirm):
  data = {'name':name,'email':email,'password':password,'password_confirmation':confirm}
  url = Base_Url + '/api/register'
  response = requests.post(url, data=data)
  return response.json()

def login(email,password):
  data = {'email':email,'password':password}
  url = Base_Url + '/api/login'
  response = requests.post(url, data=data)
  if response.status_code == 200:
    return response.json()
  

def request(perclos,blinks,token):
  headers = {'Authorization':'Bearer '+token}
  data = {'perclos':perclos,'blinks':blinks}
  url = Base_Url + '/api/drowsiness'
  response = requests.post(url, data=data, headers=headers)
  return response.json()

def profile(token):
  headers = {'Authorization':'Bearer '+token}
  url = Base_Url + '/api/user'
  response = requests.get(url, headers=headers)
  if response.status_code == 200:
    return response.json()

def logout(token):
  headers = {'Authorization':'Bearer '+token}
  url = Base_Url + '/api/logout'
  response = requests.get(url, headers=headers)
  if response.status_code != 200:
    return response.json()

def signals(token):
  headers = {'Authorization':'Bearer '+token}
  url = Base_Url + '/api/signals'
  response = requests.get(url, headers=headers)
  if response.status_code == 200:
    return response.json()
 

