import cv2
import mediapipe as mp
from keras.models import load_model
import utils
import numpy as np
import time



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

label=['Close','Open']

model = load_model('models/drowsiness_new.h5')

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
rpred=[99]
lpred=[99]

CLOSED_EYES_FRAME = 3
CEF_COUNTER = 0
TOTAL_BLINKS = 0

Open_frames = 0
Closed_frames = 0
token = ''
# 0-yawn, 1-no_yawn, 2-Closed, 3-Open

class Video(object):
    def __init__(self,token):
        self.video = cv2.VideoCapture(0)
        self.Closed_frames = Closed_frames
        self.Open_frames = Open_frames
        self.count = count
        self.score = score
        self.label = label
        self.rpred = rpred
        self.lpred = lpred
        self.CEF_COUNTER = CEF_COUNTER
        self.CLOSED_EYES_FRAME = CLOSED_EYES_FRAME
        self.TOTAL_BLINKS = TOTAL_BLINKS
        self.token = token

    def __del__(self):
        self.video.release()
    

    def get_frame(self):
        with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

            
            ret,image = self.video.read()
            height,width = image.shape[:2]

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_coords = utils.landmarksDetection(image,results,False)

                    image = utils.fillPolyTrans(image,[mesh_coords[p] for p in utils.LEFT_EYE],utils.GREEN,opacity=0.3)
                    image = utils.fillPolyTrans(image,[mesh_coords[p] for p in utils.RIGHT_EYE],utils.GREEN,opacity=0.3)

                    # Getting the Blink Ratio
                    ratio = utils.blinkRatio(image,mesh_coords,utils.RIGHT_EYE,utils.LEFT_EYE)
                    cv2.putText(image, f'ratio {ratio}', (100, 100),
                            font, 1.0, utils.GREEN, 2)
                    if ratio > 4.9:
                        self.CEF_COUNTER += 1 
                    cv2.putText(image, 'Blinks', (200, 30),font, 1.3, utils.PINK, 2)

                    if self.CEF_COUNTER>self.CLOSED_EYES_FRAME:
                        self.TOTAL_BLINKS +=1
                        self.CEF_COUNTER = 0 
                    cv2.putText(image, f'Total Blinks {self.TOTAL_BLINKS}', (100, 150),font, 0.6, utils.PINK, 2)                  

                    right_coords = [mesh_coords[p] for p in utils.RIGHT_EYE]
                    left_coords = [mesh_coords[p] for p in utils.LEFT_EYE]

                    # Check if eyes are open or closed
                    crop_right, crop_left = utils.eyesExtractor(image, right_coords, left_coords)
                    print(crop_right.shape)
                    # cv2.imshow("right",crop_right)
                    # cv2.imshow("left",crop_left)

                    for x in crop_right:
                        self.count=self.count+1
                        r_eye = cv2.cvtColor(crop_right,cv2.IMREAD_COLOR)
                        r_eye= r_eye/255
                        r_eye = cv2.resize(r_eye,(145,145))
                        r_eye=  r_eye.reshape(-1,145,145,3)
                        self.rpred = np.argmax(model.predict(r_eye))
                        if(self.rpred==3):
                            self.label='Open' 
                        if(rpred==2):
                            self.label='Closed'
                        break
                    for y in crop_left:
                        self.count=self.count+1
                        l_eye = cv2.cvtColor(crop_left,cv2.IMREAD_COLOR)    
                        l_eye= l_eye/255
                        l_eye = cv2.resize(l_eye,(145,145))
                        l_eye=l_eye.reshape(-1,145,145,3)
                        self.lpred = np.argmax(model.predict(l_eye))
                        if(self.lpred==3):
                            self.label='Open'   
                        if(self.lpred==2):
                            self.label='Closed'
                        break
                    
                   
                    if(self.rpred==2 and self.lpred==2):
                        self.score=self.score+1
                        self.Closed_frames += int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                        cv2.putText(image,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                
                    else:
                        self.score=self.score-1
                        self.Open_frames += int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
                        cv2.putText(image,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                    if(self.score<0):
                        self.score=0   
                    cv2.putText(image,'Frames:'+str(self.score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

                    # Getting the PERCLOS
                    perclos = str(utils.perclos(self.Closed_frames,self.Open_frames))
                    print("PERCLOS: " + perclos)
                    
                # while True: 
                    response=utils.request(perclos,self.TOTAL_BLINKS,self.token)
                    print("response"+str(response))  



            ret,jpg = cv2.imencode('.jpg',image)
            return jpg.tobytes()

        