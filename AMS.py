import cv2    #CV-Computer Vision (Used to process images and videos, manipulating and retrive data from it)
import face_recognition   #used to detect faces and recognition
import os    #provides functions for interacting with OS for path
from flask import Flask,request,render_template    #used for web framework design
from datetime import date   #to get date 
from datetime import datetime    #To get time
import numpy as np     #to work with numerical arrays
import csv      # to store data in excel sheet
import pandas as pd     #to work with data (datasets)

#### Defining Flask App (intialisation)
app = Flask(__name__)

cap = cv2.VideoCapture(0)    #to turn on first camera

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

now = datetime.now()   #to get present datetime
current_date = now.strftime("%Y-%m-%d")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #to detect face from an image
try:
    cap = cv2.VideoCapture(1)     #to turn on second camera if first camera not working
except:
    cap = cv2.VideoCapture(0)

#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
    
if not os.path.isdir('pics'):
    os.makedirs('pics')
    
if f'Attendance/{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/{datetoday}.csv','a') as f:
        f.write('Name,Roll,Time')


## To extract the faces 
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

## To extract data from datetoday.csv 
def extract_attendance():
    df = pd.read_csv(f'Attendance/{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

## Add Attendance of a specific user
def add_attendance(name):    #name format : sravani_180497
    username = name.split('_')[0]  #sravani
    userid = name.split('_')[1]    #180497 
    current_time = datetime.now().strftime("%H:%M:%S")  #hours : minutes : seconds
    df = pd.read_csv(f'Attendance/{datetoday}.csv')
    if int(userid) not in list(df['Roll']):    
        with open(f'Attendance/{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')

    
##For already registered    
path="C:/Users/Jahnavi/FACE RECOGNITION SYSTEM/FACE RECOGNITION/FRS_using face_recognition/pics"
known_faces_names=[]
images=[]
mylist=os.listdir(path)
for x in mylist:
    curImg=cv2.imread(f'{path}/{x}')
    images.append(curImg)
    known_faces_names.append(os.path.splitext(x)[0])
    
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

known_face_encoding=findEncodings(images)

video_capture = cv2.VideoCapture(0) 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')   
def ece():
    names,rolls,times,l = extract_attendance()    
    return render_template('cse.html',names=names,rolls=rolls,times=times,l=l,datetoday2=datetoday2)

#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
                face_names.append(name)
                if name in known_faces_names:
                    cv2.putText(frame,f'{name}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                    if name in students:
                        students.remove(name)
                        add_attendance(name)
                else:
                    cv2.putText(frame,'Spoof',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                    
        cv2.imshow("Attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('cse.html',names=names,rolls=rolls,times=times,l=l,datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,newusername,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            name = newusername+'_'+newuserid+'.jpg'
            cv2.imwrite(path+'/'+name,frame[y:y+h,x:x+w])       
            break
        cv2.imshow("Attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()    
    return render_template('cse.html',names=names,rolls=rolls,times=times,l=l,datetoday2=datetoday2)
                                

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)

    
