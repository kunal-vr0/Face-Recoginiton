#import frTrainer
import cv2
import face_recognition as fr
import pickle

from frTrainer import MasterEncodings, MasterNames
font = cv2.FONT_HERSHEY_COMPLEX
height = 540
width = 1080
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))



while True:
    with open('train.pkl', 'rb') as tr:
        MasterNames = pickle.load(tr)
        MasterEncodings = pickle.load(tr)

    ig, frame = cam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faceLocs = fr.face_locations(frameRGB)
    faceEncodings = fr.face_encodings(frameRGB, faceLocs)
    for faceLoc, faceEncoding in zip(faceLocs, faceEncodings):
        t, r, b, l = faceLoc
        name = 'UNKNOWN'
        cv2.rectangle(frame, (l,t),(r,b), (0,255,0), 3)
        matches = fr.compare_faces(MasterEncodings, faceEncoding)
        if True in matches:
            index = matches.index(True)
            name = MasterNames[index]
        cv2.putText(frame, name, (l, t), font, 1, (255,0,0), 3)
    cv2.imshow('CAM', frame)
    cv2.moveWindow('CAM', 0,0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()
    

