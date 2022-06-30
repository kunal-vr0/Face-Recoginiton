import os
import face_recognition as fr
import pickle
myDir = 'F:\\openCV\Trainer'
MasterEncodings = []
MasterNames = []
for root, dirs, files in os.walk(myDir):
    for file in files:
        filePath = os.path.join(root, file)
        name, ext  = os.path.splitext(file)
        if ext == '.jpeg' or ext == '.png' or ext == '.jpg':
            person = fr.load_image_file(filePath)
            encoding = fr.face_encodings(person)[0]
            MasterEncodings.append(encoding)
            MasterNames.append(name)
with open('train.pkl', 'wb') as tr:
    pickle.dump(MasterNames, tr)
    pickle.dump(MasterEncodings, tr)
