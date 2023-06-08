import pickle
import face_recognition
import cv2
import os
from PIL import Image
import base64
import io
import numpy as np


def TrainFacesDataSets():
    print("Labeling The Images...")
    path = "Photo"
    imgs = []
    classNames = []
    mList = os.listdir(path)

    for cl in mList:
        curIMG = cv2.imread(f"{path}/{cl}")
        imgs.append(curIMG)
        classNames.append(os.path.splitext(cl)[0])

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    print("Encoding...")

    encodeListKnown = findEncodings(imgs)
    with open("dataset_faces.dat", "wb") as f:
        pickle.dump(encodeListKnown, f)

    print("Encoding Complete!")


def loadTrainedDatas():
    encodeListKnown = ""
    with open("dataset_faces.dat", "rb") as f:
        encodeListKnown = pickle.load(f)
    print("Datas Loaded!")

    return encodeListKnown


def recognizeByPhoto(imgPath):
    # Initialise your data
    arr = imgPath
    b = bytes(arr, "utf-8")
    z = b[b.find(b"/9") :]
    im = Image.open(io.BytesIO(base64.b64decode(z))).save("result.jpg")

    path = "Photo"
    imgs = []
    classNames = []
    mList = os.listdir(path)

    for cl in mList:
        curIMG = cv2.imread(f"{path}/{cl}")
        imgs.append(curIMG)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = ""

    with open("dataset_faces.dat", "rb") as f:
        encodeListKnown = pickle.load(f)

    img = cv2.imread("result.jpg", 0)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFr = face_recognition.face_locations(imgS)
    encodecurFr = face_recognition.face_encodings(imgS, facesCurFr)
    for encodeFace, faceLok in zip(encodecurFr, facesCurFr):
        mathes = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        if mathes[matchIndex]:
            name = classNames[matchIndex].upper()
            return name
        else:
            return "Nainasm"
