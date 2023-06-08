import os
from modelFunctions import *

path = "Photo"
imgs = []
classNames = []
mList = os.listdir(path)

for cl in mList:
    curIMG = cv2.imread(f"{path}/{cl}")
    imgs.append(curIMG)
    classNames.append(os.path.splitext(cl)[0])

print("Loading Datas...")

encodeListKnown = loadTrainedDatas()


print("Starting The Program...")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
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
            y1, x2, y2, x1 = faceLok
            y1, x2, y2, x1 = y1 - 40, x2 - 10, y2 + 40, x1 + 10
            # print(y1, x2, y2, x1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            y1, x2, y2, x1 = faceLok
            y1, x2, y2, x1 = y1 - 40, x2 - 10, y2 + 40, x1 + 10
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                "Nainasm",
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
