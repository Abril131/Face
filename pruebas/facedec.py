import cv2
import os
import face_recognition

#codi face

imagesFacesPath = r"C:\Users\pukia\Documents\cuatri8\IA\FaceProject\faces"

facesEncodi = []
facesName = []

for filename in os.listdir(imagesFacesPath):
    image = cv2.imread(imagesFacesPath + "/" + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f_condi = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
    facesEncodi.append(f_condi)
    facesName.append(filename.split(".")[0])

#print(facesEncodi)
#print(facesName)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame,1)
    ori = frame.copy()
    faces = faceClassif.detectMultiScale(frame, 1.1,5)

    for (x, y, w, h) in faces:
        face = ori[y:y +h, x:x + w]
        face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
        resul = face_recognition.compare_faces(facesEncodi, actual_face_encoding)
        print(resul)
        if True in resul:
            index = resul.index(True)
            name = facesName[index]
            color = (125, 220, 0)
        else:
            name = "Desconocido"
            color = (50, 50, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y +h ), color, 2)
        cv2.putText(frame, name, (x, y + h + 25),2,1, (255, 255,255), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()