import cv2
import os

imagesPath = r"C:\Users\pukia\Documents\cuatri8\IA\FaceProject\Imagen"

if not os.path.exists("../faces"):
    os.makedirs("../faces")
    print("NC: faces")

#DEtectir de face

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
for imageName in os.listdir(imagesPath):
    print(imageName)
    image = cv2.imread(imagesPath + "/" + imageName)
    faces = faceClassif.detectMultiScale(image, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (224, 224))
        cv2.imwrite("faces/" + str(count) + ".jpg", face)
        count += 1
       # cv2.imshow("face", face)
      #  cv2.waitKey(0)
 #   cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)  # E
    cv2.imshow("Imagen", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()