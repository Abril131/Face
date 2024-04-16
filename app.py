from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import threading

app = Flask(__name__)

# Cargar imágenes de rostros conocidos
images_faces_path = r"C:\Users\pukia\Documents\cuatri8\IA\FaceProject\faces"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(images_faces_path):
    image = cv2.imread(os.path.join(images_faces_path, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(filename)[0])


def process_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 3 == 0:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame = cv2.flip(frame, 1)
            frame, face_locations, face_names = detect_faces(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def detect_faces(frame):
    # Convertir a escala de grises para una detección más rápida
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectar caras en la imagen en escala de grises
    face_locations = face_recognition.face_locations(frame)

    for (top, right, bottom, left) in face_locations:
        # Obtener la región de interés (ROI) para el reconocimiento facial
        roi_color = frame[top:bottom, left:right]

        # Convertir la imagen de la ROI de BGR a RGB (necesario para face_recognition)
        rgb_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        # Realizar el reconocimiento facial en la ROI
        face_encodings = face_recognition.face_encodings(rgb_roi)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            # Comparar la cara con las caras conocidas
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconocido"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                color = (0, 255, 0)  # Verde para rostros conocidos
            else:
                color = (0, 0, 255)  # Rojo para rostros desconocidos
            # Dibujar un rectángulo alrededor de cada cara
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            # Mostrar el nombre de la persona
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, face_locations, []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video)
    video_thread.start()
    app.run(debug=True)
