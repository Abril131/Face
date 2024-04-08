import asyncio

import numpy as np
import websockets
import json
import cv2
import face_recognition
import os


class FaceDetectionModel:
    def __init__(self):
        self.face_encodings = []
        self.face_names = []

    def load_known_faces(self, images_path):
        for filename in os.listdir(images_path):
            image = cv2.imread(os.path.join(images_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(image)[0]
            self.face_encodings.append(face_encoding)
            self.face_names.append(os.path.splitext(filename)[0])

    def detect_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return face_locations, face_encodings

    def compare_faces(self, face_encodings):
        results = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(self.face_encodings, encoding)
            name = "Desconocido"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.face_names[first_match_index]
            results.append(name)
        return results


class FaceDetectionController:
    def __init__(self, model, websocket):
        self.model = model
        self.websocket = websocket

    async def process_frame(self, frame):
        face_locations, face_encodings = self.model.detect_faces(frame)
        face_names = self.model.compare_faces(face_encodings)

        # Convertir la lista de nombres de caras a una cadena JSON
        json_data = json.dumps(face_names)

        # Enviar los nombres de las caras detectadas al cliente a través del websocket
        await self.websocket.send(json_data)


async def process_frame(websocket, path):
    model = FaceDetectionModel()
    model.load_known_faces(r"C:\Users\pukia\Documents\cuatri8\IA\FaceProject\faces")

    async for data in websocket:
        frame_data = bytearray(data)
        frame = cv2.imdecode(np.array(frame_data), cv2.IMREAD_COLOR)

        face_locations, face_encodings = model.detect_faces(frame)
        face_names = model.compare_faces(face_encodings)

        detected_faces = [{"location": location, "name": name} for location, name in zip(face_locations, face_names)]
        await websocket.send(json.dumps(detected_faces))


start_server = websockets.serve(process_frame, "localhost", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
