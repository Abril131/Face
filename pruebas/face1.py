import cv2
import face_recognition
import os
import socket
import pickle

# Cargar imágenes de referencia y codificar rostros
images_faces_path = r"/faces"
faces_encodings = []
faces_names = []

for filename in os.listdir(images_faces_path):
    image = face_recognition.load_image_file(os.path.join(images_faces_path, filename))
    face_encoding = face_recognition.face_encodings(image)[0]
    faces_encodings.append(face_encoding)
    faces_names.append(filename.split(".")[0])

# Iniciar la captura de vídeo desde la cámara
cap = cv2.VideoCapture(0)

# Configuración del socket
HOST = '127.0.0.1'  # Dirección IP del servidor
PORT = 8080# Puerto para escuchar las conexiones

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()

    print("Servidor de reconocimiento facial esperando conexiones...")
    conn, addr = server_socket.accept()

    with conn:
        print('Conexión establecida desde', addr)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir el marco a RGB (ya que OpenCV carga en BGR)
            rgb_frame = frame[:, :, ::-1]

            # Encontrar todas las caras en el marco
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                # Comparar la cara encontrada con las caras de referencia
                matches = face_recognition.compare_faces(faces_encodings, face_encoding)

                name = "Desconocido"

                # Si encontramos una coincidencia, usar el nombre correspondiente
                if True in matches:
                    match_index = matches.index(True)
                    name = faces_names[match_index]

                # Enviar los resultados al cliente a través del socket
                conn.sendall(pickle.dumps(name))

                # Mostrar el marco resultante
                cv2.imshow('Reconocimiento Facial', frame)

            # Salir del bucle si se presiona la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
