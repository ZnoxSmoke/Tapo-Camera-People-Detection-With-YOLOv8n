import cv2
from ultralytics import YOLO

USER = ""
PASSWORD = ""
IP = ""
url = f"rtsp://{USER}:{PASSWORD}@{IP}:554/stream1"

model = YOLO('yolov8n.onnx')

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: No se pudo conectar al flujo RTSP de la cámara")
    exit()

cv2.namedWindow("Camara de abitacion - Deteccion de personas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camara de abitacion - Deteccion de personas", 800, 415)

print("Conexio exitosa, presione 'q' para salir.")

while True:
    cap.grab()
    cap.grab()

    ret, frame = cap.read()

    if not ret:
        print("Se perdio la conexion con el cuadro de video")
        break

    results = model(frame, classes=[0], verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Camara de abitacion - Deteccion de personas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

