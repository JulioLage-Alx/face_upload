import cv2 
import mediapipe as mp 

webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecimento_rosto = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador, frame = webcam.read()
    if not verificador:
         break
    lista_rosto = reconhecimento_rosto.process(frame)

    if lista_rosto.detections:
         for rosto in lista_rosto.detections:
              desenho.draw_detection(frame,rosto)
    cv2.imshow("Rosto na webcam",frame)

    if cv2.waitKey(5) == 27:
         break

webcam.release()
cv2.destroyAllWindows()
