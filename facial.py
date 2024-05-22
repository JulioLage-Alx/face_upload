import cv2 
import mediapipe as mp 

webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecimento_rosto = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

# Dicionário para armazenar os dados de cada rosto reconhecido por nome
dados_reconhecidos = {}

while True:
    verificador, frame = webcam.read()
    if not verificador:
         break
    lista_rosto = reconhecimento_rosto.process(frame)

    if lista_rosto.detections:
         for rosto in lista_rosto.detections:
              # Extraindo as coordenadas do retângulo delimitador de cada rosto detectado
              bboxC = rosto.location_data.relative_bounding_box
              ih, iw, _ = frame.shape
              x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
              
              # Obtendo o nome da pessoa associada a este rosto (aqui você pode ter um método de reconhecimento de rosto)
              nome = "Nome_da_Pessoa"  # Substitua "Nome_da_Pessoa" pelo nome real da pessoa ou pelo resultado do seu método de reconhecimento de rosto
              
              # Armazenando os dados do rosto no dicionário dados_reconhecidos
              if nome in dados_reconhecidos:
                  dados_reconhecidos[nome].append({'x': x, 'y': y, 'w': w, 'h': h})
              else:
                  dados_reconhecidos[nome] = [{'x': x, 'y': y, 'w': w, 'h': h}]

              # Desenhando o retângulo delimitador ao redor do rosto
              desenho.draw_detection(frame, rosto)

              # Escrevendo o nome da pessoa abaixo do rosto
              cv2.putText(frame, nome, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
              
    cv2.imshow("Rosto na webcam",frame)

    if cv2.waitKey(5) == 27:
         break

webcam.release()
cv2.destroyAllWindows()
