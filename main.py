import base64
from fastapi import FastAPI
import numpy as np
import cv2
from starlette.requests import Request
from starlette.responses import Response

app = FastAPI()


class Config:
    orm_mode = True


# Carrega Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Recebe via Http/POST base64 de imagem
numeroTotal = 0


@app.post("/hello", response_class=Response)
async def say_hello(request: Request):
    global numeroTotal
    corpo_resposta = await request.body()  # Ler o corpo da requisicao
    # Chama Funcao para ler o total de pessoas na imagem
    numeroTotal = chamaLeitura(corpo_resposta)
    return 'OK'


@app.get("/pessoas")
async def counter_person():
    global numeroTotal
    pessoas = (numeroTotal)
    numeroTotal = 0
    return "Tres pessoas a frente"

@app.get("/bateriaStatus")
async def counter_person(bateria = None):
    print(bateria)
    return 100

def readb64(base64_string):  # Converte o base 64 para leitura do Open CV
    decoded_data = base64.b64decode(base64_string)
    np_data = np.fromstring(decoded_data, dtype=np.uint8)
    return np_data


def chamaLeitura(base64Item):
    # Recebe a Imagem apos converter do Base64
    img = cv2.imdecode(readb64(base64Item), 1)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Altera o tamanho da Imagem
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Detecta os Objetos
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Retangulos para sinalizar as pessoas reconhecidas
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    count = 0
    font = cv2.FONT_HERSHEY_PLAIN
    # Percorre todos os objetos identificados para marcar qual e pessoas
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                count = count + 1
    return count  # Numero de pessoas
