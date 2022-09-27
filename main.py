import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from collections import deque
from scipy.spatial import distance as dist

from tensorflow.lite.python.interpreter import Interpreter

interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

cap = cv.VideoCapture("servingv2.mp4")

EDGES = {
    # (0, 1): 'm',
    # (0, 2): 'c',
    # (1, 3): 'm',
    # (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
paintedPoints = deque(maxlen=30)
savedPaintedPoints = []
interestPoint = 0
frameArray = []
novoArray = deque(maxlen=30)


def motionTrail(paintedPoints):
    mediaX1 = np.sum(paintedPoints[0][0] + paintedPoints[1][0] + paintedPoints[2][0] + paintedPoints[3][0])
    mediaY1 = np.sum(paintedPoints[0][1] + paintedPoints[1][1] + paintedPoints[2][1] + paintedPoints[3][1])

    mediaX2 = np.sum(paintedPoints[4][0] + paintedPoints[5][0] + paintedPoints[6][0] + paintedPoints[7][0])
    mediaY2 = np.sum(paintedPoints[4][1] + paintedPoints[5][1] + paintedPoints[6][1] + paintedPoints[7][1])

    novoArray.appendleft([int(mediaX1 / 4), int(mediaY1 / 4)])
    novoArray.appendleft([int(mediaX2 / 4), int(mediaY2 / 4)])

    for i in np.arange(1, len(novoArray)):
        if novoArray[i - 1] is None or novoArray[i] is None:
            continue
        # thickness = int(np.sqrt(args1["buffer"] / float(i + 1)) * 2.5)
        if(dist.euclidean(novoArray[i], novoArray[i - 1]) < 60):
            cv.line(frame, novoArray[i - 1], novoArray[i], (211, 0, 148), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (211, 0, 148), 2)
        else:
            cv.line(frame, novoArray[i - 1], novoArray[i], (0, 0, 148), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (0, 0, 148), 2)


def getAngle(pt1, pt2, pt3):
    ypt1, xpt1 = pt1

    radians = np.arctan2(pt3[1] - pt1[1], pt3[0] - pt1[0]) - np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle = round(np.abs(radians * 180.0 / np.pi))  # conversao

    if angle > 180:
        angle = 360 - angle
    cv.putText(frame2, str(angle), (int(xpt1), int(ypt1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            cv.circle(frame2, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


while cap.isOpened():
    ret, frame = cap.read()
    key = cv.waitKey(10)
    # ---------------loop video---------------
    if not ret:
        novoArray.clear()
        paintedPoints.clear()
        cv.destroyAllWindows()
        cap = cv.VideoCapture("servingv2.mp4")
        hasFrame, frame = cap.read()

    frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
    frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta

    # reshape imagem para 192x192x3
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    plt.imshow(tf.cast(np.squeeze(img), dtype=tf.int32))

    # inputs e ouputs
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # predicoes e pontos
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    print(keypoints_with_scores)

    # desenha o frame com os pontos
    draw_connections(frame, keypoints_with_scores, EDGES, 0.20)
    draw_keypoints(frame, keypoints_with_scores, 0.20)



    # ------------------------ANGLES---------------------------------
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
    shaped = shaped.astype(int)
    shaped = np.delete(shaped, 2, 1)

    getAngle(shaped[8], shaped[6], shaped[10])  # cotovelo dirrrr
    getAngle(shaped[7], shaped[5], shaped[9])  # cotovelo esq22222222
    getAngle(shaped[14], shaped[12], shaped[16])  # joelho dir
    getAngle(shaped[13], shaped[11], shaped[15])  # joelho esq

    # ----------------------DESENHA LINHA MOTION TRACKING--------------------
    # mao direita
    if key == ord("1"):
        paintedPoints.clear()
        interestPoint = 10
    # mao equerda
    if key == ord("2"):
        paintedPoints.clear()
        interestPoint = 9
    # joelho direito
    if key == ord("3"):
        paintedPoints.clear()
        interestPoint = 14
    # joelho esquerdo
    if key == ord("4"):
        paintedPoints.clear()
        interestPoint = 13
    # quadril direito
    if key == ord("5"):
        paintedPoints.clear()
        interestPoint = 12
    # quadril esquerdo
    if key == ord("6"):
        paintedPoints.clear()
        interestPoint = 11
    # pe direito
    if key == ord("7"):
        paintedPoints.clear()
        interestPoint = 16
    # pe esquerdo
    if key == ord("8"):
        paintedPoints.clear()
        interestPoint = 15

    if interestPoint is not 0:
        paintedPoints.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])
        savedPaintedPoints.append(shaped[interestPoint])

    if len(paintedPoints) > 7:
        motionTrail(paintedPoints)

    frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
    frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
    frameArray.append(frame2)
    cv.imshow("tela2", frame2)
    cv.imshow("tela", frame)

    if key == ord('q'):
        break
    if key == ord('p'):
        cv.waitKey(-1)  # wait until any key is pressed

        # ------------------MANIPULANDO VIDEO/FRAMES-----------------q
    if key == ord('j'):
        cv.destroyWindow('tela')
        cv.destroyWindow('tela2')
        frame3 = np.zeros((255, 255, 3), np.uint8)
        cv.imshow('Frame3', frameArray[0])
        counter = 0
        i = 0

        while key != ord('q'):
            # -------------START/PAUSE------------------
            if key == ord('p'):
                while True:
                    if i > 0:
                        for i in np.arange(i, len(frameArray)):
                            cv.imshow('Frame3', frameArray[i])
                            key = cv.waitKey(50)
                            if key == ord('r'):
                                i = 0
                                break
                            if key == ord('p'):
                                break
                    else:
                        for i in np.arange(1, len(frameArray)):
                            cv.imshow('Frame3', frameArray[i])
                            key = cv.waitKey(50)
                            if key == ord('r'):
                                i = 0
                                break
                            if key == ord('p'):
                                break
                    break

            # -----------AVANÇAR FRAME-------------
            if key == ord('l'):
                if i < len(frameArray) - 1:
                    i = 1 + i
                    cv.imshow('Frame3', frameArray[i])

            # ----------VOLTAR FRAME--------------
            if key == ord('k'):
                if i > 0:
                    i = i - 1
                    cv.imshow('Frame3', frameArray[i])

            # ----------RESTART------------------
            if key == ord('r'):
                i = 0
                cv.imshow('Frame3', frameArray[i])

            key = cv.waitKey(1)
            if key == ord('q'):
                break
        cv.destroyAllWindows()

cap.release()
cv.destroyWindow()
