from ctypes.wintypes import POINT

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt3
import cv2 as cv
from collections import deque
from scipy.spatial import distance as dist
from tkinter import Label, Tk, Button, ttk
from PIL import Image
from PIL import ImageTk

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework.test_ops import old

interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

video = "mauro4v2.mp4"
cap = cv.VideoCapture(video)

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
pointsToPaint = deque(maxlen=40)
interestPoint = 0
frameArray = []
frameArrayOriginal = []
novoArray = deque(maxlen=40)
i = 0


def clickEvent(event, x, y, flag, param):
    yFrame, xFrame, cFrame = frame.shape
    if event == cv.EVENT_LBUTTONDOWN:
        cv.line(frame, (x, y), (xFrame, y), (255, 0, 255), 2)
        cv.line(frame, (x, y), (0, y), (255, 0, 255), 2)

        cv.line(frame, (x, y), (x, yFrame), (255, 0, 255), 2)
        cv.line(frame, (x, y), (x, 0), (255, 0, 255), 2)
        cv.imshow("tela", frame)


def motionTrail(paintedPoints):
    mediaX1 = np.sum(
        paintedPoints[0][0] + paintedPoints[1][0] + paintedPoints[2][0] + paintedPoints[3][0] + paintedPoints[4][0] +
        paintedPoints[5][0] + paintedPoints[6][0] + paintedPoints[7][0])
    mediaY1 = np.sum(
        paintedPoints[0][1] + paintedPoints[1][1] + paintedPoints[2][1] + paintedPoints[3][1] + paintedPoints[4][1] +
        paintedPoints[5][1] + paintedPoints[6][1] + paintedPoints[7][1])

    mediaX2 = np.sum(
        paintedPoints[8][0] + paintedPoints[9][0] + paintedPoints[10][0] + paintedPoints[11][0] + paintedPoints[12][0] +
        paintedPoints[13][0] + paintedPoints[14][0] + paintedPoints[15][0])
    mediaY2 = np.sum(
        paintedPoints[8][1] + paintedPoints[9][1] + paintedPoints[10][1] + paintedPoints[11][1] + paintedPoints[12][1] +
        paintedPoints[13][1] + paintedPoints[14][1] + paintedPoints[15][1])

    novoArray.appendleft([int(mediaX1 / 8), int(mediaY1 / 8)])
    novoArray.appendleft([int(mediaX2 / 8), int(mediaY2 / 8)])

    for i in np.arange(1, len(novoArray)):
        if novoArray[i - 1] is None or novoArray[i] is None:
            continue

        if dist.euclidean(novoArray[i], novoArray[i - 1]) < 60:
            cv.line(frame, novoArray[i - 1], novoArray[i], (255, 0, 255), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (255, 0, 255), 2)
        else:
            cv.line(frame, novoArray[i - 1], novoArray[i], (0, 255, 255), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (0, 255, 255), 2)


def getAngle(pt1, pt2, pt3, index):
    ypt1, xpt1 = pt1

    radians = np.arctan2(pt3[1] - pt1[1], pt3[0] - pt1[0]) - np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle = round(np.abs(radians * 180.0 / np.pi))  # conversao radianos para graus

    if angle > 180:
        angle = 360 - angle
    cv.putText(frame2, str(angle), (int(xpt1), int(ypt1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    if index == 5:
        cv.putText(frameAngles, "OMBRO E.: " + str(angle), (0, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 6:
        cv.putText(frameAngles, "OMBRO D.: " + str(angle), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 7:
        cv.putText(frameAngles, "COTOVELO E.: " + str(angle), (0, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 8:
        cv.putText(frameAngles, "COTOVELO D.: " + str(angle), (0, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 11:
        cv.putText(frameAngles, "QUADRIL E.: " + str(angle), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 12:
        cv.putText(frameAngles, "QUADRIL D.: " + str(angle), (0, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 13:
        cv.putText(frameAngles, "JOELHO E.: " + str(angle), (0, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if index == 14:
        cv.putText(frameAngles, "JOELHO D.: " + str(angle), (0, 160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv.imshow("tela3", frameAngles)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:  # se nivel confianca for maior que confidence_threshold
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            cv.circle(frame2, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def parar():
    global running
    running = False


def start():
    global running
    running = True
    visu()


def reprise():
    global running
    running = False
    global i
    hori2 = np.concatenate((frameArrayOriginal[0], frameArray[0]), axis=1)
    im = Image.fromarray(hori2)
    im2 = ImageTk.PhotoImage(image=im)
    lblVideo.configure(image=im2)
    lblVideo.image = im2
    i = 0
    Button(win, text="▶", bg="gray", fg="white", command=seguido).place(x=30, y=500, width=40)
    Button(win, text="▶", bg="gray", fg="white", command=frente).place(x=120, y=500, width=40)
    Button(win, text="◀", bg="gray", fg="white", command=volta).place(x=70, y=500, width=40)


def frente():

    global i
    if i < len(frameArray) - 1:
        i = 1 + i
        hori2 = np.concatenate((frameArrayOriginal[i], frameArray[i]), axis=1)
        im = Image.fromarray(hori2)
        im2 = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=im2)
        lblVideo.image = im2


def volta():
    global i
    if i > 0:
        i = i - 1
        hori2 = np.concatenate((frameArrayOriginal[i], frameArray[i]), axis=1)
        im = Image.fromarray(hori2)
        im2 = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=im2)
        lblVideo.image = im2


def seguido():
    global i
    while True:
        if i > 0:
            for i in np.arange(i, len(frameArray)):
                im = Image.fromarray(frameArray[i])
                im2 = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=im2)
                lblVideo.image = im2

                #lblVideo.after(10, seguido)

        else:
            for i in np.arange(1, len(frameArray)):
                im = Image.fromarray(frameArray[i])
                im2 = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=im2)
                lblVideo.image = im2
                #lblVideo.after(10, seguido)

        break


# # ------------------MANIPULANDO VIDEO/FRAMES-----------------q
# if key == ord('j'):
#     # cv.destroyWindow('tela')
#     # cv.destroyWindow('tela2')
#     # cv.destroyWindow('tela3')
#     cv.destroyWindow('hori')
#     frame3 = np.zeros((255, 255, 3), np.uint8)
#     cv.imshow('Frame3', frameArray[0])
#     counter = 0
#     i = 0
#
#     while key != ord('q'):
#         # -------------START/PAUSE------------------
#         if key == ord('p'):
#             while True:
#                 if i > 0:
#                     for i in np.arange(i, len(frameArray)):
#                         cv.imshow('Frame3', frameArray[i])
#                         key = cv.waitKey(50)
#                         if key == ord('r'):
#                             i = 0
#                             break
#                         if key == ord('p'):
#                             break
#                 else:
#                     for i in np.arange(1, len(frameArray)):
#                         cv.imshow('Frame3', frameArray[i])
#                         key = cv.waitKey(50)
#                         if key == ord('r'):
#                             i = 0
#                             break
#                         if key == ord('p'):
#                             break
#                 break
#
#         # -----------AVANÇAR FRAME-------------
#         if key == ord('l'):
#             if i < len(frameArray) - 1:
#                 i = 1 + i
#                 cv.imshow('Frame3', frameArray[i])
#
#         # ----------VOLTAR FRAME--------------
#         if key == ord('k'):
#             if i > 0:
#                 i = i - 1
#                 cv.imshow('Frame3', frameArray[i])
#
#         # ----------RESTART------------------
#         if key == ord('r'):
#             i = 0
#             cv.imshow('Frame3', frameArray[i])
#
#         key = cv.waitKey(1)
#         if key == ord('q'):
#             break
#     cv.destroyAllWindows()

def visu():
    global cap
    global frame2
    global novoArray
    global pointsToPaint
    global frameAngles
    global frame
    global interestPoint
    global pointsToPaint
    global frameArray
    global frameArrayOriginal

    ret, frame = cap.read()
    key = cv.waitKey(10)
    if ret:
        if running:
            # --------------- LOOP VIDEO ---------------
            if not ret:
                novoArray.clear()
                pointsToPaint.clear()
                cv.destroyAllWindows()
                cap = cv.VideoCapture(video)
                hasFrame, frame = cap.read()

            frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
            frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta
            frameAngles = np.zeros((360, 160, 3), np.uint8)

            # ---------------------- TENSOR FLOW ----------------------------------------
            # reshape imagem para 192x192x3 (padrao documento do modelo treinado)
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype=tf.float32)
            # plt.imshow(tf.cast(np.squeeze(img), dtype=tf.int32))

            # inputs e ouputs
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # predicoes e pontos
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            print(keypoints_with_scores)

            # ------------------ PONTOS E RETAS -----------------------------
            draw_connections(frame, keypoints_with_scores, EDGES, 0.25)
            draw_keypoints(frame, keypoints_with_scores, 0.25)

            # ------------------------ANGLES---------------------------------
            y, x, c = frame.shape
            old_shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
            shaped = old_shaped.astype(int)
            shaped = np.delete(shaped, 2, 1)

            if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
                getAngle(shaped[6], shaped[8], shaped[12], 6)
            if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
                getAngle(shaped[5], shaped[7], shaped[11], 5)
            if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
                getAngle(shaped[8], shaped[6], shaped[10], 8)
            if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
                getAngle(shaped[7], shaped[5], shaped[9], 7)
            if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
                getAngle(shaped[12], shaped[6], shaped[14], 12)
            if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
                getAngle(shaped[11], shaped[5], shaped[13], 11)
            if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
                getAngle(shaped[14], shaped[12], shaped[16], 14)
            if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
                getAngle(shaped[13], shaped[11], shaped[15], 13)

            # ----------------------DESENHA MOTION TRACKING--------------------
            # mao direita
            if key == ord("1"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 10
            # mao equerda
            if key == ord("2"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 9
            # joelho direito
            if key == ord("3"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 14
            # joelho esquerdo
            if key == ord("4"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 13
            # quadril direito
            if key == ord("5"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 12
            # quadril esquerdo
            if key == ord("6"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 11
            # pe direito
            if key == ord("7"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 16
            # pe esquerdo
            if key == ord("8"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 15
            if key == ord("0"):
                pointsToPaint = pointsToPaint.clear()
                pointsToPaint = deque(maxlen=40)
                novoArray = novoArray.clear()
                novoArray = deque(maxlen=40)
                interestPoint = 0

            if interestPoint is not 0:
                pointsToPaint.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])

            if len(pointsToPaint) > 15:
                motionTrail(pointsToPaint)

            # ---------------------------- SAIDA TELA ---------------------------
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
            frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
            frameArray.append(frame2)
            frameArrayOriginal.append(frame)

            hori = np.concatenate((frame, frame2, frameAngles), axis=1)
            im = Image.fromarray(hori)
            im2 = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=im2)
            lblVideo.image = im2

            lblVideo.after(10, visu)

            # cv.imshow("hori", hori)
            # cv.imshow("tela2", frame2)
            # cv.imshow("tela", frame)

            # if key == ord('q'):  # QUIT
            # break

            # if key == ord('p'):  # PAUSE
            #     oldFrame = frame
            #     cv.setMouseCallback("hori", clickEvent)
            #     cv.waitKey(-1)
            #     # cv.destroyAllWindows()


win = Tk()
win.geometry("1125x600")
# Button(win, text="▶", bg="gray", fg="white").place(x=150, y=400, width=40)
# Button(win, text="◀", bg="gray", fg="white").place(x=100, y=400, width=40)

Button(win, text="▶", bg="gray", fg="white", command=start).place(x=30, y=400, width=40)
Button(win, text="⏸", bg="gray", fg="white", command=parar).place(x=100, y=400, width=40)
Button(win, text="⏏", bg="gray", fg="white", command=reprise).place(x=30, y=450, width=40)
# Button(win, text="Iniciar2", bg="gray", fg="white", command=start).place(x=200, y=400, width=40)
lblVideo = Label(win)
lblVideo.grid(column=0, row=2, columnspan=2)

win.mainloop()

# cap.release()
# cv.destroyAllWindows()
