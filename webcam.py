import time
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2 as cv
from collections import deque
from scipy.spatial import distance as dist
from tkinter import *
from PIL import Image
from PIL import ImageTk
import gui
import main

interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

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

pointsToPaintMd = []
pointsToPaintMdDeq = deque(maxlen=40)
mTactive = False

pointsToPaintMe = deque(maxlen=40)
pointsToPaintOd = deque(maxlen=40)
pointsToPaintOe = deque(maxlen=40)
pointsToPaintPd = deque(maxlen=40)
pointsToPaintPe = deque(maxlen=40)

pointsToPaintJd = deque(maxlen=40)
pointsToPaintJe = deque(maxlen=40)
pointsToPaintQd = deque(maxlen=40)
pointsToPaintQe = deque(maxlen=40)
pointsToPaintPd = deque(maxlen=40)
pointsToPaintPe = deque(maxlen=40)

# globalFrameIndex = 0
interestPoint = 0
frameArray = []
frameArrayOriginal = []
frameArrayAngles = []
frameDefini = []

frameArrayOriginalTemp = []

frameArray2 = []
frameArrayOriginal2 = []
novoArray = deque(maxlen=40)
globalFrameIndex = 0
video = 0
cap = None
frameCount = 0
varPlayAux = False
varPlayAuxMd = False
speedVar = 0.03
setAngle = 0

def startWebcam():
    global cap

    visuRealTime()

def visuRealTime():
    global cap, frame2, novoArray, pointsToPaint, frameAngles, frame, interestPoint, \
        pointsToPaint, frameArray, frameArrayOriginal, video, frameCount, totalFrames

    ret, frame = cap.read()

    if ret:
        key = cv.waitKey(10)

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
        main.draw_connectionsRealTime(frame, keypoints_with_scores, EDGES, 0.25)
        main.draw_keypoints(frame, keypoints_with_scores, 0.25)

        # ------------------------ANGLES---------------------------------
        y, x, c = frame.shape
        old_shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
        shaped = old_shaped.astype(int)
        shaped = np.delete(shaped, 2, 1)

        if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
            main.getAngle(shaped[6], shaped[8], shaped[12], 6)
        if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
            main.getAngle(shaped[5], shaped[7], shaped[11], 5)
        if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
            main.getAngle(shaped[8], shaped[6], shaped[10], 8)
        if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
            main.getAngle(shaped[7], shaped[5], shaped[9], 7)
        if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
            main.getAngle(shaped[12], shaped[6], shaped[14], 12)
        if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
            main.getAngle(shaped[11], shaped[5], shaped[13], 11)
        if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
            main.getAngle(shaped[14], shaped[12], shaped[16], 14)
        if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
            main.getAngle(shaped[13], shaped[11], shaped[15], 13)

        # ----------------------DESENHA MOTION TRACKING--------------------
        # mao direita
        if key == ord("1"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 10
        # # mao equerda
        if key == ord("2"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 9
        # # joelho direito
        if key == ord("3"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 14
        # # joelho esquerdo
        if key == ord("4"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 13
        # # quadril direito
        if key == ord("5"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 12
        # # quadril esquerdo
        if key == ord("6"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 11
        # # pe direito
        if key == ord("7"):
            pointsToPaint = pointsToPaint.clear()
            pointsToPaint = deque(maxlen=40)
            novoArray = novoArray.clear()
            novoArray = deque(maxlen=40)
            interestPoint = 16
        # # pe esquerdo
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
            main.motionTrailRealTime(pointsToPaint)

        # ---------------------------- SAIDA TELA ---------------------------
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
        frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
        frameArray.append(frame2)
        frameArrayOriginal.append(frame)
        frameArrayAngles.append(frameAngles)

        hori = np.concatenate((frame, frame2, frameAngles), axis=1)
        im = Image.fromarray(hori)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideo.configure(image=im2)
        gui.lblVideo.image = im2

        # ----- ATUALIZA PROGRESS BAR -----------

        # frameCount = frameCount + 1
        # print(frameCount)
        # gui.guiProgressBar(frameCount)
        # gui.win.update()

        gui.lblVideo.after(1, visuRealTime())

        # cv.imshow("hori", hori)
        # cv.imshow("tela2", frame2)
        # cv.imshow("tela", frame)

        # if key == ord('p'):  # PAUSE
        #     oldFrame = frame
        #     cv.setMouseCallback("hori", clickEvent)
        #     cv.waitKey(-1)
        #     # cv.destroyAllWindows()
