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
import webcam

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
runningWeb = False


ombroDangulo = None
ombroEangulo = None
joelhoDangulo = None
joelhoEangulo = None
quadrilDangulo = None
quadrilEangulo = None
cotoveloDangulo = None
cotoveloEangulo = None

def motionTrail(paintedPoints):
    global globalFrameIndex, frameArrayOriginal2, frameArray2

    vetorTemp = frameArrayOriginal2

    # paintedPoints = paintedPoints[-40:]

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
            cv.line(vetorTemp[globalFrameIndex], novoArray[i - 1], novoArray[i], (255, 0, 255), 2)
            cv.line(frameArray2[globalFrameIndex], novoArray[i - 1], novoArray[i], (255, 0, 255), 2)

        else:
            cv.line(vetorTemp[globalFrameIndex], novoArray[i - 1], novoArray[i], (0, 255, 255), 2)
            cv.line(frameArray2[globalFrameIndex], novoArray[i - 1], novoArray[i], (0, 255, 255), 2)


def motionTrailRealTime(paintedPoints):
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

    return angle


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:  # se nivel confianca for maior que confidence_threshold
            cv.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
            cv.circle(frame2, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold, setAngle):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            if setAngle == p1 or setAngle == p2:
                cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            else:
                cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def draw_connectionsRealTime(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def addVideo():
    global cap, frameArray, frameArrayOriginal, running, frameArrayAngles, frameCount, runningWeb
    if cap is not None:
        cap.release()
        cap = None
        gui.lblVideo.image = ""

    running = False
    gui.guiWebcamDelete()
    runningWeb = False
    gui.lblVideo.image = ""
    gui.win.update()
    frameArrayOriginal = frameArrayOriginal.clear()
    frameArray = frameArray.clear()
    frameArrayAngles = frameArrayAngles.clear()
    frameArrayOriginal = []
    frameArray = []
    frameArrayAngles = []
    frameCount = 0

    video = filedialog.askopenfilename(title="Escolha um vídeo", filetypes=(("mp4 Files", ".mp4"),))
    cap = cv.VideoCapture(video)
    gui.guiManipulLoadedFramesDelete()
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    gui.guiCreateProgressBar(totalFrames)
    # ret, frame = cap.read()
    gui.guiProgressBar(0)
    gui.guiStartLoadedFrames()

    # frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # im = Image.fromarray(frame)
    # im2 = ImageTk.PhotoImage(image=im)
    # gui.lblVideo.configure(image=im2)
    # gui.lblVideo.image = im2


# ------------------------------------- FUNCOES BOTOES ----------------------------------------------
def parar():
    global running
    running = False


def start():
    global running
    running = True
    visu()


def reprise():
    global running, globalFrameIndex, varPlayAuxMd

    gui.lblVideo.image = ""
    gui.win.update()

    for indexx in np.arange(1, len(frameDefini)):
        frameArrayOriginal.append(frameDefini[indexx])

    varPlayAuxMd = False
    globalFrameIndex = 0
    running = False

    hori2 = np.concatenate((frameArrayOriginal[0], frameArray[0], frameArrayAngles[0]), axis=1)
    im = Image.fromarray(hori2)
    im2 = ImageTk.PhotoImage(image=im)
    gui.lblVideo.configure(image=im2)
    gui.lblVideo.image = im2
    gui.guiManipulCommands()


def frente():
    global globalFrameIndex
    if globalFrameIndex < len(frameArray) - 1:
        globalFrameIndex = 1 + globalFrameIndex
        hori2 = np.concatenate(
            (frameArrayOriginal[globalFrameIndex], frameArray[globalFrameIndex], frameArrayAngles[globalFrameIndex]),
            axis=1)
        im = Image.fromarray(hori2)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideo.configure(image=im2)
        gui.lblVideo.image = im2


def volta():
    global globalFrameIndex, globalFrameIndex
    if globalFrameIndex > 0:
        globalFrameIndex = globalFrameIndex - 1
        hori2 = np.concatenate(
            (frameArrayOriginal[globalFrameIndex], frameArray[globalFrameIndex], frameArrayAngles[globalFrameIndex]),
            axis=1)
        im = Image.fromarray(hori2)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideo.configure(image=im2)
        gui.lblVideo.image = im2


def seguido():
    global varPlayAux, varPlayAuxMd
    global speedVar, globalFrameIndex, pointsToPaintMdDeq, frameArrayOriginal2, frameArray2, frameDefini, frameArrayOriginal

    varPlayAuxMd = False
    pointsToPaintMdDeq = pointsToPaintMdDeq.clear()
    pointsToPaintMdDeq = deque(maxlen=40)

    if varPlayAux:
        varPlayAux = False
    else:
        varPlayAux = True

    if varPlayAux:
        # frameArrayOriginal = frameArrayOriginal.clear()
        # frameArrayOriginal = []

        # for indexx in np.arange(1, len(frameDefini)):
        # frameArrayOriginal.append(frameDefini[indexx])

        if globalFrameIndex > 0:
            for globalFrameIndex in np.arange(globalFrameIndex, len(frameArray)):
                hori = np.concatenate((frameArrayOriginal[globalFrameIndex], frameArray[globalFrameIndex],
                                       frameArrayAngles[globalFrameIndex]), axis=1)
                im = Image.fromarray(hori)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideo.configure(image=im2)
                gui.lblVideo.image = im2

                print("ESTOU AQUI NORMAL")
                time.sleep(speedVar)
                gui.win.update()
                if not varPlayAux:
                    break

        else:
            for globalFrameIndex in np.arange(1, len(frameArray)):
                hori = np.concatenate((frameArrayOriginal[globalFrameIndex], frameArray[globalFrameIndex],
                                       frameArrayAngles[globalFrameIndex]), axis=1)
                im = Image.fromarray(hori)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideo.configure(image=im2)
                gui.lblVideo.image = im2

                print("ESTOU AQUI NORMAL")
                time.sleep(speedVar)
                gui.win.update()
                if not varPlayAux:
                    break


def metaVideo():
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return totalFrames


def speed2x():
    global speedVar
    speedVar = 0.015


def speed1x():
    global speedVar
    speedVar = 0.03


# ------------------------------ ATIVAR MOTION TRAILL----------------------------------
def motionTrailMd():
    global mTactive, frameArrayOriginal2, frameArrayTemp, novoArray, frameArray2, pointsToPaintMdDeq, globalFrameIndex, globalFrameIndex, varPlayAuxMd
    mTactive = True
    # globalFrameIndex = 0
    # pointsToPaintMdDeq = pointsToPaintMdDeq.clear()
    # pointsToPaintMdDeq = deque(maxlen=40)

    if varPlayAuxMd:
        varPlayAuxMd = False
    else:
        varPlayAuxMd = True

    if varPlayAuxMd:
        # if globalFrameIndex > 0:
        for globalFrameIndex in np.arange(globalFrameIndex, len(frameArray)):
            pointsToPaintMdDeq.appendleft(pointsToPaintMd[globalFrameIndex])
            if len(pointsToPaintMdDeq) > 15:
                motionTrail(pointsToPaintMdDeq)
                print("ESTOU AQUI NORMAL MDDDDDDDDDDDDDDDD")
            hori3 = np.concatenate((frameArrayOriginal2[globalFrameIndex], frameArray2[globalFrameIndex],
                                    frameArrayAngles[globalFrameIndex]), axis=1)

            imagem = Image.fromarray(hori3)
            imagem2 = ImageTk.PhotoImage(image=imagem)
            gui.lblVideo.configure(image=imagem2)
            gui.lblVideo.image = imagem2
            time.sleep(speedVar)
            gui.win.update()

            if not varPlayAuxMd:
                break


# --------------------------------FUNCOES BOTOES REALTIME-----------------------------
def motionTrailMdRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 10


def motionTrailMeRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 9


def motionTrailJdRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 14


def motionTrailJeRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 13


def motionTrailQdRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 12


def motionTrailQeRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 11


def motionTrailPdRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 16


def motionTrailPeRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 15


def motionTrailNdaRealTime():
    global pointsToPaint, novoArray, interestPoint

    pointsToPaint = pointsToPaint.clear()
    pointsToPaint = deque(maxlen=40)
    novoArray = novoArray.clear()
    novoArray = deque(maxlen=40)
    interestPoint = 0
# ------------------------------------------------------------------------------------

def atualizarAngles():
    global ombroDangulo, ombroEangulo, joelhoDangulo, joelhoEangulo, quadrilDangulo, quadrilEangulo, cotoveloDangulo, cotoveloEangulo

    joelhoDangulo = gui.varAngleMd.get()


def visu():
    global cap, frame2, novoArray, pointsToPaint, frameAngles, frame, interestPoint, setAngle, \
        pointsToPaint, frameArray, frameArrayOriginal, frameArrayOriginal2, frameArray2, video, frameCount, totalFrames, angleJd

    global joelhoDangulo


    ret, frame = cap.read()

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

            # ------------------------ANGLES---------------------------------
            y, x, c = frame.shape
            old_shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
            shaped = old_shaped.astype(int)
            shaped = np.delete(shaped, 2, 1)

            #joelhoDangulo = None
            cotDangulo = 90
            angleOd = None
            angleOe = None
            angleCd = None
            angleCe = None
            angleQd = None
            angleQe = None
            angleJd = None
            angleJe = None
            setAngle = None
            if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
                angleOd = getAngle(shaped[6], shaped[8], shaped[12], 6)
            if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
                angleOe = getAngle(shaped[5], shaped[7], shaped[11], 5)
            if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
                angleCd = getAngle(shaped[8], shaped[6], shaped[10], 8)
            if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
                angleCe = getAngle(shaped[7], shaped[5], shaped[9], 7)
            if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
                angleQd = getAngle(shaped[12], shaped[6], shaped[14], 12)
            if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
                angleQe = getAngle(shaped[11], shaped[5], shaped[13], 11)
            if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
                angleJd = getAngle(shaped[14], shaped[12], shaped[16], 14)
            if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
                angleJe = getAngle(shaped[13], shaped[11], shaped[15], 13)

            # -------------------------COMPARADOR DE ANGULOS----------------------------------
            if joelhoDangulo is not None:
                if angleJd in range(joelhoDangulo - 10, joelhoDangulo + 10):
                    setAngle = 14

            if angleCd in range(cotDangulo - 10, cotDangulo + 10):
                setAngle = 8

            # ------------------ PONTOS E RETAS -----------------------------
            draw_connections(frame, keypoints_with_scores, EDGES, 0.25, setAngle)
            draw_keypoints(frame, keypoints_with_scores, 0.25)

            # ----------------------DESENHA MOTION TRACKING--------------------
            # mao direita
            # if key == ord("1"):
            # pointsToPaint = pointsToPaint.clear()
            # pointsToPaint = deque(maxlen=40)
            # novoArray = novoArray.clear()
            # novoArray = deque(maxlen=40)
            # interestPoint = 10
            # # mao equerda
            # if key == ord("2"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 9
            # # joelho direito
            # if key == ord("3"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 14
            # # joelho esquerdo
            # if key == ord("4"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 13
            # # quadril direito
            # if key == ord("5"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 12
            # # quadril esquerdo
            # if key == ord("6"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 11
            # # pe direito
            # if key == ord("7"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 16
            # # pe esquerdo
            # if key == ord("8"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 15
            # if key == ord("0"):
            #     pointsToPaint = pointsToPaint.clear()
            #     pointsToPaint = deque(maxlen=40)
            #     novoArray = novoArray.clear()
            #     novoArray = deque(maxlen=40)
            #     interestPoint = 0

            pointsToPaintMd.append([shaped[10][1], shaped[10][0], frameCount])
            # pointsToPaintMe.appendleft([shaped[9][1], shaped[9][0]])
            #
            # pointsToPaintJd.appendleft([shaped[3][1], shaped[3][0]])
            # pointsToPaintJe.appendleft([shaped[4][1], shaped[4][0]])
            #
            # pointsToPaintQd.appendleft([shaped[5][1], shaped[5][0]])
            # pointsToPaintQe.appendleft([shaped[6][1], shaped[6][0]])
            #
            # pointsToPaintPd.appendleft([shaped[7][1], shaped[7][0]])
            # pointsToPaintPe.appendleft([shaped[8][1], shaped[8][0]])

            # ---------------------------- SAIDA TELA ---------------------------
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
            frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
            frameArray.append(frame2)
            frameArrayOriginal.append(frame)
            frameArrayOriginal2.append(frame)
            frameDefini.append(frame)
            frameArray2.append(frame2)
            frameArrayAngles.append(frameAngles)

            # ----- ATUALIZA PROGRESS BAR -----------
            frameCount = frameCount + 1
            print(frameCount)
            gui.guiProgressBar(frameCount)

            gui.lblVideo.after(1, visu)


def addWebcam():
    global cap, frame2, novoArray, pointsToPaint, frameAngles, frame, interestPoint, \
        pointsToPaint, frameArray, frameArrayOriginal, video, frameCount, totalFrames, running, runningWeb

    global joelhoDangulo

    gui.guiWebcam()
    running = False
    runningWeb = True
    frameArrayOriginal = frameArrayOriginal.clear()
    frameArray = frameArray.clear()
    frameArrayOriginal = []
    frameArray = []
    frameCount = 0
    cap = cv.VideoCapture(0)

    while runningWeb:
        ret, frame = cap.read()
        frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
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


        # ------------------------ANGLES---------------------------------
        y, x, c = frame.shape
        old_shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
        shaped = old_shaped.astype(int)
        shaped = np.delete(shaped, 2, 1)

        cotDangulo = 90
        angleOd = None
        angleOe = None
        angleCd = None
        angleCe = None
        angleQd = None
        angleQe = None
        angleJd = None
        angleJe = None
        setAngle = None

        if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
            angleOd = getAngle(shaped[6], shaped[8], shaped[12], 6)
        if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
            angleOe = getAngle(shaped[5], shaped[7], shaped[11], 5)
        if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
            angleCd = getAngle(shaped[8], shaped[6], shaped[10], 8)
        if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
            angleCe = getAngle(shaped[7], shaped[5], shaped[9], 7)
        if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
            angleQd = getAngle(shaped[12], shaped[6], shaped[14], 12)
        if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
            angleQe = getAngle(shaped[11], shaped[5], shaped[13], 11)
        if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
            angleJd = getAngle(shaped[14], shaped[12], shaped[16], 14)
        if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
            angleJe = getAngle(shaped[13], shaped[11], shaped[15], 13)


        # -------------------------COMPARADOR DE ANGULOS----------------------------------
        if ombroDangulo is not None:
            if angleOd in range(int(ombroDangulo) - 10, int(ombroDangulo) + 10):
                setAngle = 6

        if ombroEangulo is not None:
            if angleOe in range(int(ombroEangulo) - 10, int(ombroEangulo) + 10):
                setAngle = 5

        if cotoveloDangulo is not None:
            if angleCd in range(int(cotoveloDangulo) - 10, int(cotoveloDangulo) + 10):
                setAngle = 8

        if cotoveloEangulo is not None:
            if angleCe in range(int(cotoveloEangulo) - 10, int(cotoveloEangulo) + 10):
                setAngle = 7

        if joelhoDangulo is not None:
            if angleJd in range(int(joelhoDangulo) - 10, int(joelhoDangulo) + 10):
                setAngle = 14

        if joelhoEangulo is not None:
            if angleJe in range(int(joelhoEangulo) - 10, int(joelhoEangulo) + 10):
                setAngle = 13

        if quadrilDangulo is not None:
            if angleQd in range(int(quadrilDangulo) - 10, int(quadrilDangulo) + 10):
                setAngle = 12

        if quadrilEangulo is not None:
            if angleQe in range(int(quadrilEangulo) - 10, int(quadrilEangulo) + 10):
                setAngle = 11


        # ------------------ DESENHA PONTOS E RETAS -----------------------------
        draw_connections(frame, keypoints_with_scores, EDGES, 0.25, setAngle)
        draw_keypoints(frame, keypoints_with_scores, 0.25)


        # # ----------------------DESENHA MOTION TRACKING--------------------
        if interestPoint is not 0:
            pointsToPaint.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])

            if len(pointsToPaint) > 15:
                motionTrailRealTime(pointsToPaint)

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
        gui.win.update()






























































#def clickEvent(event, x, y, flag, param):
#     yFrame, xFrame, cFrame = frame.shape
#     if event == cv.EVENT_LBUTTONDOWN:
#         cv.line(frame, (x, y), (xFrame, y), (255, 0, 255), 2)
#         cv.line(frame, (x, y), (0, y), (255, 0, 255), 2)
#
#         cv.line(frame, (x, y), (x, yFrame), (255, 0, 255), 2)
#         cv.line(frame, (x, y), (x, 0), (255, 0, 255), 2)
#         cv.imshow("tela", frame)





# def visuRealTime():
#     global cap, frame2, novoArray, pointsToPaint, frameAngles, frame, interestPoint, \
#         pointsToPaint, frameArray, frameArrayOriginal, video, frameCount, totalFrames
#
#     ret, frame = cap.read()
#
#     if ret:
#         key = cv.waitKey(10)
#
#         # --------------- LOOP VIDEO ---------------
#         if not ret:
#             novoArray.clear()
#             pointsToPaint.clear()
#             cv.destroyAllWindows()
#             cap = cv.VideoCapture(video)
#             hasFrame, frame = cap.read()
#
#         frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
#         frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta
#         frameAngles = np.zeros((360, 160, 3), np.uint8)
#
#         # ---------------------- TENSOR FLOW ----------------------------------------
#         # reshape imagem para 192x192x3 (padrao documento do modelo treinado)
#         img = frame.copy()
#         img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
#         input_image = tf.cast(img, dtype=tf.float32)
#         # plt.imshow(tf.cast(np.squeeze(img), dtype=tf.int32))
#
#         # inputs e ouputs
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
#
#         # predicoes e pontos
#         interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
#         interpreter.invoke()
#         keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
#         print(keypoints_with_scores)
#
#         # ------------------ PONTOS E RETAS -----------------------------
#         draw_connectionsRealTime(frame, keypoints_with_scores, EDGES, 0.25)
#         draw_keypoints(frame, keypoints_with_scores, 0.25)
#
#         # ------------------------ANGLES---------------------------------
#         y, x, c = frame.shape
#         old_shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, c]))
#         shaped = old_shaped.astype(int)
#         shaped = np.delete(shaped, 2, 1)
#
#         if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
#             getAngle(shaped[6], shaped[8], shaped[12], 6)
#         if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
#             getAngle(shaped[5], shaped[7], shaped[11], 5)
#         if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
#             getAngle(shaped[8], shaped[6], shaped[10], 8)
#         if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
#             getAngle(shaped[7], shaped[5], shaped[9], 7)
#         if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
#             getAngle(shaped[12], shaped[6], shaped[14], 12)
#         if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
#             getAngle(shaped[11], shaped[5], shaped[13], 11)
#         if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
#             getAngle(shaped[14], shaped[12], shaped[16], 14)
#         if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
#             getAngle(shaped[13], shaped[11], shaped[15], 13)
#
#         # ----------------------DESENHA MOTION TRACKING--------------------
#         # mao direita
#         if key == ord("1"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 10
#         # # mao equerda
#         if key == ord("2"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 9
#         # # joelho direito
#         if key == ord("3"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 14
#         # # joelho esquerdo
#         if key == ord("4"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 13
#         # # quadril direito
#         if key == ord("5"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 12
#         # # quadril esquerdo
#         if key == ord("6"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 11
#         # # pe direito
#         if key == ord("7"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 16
#         # # pe esquerdo
#         if key == ord("8"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 15
#         if key == ord("0"):
#             pointsToPaint = pointsToPaint.clear()
#             pointsToPaint = deque(maxlen=40)
#             novoArray = novoArray.clear()
#             novoArray = deque(maxlen=40)
#             interestPoint = 0
#
#         if interestPoint is not 0:
#             pointsToPaint.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])
#
#         if len(pointsToPaint) > 15:
#             motionTrailRealTime(pointsToPaint)
#
#         # ---------------------------- SAIDA TELA ---------------------------
#         frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
#         frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
#         frame2 = cv.resize(frame2, [480, 360], interpolation=cv.INTER_BITS)
#         frameArray.append(frame2)
#         frameArrayOriginal.append(frame)
#         frameArrayAngles.append(frameAngles)
#
#         hori = np.concatenate((frame, frame2, frameAngles), axis=1)
#         im = Image.fromarray(hori)
#         im2 = ImageTk.PhotoImage(image=im)
#         gui.lblVideo.configure(image=im2)
#         gui.lblVideo.image = im2

# ----- ATUALIZA PROGRESS BAR -----------

# frameCount = frameCount + 1
# print(frameCount)
# gui.guiProgressBar(frameCount)
# gui.win.update()

# gui.lblVideo.after(1, visuRealTime())

# cv.imshow("hori", hori)
# cv.imshow("tela2", frame2)
# cv.imshow("tela", frame)

# if key == ord('p'):  # PAUSE
#     oldFrame = frame
#     cv.setMouseCallback("hori", clickEvent)
#     cv.waitKey(-1)
#     # cv.destroyAllWindows()

# ----------------------------------------GUI-------------------------------------------------------
# win = Tk()
# win.geometry("1125x600")
#
# Button(win, text="▶", bg="gray", fg="white", command=start).place(x=30, y=400, width=40)
# Button(win, text="⏸", bg="gray", fg="white", command=parar).place(x=100, y=400, width=40)
# Button(win, text="⏏", bg="gray", fg="white", command=reprise).place(x=30, y=450, width=40)
# # Button(win, text="Iniciar2", bg="gray", fg="white", command=start).place(x=200, y=400, width=40)
# my_menu = Menu(win)
# win.config(menu=my_menu)
#
# add_video_menu = Menu(my_menu)
# my_menu.add_cascade(label="Abrir", menu=add_video_menu)
# add_video_menu.add_command(label="Arquivo", command=addVideo)
# add_video_menu.add_command(label="Webcam", command=addWebcam)
#
# lblVideo = Label(win)
# lblVideo.grid(column=0, row=2, columnspan=2)
#
# win.mainloop()

# cap.release()
# cv.destroyAllWindows()
