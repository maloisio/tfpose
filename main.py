import time
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
import cv2 as cv
from collections import deque
from scipy.spatial import distance as dist
from tkinter import *
from PIL import Image
from PIL import ImageTk
import gui
import psutil

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

# globalFrameIndex = 0
interestPoint = 0
frameArray = []
frameArrayOriginal = []
frameArrayAngles = []
frameDefini = []

frameArrayOriginalTemp = []

frameArray2 = []
frameArrayOriginal2 = []

frameFoto = []
frameFotoOriginal = []
fotoToSave = []
frameFotoArray = []

frameVideoOriginal = []
frameVideoArray = []
globalFrameIndexVideo = 0

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

corAttPointsBlue = 0
corAttPointsGreen = 255
corAttPointsRed = 0

corAttLinesBlue = 0
corAttLinesGreen = 0
corAttLinesRed = 255

corAttMtBlue = 255
corAttMtGreen = 0
corAttMtRed = 255

corAttRed = 255
corAttGreen = 0
corAttBlue = 0
attThr = 0.25

ombroDangulo = 0
ombroEangulo = 0
joelhoDangulo = 0
joelhoEangulo = 0
quadrilDangulo = 0
quadrilEangulo = 0
cotoveloDangulo = 0
cotoveloEangulo = 0

anguloOmbroD = None
anguloOmbroE = None
anguloJoelhoD = None
anguloJoelhoE = None
anguloQuadrilD = None
anguloQuadrilE = None
anguloCotoveloD = None
anguloCotoveloE = None

setInterestAngleToColor = [None] * 16

startGravacao = False
varPlayAuxFramesGravacao = False
speedVarVideoGravacao = 0.03

anglesToCompare = []
capVid = None
runningVid = False

def motionTrailRealTime(paintedPoints):
    global corAttMtBlue, corAttMtGreen, corAttMtRed

    mediaX1 = np.sum(
        paintedPoints[0][0] + paintedPoints[1][0] + paintedPoints[2][0] + paintedPoints[3][0] + paintedPoints[4][0] +
        paintedPoints[5][0] + paintedPoints[6][0])

    mediaY1 = np.sum(
        paintedPoints[0][1] + paintedPoints[1][1] + paintedPoints[2][1] + paintedPoints[3][1] + paintedPoints[4][1] +
        paintedPoints[5][1] + paintedPoints[6][0])

    mediaX2 = np.sum(
        paintedPoints[7][0] + paintedPoints[8][0] + paintedPoints[9][0] + paintedPoints[10][0] + paintedPoints[11][0] +
        paintedPoints[12][0] + paintedPoints[13][0])
    mediaY2 = np.sum(
        paintedPoints[7][1] + paintedPoints[8][1] + paintedPoints[9][1] + paintedPoints[10][1] + paintedPoints[11][1] +
        paintedPoints[12][1] + paintedPoints[13][0])

    novoArray.appendleft([int(mediaX1 / 7), int(mediaY1 / 7)])
    novoArray.appendleft([int(mediaX2 / 7), int(mediaY2 / 7)])

    for i in np.arange(1, len(novoArray)):
        if novoArray[i - 1] is None or novoArray[i] is None:
            continue

        if dist.euclidean(novoArray[i], novoArray[i - 1]) < 60:
            cv.line(frame, novoArray[i - 1], novoArray[i], (corAttMtRed, corAttMtGreen, corAttMtBlue), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (corAttMtRed, corAttMtGreen, corAttMtBlue), 2)
        else:
            cv.line(frame, novoArray[i - 1], novoArray[i], (corAttMtRed, corAttMtGreen, corAttMtBlue), 2)
            cv.line(frame2, novoArray[i - 1], novoArray[i], (corAttMtRed, corAttMtGreen, corAttMtBlue), 2)


def getAngleRealTime(pt1, pt2, pt3, interestPoint):
    ypt1, xpt1 = pt1

    radians = np.arctan2(pt3[1] - pt1[1], pt3[0] - pt1[0]) - np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    angle = round(np.abs(radians * 180.0 / np.pi))  # conversao radianos para graus

    if angle > 180:
        angle = 360 - angle
    cv.putText(frame2, str(angle), (int(xpt1), int(ypt1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    if interestPoint == 5:
        gui.varAngleOeValue.set(angle)

    if interestPoint == 6:
        gui.varAngleOdValue.set(angle)

    if interestPoint == 7:
        gui.varAngleCeValue.set(angle)

    if interestPoint == 8:
        gui.varAngleCdValue.set(angle)

    if interestPoint == 11:
        gui.varAngleQeValue.set(angle)

    if interestPoint == 12:
        gui.varAngleQdValue.set(angle)

    if interestPoint == 13:
        gui.varAngleJeValue.set(angle)

    if interestPoint == 14:
        gui.varAngleJdValue.set(angle)

    return angle


def draw_keypoints(frame, keypoints, confidence_threshold):
    global corAttPointsBlue, corAttPointsGreen, corAttPointsRed

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:  # se nivel confianca for maior que confidence_threshold
            cv.circle(frame, (int(kx), int(ky)), 4, (corAttPointsRed, corAttPointsGreen, corAttPointsBlue), -1)
            cv.circle(frame2, (int(kx), int(ky)), 4, (corAttPointsRed, corAttPointsGreen, corAttPointsBlue), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    global corAttLinesBlue, corAttLinesGreen, corAttLinesRed

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (corAttLinesRed, corAttLinesGreen, corAttLinesBlue),
                    1)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (corAttLinesRed, corAttLinesGreen, corAttLinesBlue),
                    2)


def draw_connectionsRealTime(frame, keypoints, edges, confidence_threshold):
    global corAttRed, corAttGreen, corAttBlue, setInterestAngleToColor
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  # multiplica pelo tamanho widht e height da imagem

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (corAttBlue, corAttGreen, corAttRed), 1)
            cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (corAttBlue, corAttGreen, corAttRed), 2)

        for i in np.arange(1, len(setInterestAngleToColor)):
            if setInterestAngleToColor[i] is not None:
                if setInterestAngleToColor[i] == p1 or setInterestAngleToColor[i] == p2:
                    cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                    cv.line(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)



# ------------------------------------- FUNCOES BOTOES ----------------------------------------------
def parar():
    global runningVid
    runningVid = False


def start():
    global runningVid
    runningVid = True


def reprise():
    global running, globalFrameIndex, varPlayAuxMd, varPlayAux

    gui.lblVideo.image = ""
    gui.win.update()

    for indexx in np.arange(1, len(frameDefini)):
        frameArrayOriginal.append(frameDefini[indexx])

    varPlayAuxMd = False
    globalFrameIndex = 0
    running = False
    varPlayAux = False

    outputFrameArrayOriginal = frameArrayOriginal[0]
    outputFrameArray = frameArray[0]

    im = Image.fromarray(outputFrameArrayOriginal)
    im2 = ImageTk.PhotoImage(image=im)
    gui.lblVideo.configure(image=im2)
    gui.lblVideo.image = im2

    im = Image.fromarray(outputFrameArray)
    im2 = ImageTk.PhotoImage(image=im)
    gui.lblVideo2.configure(image=im2)
    gui.lblVideo2.image = im2

    # hori2 = np.concatenate((frameArrayOriginal[0], frameArray[0], frameArrayAngles[0]), axis=1)
    # im = Image.fromarray(hori2)
    # im2 = ImageTk.PhotoImage(image=im)
    # gui.lblVideo.configure(image=im2)
    # gui.lblVideo.image = im2
    gui.guiManipulCommands()





def metaVideo():
    totalFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    return totalFrames


def speedMenos2x():
    global speedVar
    speedVar = 0.06


def speed1x():
    global speedVar
    speedVar = 0.03


def speed2x():
    global speedVar
    speedVar = 0.015



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


def resetParams():
    global corAttBlue, corAttRed, corAttGreen, attThr
    global corAttMtBlue, corAttMtGreen, corAttMtRed
    global corAttPointsBlue, corAttPointsGreen, corAttPointsRed
    global corAttLinesBlue, corAttLinesGreen, corAttLinesRed

    corAttPointsBlue = 0
    corAttPointsGreen = 255
    corAttPointsRed = 0

    corAttLinesBlue = 0
    corAttLinesGreen = 0
    corAttLinesRed = 255

    corAttMtBlue = 255
    corAttMtGreen = 0
    corAttMtRed = 255

    corAttRed = 0
    corAttGreen = 0
    corAttBlue = 0
    attThr = 0.25
    motionTrailNdaRealTime()

    gui.scale_hori_Red.set(corAttRed)
    gui.scale_hori_Green.set(corAttGreen)
    gui.scale_hori_Blue.set(corAttBlue)
    gui.scale_threshold.set(attThr)


def saveFoto():
    global fotoToSave
    try:
        # frameFotoOriginal = cv.cvtColor(frameFotoOriginal, cv.COLOR_BGR2RGB)
        fotoToSave = cv.cvtColor(fotoToSave, cv.COLOR_RGB2BGR)
        export_file_path = filedialog.asksaveasfilename(filetypes=(('image', '*.jpg'), ('All', '*.*')),
                                                        defaultextension='*.jpg')
        cv.imwrite(export_file_path, fotoToSave)

    except NameError:
        messagebox.showwarning("Warning", "Import image first")


def attColorPoints():
    global corAttRed, corAttBlue, corAttGreen, corAttPointsRed, corAttPointsBlue, corAttPointsGreen

    corAttPointsBlue = corAttBlue
    corAttPointsGreen = corAttGreen
    corAttPointsRed = corAttRed


def attColorLines():
    global corAttRed, corAttBlue, corAttGreen, corAttLinesRed, corAttLinesBlue, corAttLinesGreen

    corAttLinesBlue = corAttBlue
    corAttLinesGreen = corAttGreen
    corAttLinesRed = corAttRed


def attColorMt():
    global corAttRed, corAttBlue, corAttGreen, corAttMtRed, corAttMtBlue, corAttMtGreen

    corAttMtBlue = corAttBlue
    corAttMtGreen = corAttGreen
    corAttMtRed = corAttRed


def selectOriginalPhoto():
    global frameFotoOriginal, fotoToSave

    fotoToSave = frameFotoOriginal


def selectLinesAndPointsPhoto():
    global frameFotoArray, fotoToSave

    fotoToSave = frameFotoArray


def gravarVideo():
    global startGravacao, globalFrameIndexVideo, frameVideoArray, frameVideoOriginal
    gui.btn_visualizar_video.place_forget()
    globalFrameIndexVideo = 0
    frameVideoArray = []
    frameVideoOriginal = []
    gui.label_frameVideo_gravando.place(x=65, y=10, width=60)
    startGravacao = True


def pararGravacaoVideo():
    global startGravacao
    gui.label_frameVideo_gravando.place_forget()
    gui.btn_visualizar_video.place(x=185, y=10, width=60, height=20)
    startGravacao = False


def salvarVideo():
    global frameVideoOriginal

    out = "C:/users/Mauro/Downloads/video"
    #wvideo, hvideo, c = frameVideoOriginal.shape


    #export_file_path = filedialog.asksaveasfilename(filetypes=(('video', '*.avi'), ('All', '*.*')),defaultextension='*.avi')


    videoSave = cv.VideoWriter('video.avi', cv.VideoWriter_fourcc(*'DIVX'), 30.0, (400,600))

    for i in range(len(frameVideoOriginal)):
        videoSave.write(frameVideoOriginal[i])

    videoSave.release()


# ------------------------------------- BOTOES GRAVACAO ---------------------------
def newSecondWindow():
    gui.guiVisualizarVideo()


def visualizarVideoGravacao():
    global frameVideoOriginal, frameVideoArray, globalFrameIndexVideo, varPlayAuxFramesGravacao

    if varPlayAuxFramesGravacao:
        varPlayAuxFramesGravacao = False
    else:
        varPlayAuxFramesGravacao = True

    if varPlayAuxFramesGravacao:
        if globalFrameIndexVideo > 0:
            for globalFrameIndexVideo in np.arange(globalFrameIndexVideo, len(frameVideoArray)):
                outputFrameVideoArrayOriginal = frameVideoOriginal[globalFrameIndexVideo]
                outputFrameVideoArray = frameVideoArray[globalFrameIndexVideo]

                im = Image.fromarray(outputFrameVideoArrayOriginal)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideoGravacaoOriginal.configure(image=im2)
                gui.lblVideoGravacaoOriginal.image = im2

                im = Image.fromarray(outputFrameVideoArray)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideoGravacaoArray.configure(image=im2)
                gui.lblVideoGravacaoArray.image = im2

                time.sleep(speedVarVideoGravacao)
                gui.secondWindow.update()
                if not varPlayAuxFramesGravacao:
                    break

        else:
            for globalFrameIndexVideo in np.arange(1, len(frameVideoArray)):

                outputFrameVideoArrayOriginal = frameVideoOriginal[globalFrameIndexVideo]
                outputFrameVideoArray = frameVideoArray[globalFrameIndexVideo]

                im = Image.fromarray(outputFrameVideoArrayOriginal)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideoGravacaoOriginal.configure(image=im2)
                gui.lblVideoGravacaoOriginal.image = im2

                im = Image.fromarray(outputFrameVideoArray)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblVideoGravacaoArray.configure(image=im2)
                gui.lblVideoGravacaoArray.image = im2

                time.sleep(speedVarVideoGravacao)
                gui.secondWindow.update()
                if not varPlayAuxFramesGravacao:
                    break


def startFramesVideoGravacao():
    global varPlayAuxFramesGravacao
    varPlayAuxFramesGravacao = True


def speedMenos2xVideoGravacao():
    global speedVarVideoGravacao
    speedVarVideoGravacao = 0.06


def speed1xVideoGravacao():
    global speedVarVideoGravacao
    speedVarVideoGravacao = 0.03


def speed2xVideoGravacao():
    global speedVarVideoGravacao
    speedVarVideoGravacao = 0.015


def resetFramesVideoGravacao():
    global varPlayAuxFramesGravacao, globalFrameIndexVideo, frameVideoOriginal, frameVideoArray

    varPlayAuxFramesGravacao = False
    globalFrameIndexVideo = 0

    outputFrameVideoArrayOriginal = frameVideoOriginal[globalFrameIndexVideo]
    outputFrameVideoArray = frameVideoArray[globalFrameIndexVideo]

    im = Image.fromarray(outputFrameVideoArrayOriginal)
    im2 = ImageTk.PhotoImage(image=im)
    gui.lblVideoGravacaoOriginal.configure(image=im2)
    gui.lblVideoGravacaoOriginal.image = im2

    im = Image.fromarray(outputFrameVideoArray)
    im2 = ImageTk.PhotoImage(image=im)
    gui.lblVideoGravacaoArray.configure(image=im2)
    gui.lblVideoGravacaoArray.image = im2

    time.sleep(speedVarVideoGravacao)
    gui.secondWindow.update()


def frenteVideoGravacao():
    global globalFrameIndexVideo

    if globalFrameIndexVideo < len(frameVideoArray) - 1:
        globalFrameIndexVideo = 1 + globalFrameIndexVideo

        outputFrameVideoArrayOriginal = frameVideoOriginal[globalFrameIndexVideo]
        outputFrameVideoArray = frameVideoArray[globalFrameIndexVideo]

        im = Image.fromarray(outputFrameVideoArrayOriginal)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideoGravacaoOriginal.configure(image=im2)
        gui.lblVideoGravacaoOriginal.image = im2

        im = Image.fromarray(outputFrameVideoArray)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideoGravacaoArray.configure(image=im2)
        gui.lblVideoGravacaoArray.image = im2
        gui.secondWindow.update()


def voltaVideoGravacao():
    global globalFrameIndexVideo
    if globalFrameIndexVideo > 0:
        globalFrameIndexVideo = globalFrameIndexVideo - 1

        outputFrameVideoArrayOriginal = frameVideoOriginal[globalFrameIndexVideo]
        outputFrameVideoArray = frameVideoArray[globalFrameIndexVideo]

        im = Image.fromarray(outputFrameVideoArrayOriginal)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideoGravacaoOriginal.configure(image=im2)
        gui.lblVideoGravacaoOriginal.image = im2

        im = Image.fromarray(outputFrameVideoArray)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideoGravacaoArray.configure(image=im2)
        gui.lblVideoGravacaoArray.image = im2
        gui.secondWindow.update()


# -------------------------------- ENTRYS -------------------------------------------
def atualizarAngles():
    global ombroDangulo, ombroEangulo, joelhoDangulo, joelhoEangulo, quadrilDangulo, quadrilEangulo, cotoveloDangulo, cotoveloEangulo
    global anglesToCompare

    anglesToCompare = []

    ombroDangulo = gui.varAngleOd.get()
    if ombroDangulo == 0 or ombroDangulo == '':
        ombroDangulo = None

    ombroEangulo = gui.varAngleOe.get()
    if ombroEangulo == 0 or ombroEangulo == '':
        ombroEangulo = None

    cotoveloDangulo = gui.varAngleCd.get()
    if cotoveloDangulo == 0 or cotoveloDangulo == '':
        cotoveloDangulo = None

    cotoveloEangulo = gui.varAngleCe.get()
    if cotoveloEangulo == 0 or cotoveloEangulo == '':
        cotoveloEangulo = None

    quadrilDangulo = gui.varAngleQd.get()
    if quadrilDangulo == 0 or quadrilDangulo == '':
        quadrilDangulo = None

    quadrilEangulo = gui.varAngleQe.get()
    if quadrilEangulo == 0 or quadrilEangulo == '':
        quadrilEangulo = None

    joelhoDangulo = gui.varAngleJd.get()
    if joelhoDangulo == 0 or joelhoDangulo == '':
        joelhoDangulo = None

    joelhoEangulo = gui.varAngleJe.get()
    if joelhoEangulo == 0 or joelhoEangulo == '':
        joelhoEangulo = None


# --------------------------------- SCALES ---------------------------------
def update_corRed(v):
    global corAttRed
    corAttRed = int(gui.scale_hori_Red.get())


def update_corGreen(v):
    global corAttGreen
    corAttGreen = int(gui.scale_hori_Green.get())


def update_corBlue(v):
    global corAttBlue
    corAttBlue = int(gui.scale_hori_Blue.get())


def update_thr(v):
    global attThr
    attThr = float(gui.scale_threshold.get())


# ------------------------------ VIDEO -------------------------------
def addVid():
    global frame2, novoArray, pointsToPaint, frame, interestPoint, \
        pointsToPaint, frameArray, frameArrayOriginal, video, frameCount, runningVid, runningWeb


    gui.guiWebcamDelete()
    runningWeb = False
    gui.lblVideo.image = ""
    gui.win.update()
    frameArrayOriginal = frameArrayOriginal.clear()
    frameArray = frameArray.clear()
    frameArrayOriginal = []
    frameArray = []
    frameCount = 0

    video = filedialog.askopenfilename(title="Escolha um vídeo", filetypes=(("mp4 Files", ".mp4"),))
    capVid = cv.VideoCapture(video)

    global joelhoDangulo
    global setInterestAngleToColor
    global frameFoto, frameFotoOriginal, frameFotoArray, frameVideoOriginal, frameVideoArray, anglesToCompare

    gui.guiVisuVideo()
    gui.guiWebcam()

    #runningWeb = True
    frameArrayOriginal = frameArrayOriginal.clear()
    frameArray = frameArray.clear()
    frameArrayOriginal = []
    frameArray = []
    frameCount = 0


    resetParams()

    angleOd = None
    angleOe = None
    angleCd = None
    angleCe = None
    angleQd = None
    angleQe = None
    angleJd = None
    angleJe = None


    while True:
        if runningVid:
            ret, frame = capVid.read()

            if not ret:
                capVid = cv.VideoCapture(video)
                ret, frame = capVid.read()

            frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
            frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta

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

            if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
                angleOd = getAngleRealTime(shaped[6], shaped[8], shaped[12], 6)
            else:
                gui.varAngleOdValue.set(" ")

            if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
                angleOe = getAngleRealTime(shaped[5], shaped[7], shaped[11], 5)
            else:
                gui.varAngleOeValue.set(" ")

            if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
                angleCd = getAngleRealTime(shaped[8], shaped[6], shaped[10], 8)
            else:
                gui.varAngleCdValue.set(" ")

            if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
                angleCe = getAngleRealTime(shaped[7], shaped[5], shaped[9], 7)
            else:
                gui.varAngleCeValue.set(" ")

            if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
                angleQd = getAngleRealTime(shaped[12], shaped[6], shaped[14], 12)
            else:
                gui.varAngleQdValue.set(" ")

            if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
                angleQe = getAngleRealTime(shaped[11], shaped[5], shaped[13], 11)
            else:
                gui.varAngleQeValue.set(" ")

            if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
                angleJd = getAngleRealTime(shaped[14], shaped[12], shaped[16], 14)
            else:
                gui.varAngleJdValue.set(" ")

            if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
                angleJe = getAngleRealTime(shaped[13], shaped[11], shaped[15], 13)
            else:
                gui.varAngleJeValue.set(" ")

            # ------------------ DESENHA PONTOS E RETAS -----------------------------
            draw_connections(frame, keypoints_with_scores, EDGES, attThr)
            draw_keypoints(frame, keypoints_with_scores, attThr)

            # # ----------------------DESENHA MOTION TRACKING--------------------
            if interestPoint is not 0:
                pointsToPaint.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])

                if len(pointsToPaint) > 15:
                    motionTrailRealTime(pointsToPaint)

            # -------------------------COMPARADOR DE ANGULOS----------------------------------
            if (ombroEangulo is None or angleOe in range(int(ombroEangulo) - 2, int(ombroEangulo) + 2)) and \
                    (ombroDangulo is None or angleOd in range(int(ombroDangulo) - 2, int(ombroDangulo) + 2)) and \
                    (cotoveloDangulo is None or angleCd in range(int(cotoveloDangulo) - 2, int(cotoveloDangulo) + 2)) and \
                    (cotoveloEangulo is None or angleCe in range(int(cotoveloEangulo) - 2, int(cotoveloEangulo) + 2)) and \
                    (joelhoDangulo is None or angleJd in range(int(joelhoDangulo) - 2, int(joelhoDangulo) + 2)) and \
                    (joelhoEangulo is None or angleJe in range(int(joelhoEangulo) - 2, int(joelhoEangulo) + 2)) and \
                    (quadrilDangulo is None or angleQd in range(int(quadrilDangulo) - 2, int(quadrilDangulo) + 2)) and \
                    (quadrilEangulo is None or angleQe in range(int(quadrilEangulo) - 2, int(quadrilEangulo) + 2)):
                frameFoto = frame
                frameFotoOriginal = frame
                frameFotoArray = frame2
                frameFoto = cv.resize(frameFoto, [235, 180], interpolation=cv.INTER_BITS)
                im = Image.fromarray(frameFoto)
                im2 = ImageTk.PhotoImage(image=im)
                gui.lblFoto.configure(image=im2)
                gui.lblFoto.image = im2


            # ---------------------------- SAIDA TELA ---------------------------
            #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            #frame2 = cv.cvtColor(frame2, cv.COLOR_RGB2BGR)
            frame = cv.resize(frame, [400, 300], interpolation=cv.INTER_BITS)
            frame2 = cv.resize(frame2, [400, 300], interpolation=cv.INTER_BITS)

            # --------------------- OPCOES DA GRAVACAO -------------------------
            if startGravacao:
                frameVideoOriginal.append(frame)
                frameVideoArray.append(frame2)


            im = Image.fromarray(frame)
            im2 = ImageTk.PhotoImage(image=im)
            gui.lblVideo.configure(image=im2)
            gui.lblVideo.image = im2

            im = Image.fromarray(frame2)
            im2 = ImageTk.PhotoImage(image=im)
            gui.lblVideo2.configure(image=im2)
            gui.lblVideo2.image = im2

        gui.win.update()


#-------------------------------------------------------WEBCAM--------------------------------------------
def addWebcam():
    global cap, frame2, novoArray, pointsToPaint, frame, interestPoint, \
        pointsToPaint, frameArray, frameArrayOriginal, video, frameCount, running, runningWeb

    global joelhoDangulo
    global setInterestAngleToColor
    global frameFoto, frameFotoOriginal, frameFotoArray, frameVideoOriginal, frameVideoArray, anglesToCompare

    gui.guiVisuVideoDelete()
    gui.guiWebcam()
    running = False
    runningWeb = True
    frameArrayOriginal = frameArrayOriginal.clear()
    frameArray = frameArray.clear()
    frameArrayOriginal = []
    frameArray = []
    frameCount = 0
    cap = cv.VideoCapture(0)

    resetParams()

    angleOd = None
    angleOe = None
    angleCd = None
    angleCe = None
    angleQd = None
    angleQe = None
    angleJd = None
    angleJe = None

    while runningWeb:
        ret, frame = cap.read()
        frame = cv.resize(frame, [480, 360], interpolation=cv.INTER_BITS)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, [480, 480], interpolation=cv.INTER_BITS)
        frame2 = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)  # criacao imagem preta
        #frameAngles = np.zeros((360, 160, 3), np.uint8)

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

        if (old_shaped[6][2] and old_shaped[8][2] and old_shaped[12][2]) > 0.25:  # ombro dir (6)
            angleOd = getAngleRealTime(shaped[6], shaped[8], shaped[12], 6)
        else:
            gui.varAngleOdValue.set(" ")

        if (old_shaped[5][2] and old_shaped[7][2] and old_shaped[11][2]) > 0.25:  # ombro esq (5)
            angleOe = getAngleRealTime(shaped[5], shaped[7], shaped[11], 5)
        else:
            gui.varAngleOeValue.set(" ")

        if (old_shaped[8][2] and old_shaped[6][2] and old_shaped[10][2]) > 0.25:  # cotovelo dir (8)
            angleCd = getAngleRealTime(shaped[8], shaped[6], shaped[10], 8)
        else:
            gui.varAngleCdValue.set(" ")

        if (old_shaped[7][2] and old_shaped[5][2] and old_shaped[9][2]) > 0.25:  # cotovelo esq (7)
            angleCe = getAngleRealTime(shaped[7], shaped[5], shaped[9], 7)
        else:
            gui.varAngleCeValue.set(" ")

        if (old_shaped[12][2] and old_shaped[6][2] and old_shaped[14][2]) > 0.4:  # qadril dir (12)
            angleQd = getAngleRealTime(shaped[12], shaped[6], shaped[14], 12)
        else:
            gui.varAngleQdValue.set(" ")

        if (old_shaped[11][2] and old_shaped[5][2] and old_shaped[13][2]) > 0.4:  # quadril esq (11)
            angleQe = getAngleRealTime(shaped[11], shaped[5], shaped[13], 11)
        else:
            gui.varAngleQeValue.set(" ")

        if (old_shaped[14][2] and old_shaped[12][2] and old_shaped[16][2]) > 0.4:  # joelho dir (14)
            angleJd = getAngleRealTime(shaped[14], shaped[12], shaped[16], 14)
        else:
            gui.varAngleJdValue.set(" ")

        if (old_shaped[13][2] and old_shaped[11][2] and old_shaped[15][2]) > 0.4:  # joelho esq (13)
            angleJe = getAngleRealTime(shaped[13], shaped[11], shaped[15], 13)
        else:
            gui.varAngleJeValue.set(" ")

        # ------------------ DESENHA PONTOS E RETAS -----------------------------
        draw_connections(frame, keypoints_with_scores, EDGES, attThr)
        draw_keypoints(frame, keypoints_with_scores, attThr)

        # # ----------------------DESENHA MOTION TRACKING--------------------
        if interestPoint is not 0:
            pointsToPaint.appendleft([shaped[interestPoint][1], shaped[interestPoint][0]])

            if len(pointsToPaint) > 15:
                motionTrailRealTime(pointsToPaint)

        # -------------------------COMPARADOR DE ANGULOS----------------------------------
        if (ombroEangulo is None or angleOe in range(int(ombroEangulo) - 2, int(ombroEangulo) + 2)) and \
                (ombroDangulo is None or angleOd in range(int(ombroDangulo) - 2, int(ombroDangulo) + 2)) and \
                (cotoveloDangulo is None or angleCd in range(int(cotoveloDangulo) - 2, int(cotoveloDangulo) + 2)) and \
                (cotoveloEangulo is None or angleCe in range(int(cotoveloEangulo) - 2, int(cotoveloEangulo) + 2)) and \
                (joelhoDangulo is None or angleJd in range(int(joelhoDangulo) - 2, int(joelhoDangulo) + 2)) and \
                (joelhoEangulo is None or angleJe in range(int(joelhoEangulo) - 2, int(joelhoEangulo) + 2)) and \
                (quadrilDangulo is None or angleQd in range(int(quadrilDangulo) - 2, int(quadrilDangulo) + 2)) and \
                (quadrilEangulo is None or angleQe in range(int(quadrilEangulo) - 2, int(quadrilEangulo) + 2)):
            frameFoto = frame
            frameFotoOriginal = frame
            frameFotoArray = frame2
            frameFoto = cv.resize(frameFoto, [235, 180], interpolation=cv.INTER_BITS)
            im = Image.fromarray(frameFoto)
            im2 = ImageTk.PhotoImage(image=im)
            gui.lblFoto.configure(image=im2)
            gui.lblFoto.image = im2


        # ---------------------------- SAIDA TELA ---------------------------
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #frame2 = cv.cvtColor(frame2, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, [400, 300], interpolation=cv.INTER_BITS)
        frame2 = cv.resize(frame2, [400, 300], interpolation=cv.INTER_BITS)

        # --------------------- OPCOES DA GRAVACAO -------------------------
        if startGravacao:
            frameVideoOriginal.append(frame)
            frameVideoArray.append(frame2)


        im = Image.fromarray(frame)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideo.configure(image=im2)
        gui.lblVideo.image = im2

        im = Image.fromarray(frame2)
        im2 = ImageTk.PhotoImage(image=im)
        gui.lblVideo2.configure(image=im2)
        gui.lblVideo2.image = im2

        gui.win.update()
