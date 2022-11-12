import tkinter

import main
from tkinter import *
from tkinter import ttk, LabelFrame

btn_next_frame = None
btn_run_frames = None
btn_previous_frame = None
btn_start_frames = None
btn_stop = None
btn_manipul_frames = None
pb = 0

secondWindow = None
frame_video_Original_Gravacao = None
frame_video_Array_Gravacao = None
lblVideoGravacaoOriginal = None
lblVideoGravacaoArray = None

def guiCreateProgressBar(valueBarraMax):
    global pb
    pb = ttk.Progressbar(win, variable=varBarra, maximum=valueBarraMax)
    pb.place(x=20, y=0)
    win.update()


def guiProgressBar(setValue):
    valueBarraMax = main.metaVideo()
    print(setValue)
    if setValue == valueBarraMax:
        pb.destroy()
        guiManipulLoadedFrames()
        win.update()

    else:
        varBarra.set(setValue)
        pb.place(x=20, y=0)
        win.update()


def guiStartLoadedFrames():
    global btn_next_frame, btn_run_frames, btn_previous_frame, btn_start_frames, btn_stop

    frame_label_motion_trail.place_forget()
    frame_label_thr.place_forget()

    frame_video_ArrayOriginal.place(x=20, y=10, width=400, height=300)
    frame_video_Array.place(x=430, y=10, width=400, height=300)

    frame_manipul_frames.place(x=20, y=420, width=170, height=150)
    frame_colors.place(x=250, y=420, width=300, height=150)

    if btn_next_frame and btn_previous_frame and btn_run_frames is not NONE:
        btn_next_frame.place_forget()
        btn_run_frames.place_forget()
        btn_previous_frame.place_forget()
        btn_1x.place_forget()
        btn_2x.place_forget()
        btn_Menos2x.place_forget()

        win.update()

    btn_start_frames.place(x=30, y=380, width=40)
    # btn_stop = Button(win, text="⏸", bg="gray", fg="white", command=main.parar)
    # btn_stop.place(x=100, y=400, width=40)
    win.update()


def guiManipulLoadedFrames():
    btn_manipul_frames.place(x=10, y=10, width=40)
    win.update()


def guiManipulLoadedFramesDelete():
    btn_manipul_frames.place_forget()
    win.update()


def guiManipulCommands():
    global btn_next_frame, btn_run_frames, btn_previous_frame, btn_start_frames, btn_stop

    btn_start_frames.place_forget()
    btn_run_frames.place(x=10, y=55, width=40)
    btn_previous_frame.place(x=60, y=55, width=40)
    btn_next_frame.place(x=110, y=55, width=40)
    btn_Menos2x.place(x=10, y=105, width=40)
    btn_1x.place(x=60, y=105, width=40)
    btn_2x.place(x=110, y=105, width=40)

    scale_hori_Red.place(x=10, y=10)
    scale_hori_Green.place(x=10, y=50)
    scale_hori_Blue.place(x=10, y=90)

    btn_MotionTrail_Md_Web.place(x=600, y=550, width=40)
    # entryAngleMd.place(x=480, y=500, width=30, height=20)
    win.update()


def guiVisualizarVideo():
    global secondWindow, frame_video_Original_Gravacao, frame_video_Array_Gravacao, lblVideoGravacaoArray, lblVideoGravacaoOriginal

    secondWindow = Toplevel(win)
    secondWindow.title("Visualizador Gravação")
    secondWindow.geometry("850x420")
    secondWindow['background'] = '#EEEED5'

    frame_video_Original_Gravacao = Frame(secondWindow, bg="red")
    frame_video_Array_Gravacao = Frame(secondWindow, bg="red")
    frame_manipul_frames_gravacao = Frame(secondWindow, relief="groove", borderwidth=4, bg='#CDCDB7')

    btn_run_frames_video_gravacao = Button(frame_manipul_frames_gravacao, text="⏯", bg="gray", fg="white", command=main.visualizarVideoGravacao)
    btn_avanca_frame_video_gravacao = Button(frame_manipul_frames_gravacao, text="▶", bg="gray", fg="white", command=main.frenteVideoGravacao)
    btn_retrocede_frame_video_gravacao = Button(frame_manipul_frames_gravacao, text="◀", bg="gray", fg="white", command=main.voltaVideoGravacao)
    btn_reset_frames_video_gravacao = Button(frame_manipul_frames_gravacao, text="⏏", bg="gray", fg="white", command=main.resetFramesVideoGravacao)
    btn_Menos2x_frames_video_gravacao = Button(frame_manipul_frames_gravacao, text="-2x", bg="gray", fg="white", command=main.speedMenos2xVideoGravacao)
    btn_1x_frames_video_gravacao = Button(frame_manipul_frames_gravacao, text="1x", bg="gray", fg="white", command=main.speed1xVideoGravacao)
    btn_2x_frames_video_gravacao = Button(frame_manipul_frames_gravacao, text="2x", bg="gray", fg="white", command=main.speed2xVideoGravacao)

    btn_reset_frames_video_gravacao.place(x=5, y=10, width=40)
    btn_run_frames_video_gravacao.place(x=55, y=10, width=40)

    btn_retrocede_frame_video_gravacao.place(x=105, y=10, width=40)
    btn_avanca_frame_video_gravacao.place(x=155, y=10, width=40)

    btn_Menos2x_frames_video_gravacao.place(x=205, y=10, width=40)
    btn_1x_frames_video_gravacao.place(x=255, y=10, width=40)
    btn_2x_frames_video_gravacao.place(x=305, y=10, width=40)

    frame_video_Array_Gravacao.place(x=430, y=10, width=400, height=300)
    frame_video_Original_Gravacao.place(x=20, y=10, width=400, height=300)
    frame_manipul_frames_gravacao.place(x=240, y=330, width=360, height=65)

    lblVideoGravacaoOriginal = Label(frame_video_Original_Gravacao)
    lblVideoGravacaoOriginal.grid()

    lblVideoGravacaoArray = Label(frame_video_Array_Gravacao)
    lblVideoGravacaoArray.grid()
    secondWindow.update()


def guiWebcamDelete():
    frame_motion_trail.place_forget()
    frame_thr.place_forget()
    frame_colors.place_forget()

    btn_MotionTrail_Md_Web.place_forget()
    btn_MotionTrail_Me_Web.place_forget()
    btn_MotionTrail_Jd_Web.place_forget()
    btn_MotionTrail_Je_Web.place_forget()
    btn_MotionTrail_Qd_Web.place_forget()
    btn_MotionTrail_Qe_Web.place_forget()
    btn_MotionTrail_Pd_Web.place_forget()
    btn_MotionTrail_Pe_Web.place_forget()
    btn_MotionTrail_Nda_Web.place_forget()

    win.update()


def guiWebcam():
    btn_start_frames.place_forget()
    btn_manipul_frames.place_forget()

    frame_video_ArrayOriginal.place(x=20, y=10, width=400, height=300)
    frame_video_Array.place(x=430, y=10, width=400, height=300)
    frame_video_foto.place(x=10, y=190, width=235, height=180)

    frame_label_motion_trail.place(x=10, y=400, width=170, height=160)
    frame_label_colors.place(x=240, y=400, width=380, height=160)
    frame_label_thr.place(x=690, y=400, width=70, height=160)
    frame_label_angles.place(x=840, y=10, width=260, height=435)
    frame_label_gravacao.place(x=840, y=450, width=260, height=70)
    # frame_label_angles.place(x=740, y=400, width=300, height=160)

    frame_manipul_frames.place_forget()
    frame_motion_trail.place(x=20, y=420, width=170, height=150)
    frame_colors.place(x=250, y=420, width=380, height=150)
    frame_thr.place(x=700, y=420, width=70, height=150)
    frame_angles.place(x=850, y=30, width=240, height=170)

    # frame_angles.place(x=750, y=420, width=300, height=150)

    frame_angulos_separados.place(x=430, y=310, width=402, height=70)

    # ---------------------------------- BOTOES -------------------------------
    btn_MotionTrail_Md_Web.place(x=10, y=5, width=40)
    btn_MotionTrail_Me_Web.place(x=60, y=5, width=40)
    btn_MotionTrail_Jd_Web.place(x=10, y=40, width=40)
    btn_MotionTrail_Je_Web.place(x=60, y=40, width=40)
    btn_MotionTrail_Qd_Web.place(x=10, y=75, width=40)
    btn_MotionTrail_Qe_Web.place(x=60, y=75, width=40)
    btn_MotionTrail_Pd_Web.place(x=10, y=110, width=40)
    btn_MotionTrail_Pe_Web.place(x=60, y=110, width=40)
    btn_MotionTrail_Nda_Web.place(x=110, y=110, width=40)
    btn_resetParams.place(x=1050, y=542, width=40)
    btn_saveFoto.place(x=10, y=380, width=40)
    btn_radio_foto_original.place(x=55, y=380)
    btn_radio_foto_linesPoints.place(x=130, y=380)
    btn_colorPoints.place(x=290, y=15, width=70)
    btn_colorLines.place(x=290, y=55, width=70)
    btn_colorMt.place(x=290, y=95, width=70)
    btn_atualizarAngle.place(x=90, y=130, width=50, height=20)
    btn_gravar_video.place(x=10, y=10, width=50, height=20)
    btn_pararGravacao_video.place(x=130, y=10, width=50, height=20)
    #btn_visualizar_video.place(x=180, y=10, width=60, height=20)

    #--------------------------------------- ENTRYS ----------------------------------
    entryAngleOd.place(x=90, y=5, width=30, height=20)
    entryAngleOe.place(x=180, y=5, width=30, height=20)
    entryAngleCd.place(x=90, y=35, width=30, height=20)
    entryAngleCe.place(x=180, y=35, width=30, height=20)
    entryAngleQd.place(x=90, y=65, width=30, height=20)
    entryAngleQe.place(x=180, y=65, width=30, height=20)
    entryAngleJd.place(x=90, y=95, width=30, height=20)
    entryAngleJe.place(x=180, y=95, width=30, height=20)


    scale_hori_Red.place(x=10, y=10)
    scale_hori_Green.place(x=10, y=50)
    scale_hori_Blue.place(x=10, y=90)
    scale_threshold.place(x=9, y=5)

    label_frameAngles_ombros.place(x=5, y=5)
    label_frameAngles_cotovelos.place(x=5, y=35)
    label_frameAngles_quadris.place(x=5, y=65)
    label_frameAngles_joelhos.place(x=5, y=95)

    label_frameAngles_ombroD.place(x=75, y=5)
    label_frameAngles_ombroE.place(x=165, y=5)
    label_frameAngles_cotoveloD.place(x=75, y=35)
    label_frameAngles_cotoveloE.place(x=165, y=35)
    label_frameAngles_quadrilD.place(x=75, y=65)
    label_frameAngles_quadrilE.place(x=165, y=65)
    label_frameAngles_joelhoD.place(x=75, y=95)
    label_frameAngles_joelhoE.place(x=165, y=95)

    # --------------------------------------- FRAME ANGULOS SEPARADOS ---------------------------
    label_anguloOmbroD.place(x=3, y=3)
    label_anguloOmbroE.place(x=3, y=23)
    label_anguloCotoveloD.place(x=95, y=3)
    label_anguloCotoveloE.place(x=95, y=23)
    label_anguloQuadrilD.place(x=210, y=3)
    label_anguloQuadrilE.place(x=210, y=23)
    label_anguloJoelhoD.place(x=310, y=3)
    label_anguloJoelhoE.place(x=310, y=23)

    label_anguloOmbroD_value.place(x=60, y=3)
    label_anguloOmbroE_value.place(x=60, y=23)
    label_anguloCotoveloD_value.place(x=162, y=3)
    label_anguloCotoveloE_value.place(x=162, y=23)
    label_anguloQuadrilD_value.place(x=267, y=3)
    label_anguloQuadrilE_value.place(x=267, y=23)
    label_anguloJoelhoD_value.place(x=364, y=3)
    label_anguloJoelhoE_value.place(x=364, y=23)

    label_frameAngles_valor_ombroD.place(x=120, y=35)
    win.update()


# ========================================= GUI ============================================
win = Tk()
win.geometry("1125x600")
win['background'] = '#EEEED5'
win.title("Análise de Movimento")

my_menu = Menu(win)
win.config(menu=my_menu)
add_video_menu = Menu(my_menu)
my_menu.add_cascade(label="Abrir", menu=add_video_menu)
add_video_menu.add_command(label="Arquivo", command=main.addVideo)
add_video_menu.add_command(label="Webcam", command=main.addWebcam)

# --------------------------- FRAMES ------------------------------------

frame_video_ArrayOriginal = Frame(win, bg="red")
frame_video_Array = Frame(win, bg="red")
frame_video_ArrayAngles = Frame(win, bg="red")


frame_label_motion_trail = LabelFrame(win, text="Motion Trail", bg='#EEEED5')
frame_label_colors = LabelFrame(win, text="Cores Pontos/Retas/Motion Trail", bg='#EEEED5')
frame_label_thr = LabelFrame(win, text="Threshold", bg='#EEEED5')
frame_label_angles = LabelFrame(win, text="Comparador de poses", bg='#EEEED5')
frame_label_gravacao = LabelFrame(win, text="Gravação", bg='#EEEED5')

frame_manipul_frames = Frame(win, relief="groove", borderwidth=4, bg='#CDCDB7')
frame_colors = Frame(win, relief="groove", borderwidth=4, bg='#CDCDB7')
frame_motion_trail = Frame(win, relief="groove", borderwidth=4, bg='#CDCDB7')
frame_thr = Frame(win, relief="groove", borderwidth=4, bg='#CDCDB7')
frame_angles = Frame(win, relief="groove", borderwidth=4, bg='#CDCDB7')
frame_video_foto = Frame(frame_label_angles, bg="red")

frame_angulos_separados = LabelFrame(win, bg='#EEEED5', text="Ângulos")

# ----------------------------------- BOTOES ----------------------------------
btn_manipul_frames = Button(frame_manipul_frames, text="⏏", bg="gray", fg="white", command=main.reprise)
btn_start_frames = Button(win, text="▶", bg="gray", fg="white", command=main.start)
btn_run_frames = Button(frame_manipul_frames, text="⏯", bg="gray", fg="white", command=main.seguido)
btn_next_frame = Button(frame_manipul_frames, text="▶", bg="gray", fg="white", command=main.frente)
btn_previous_frame = Button(frame_manipul_frames, text="◀", bg="gray", fg="white", command=main.volta)

btn_MotionTrail_Md = Button(win, text="MD", bg="gray", fg="white", command=main.motionTrailMd)

btn_MotionTrail_Md_Web = Button(frame_motion_trail, text="MD", bg="gray", fg="white",
                                command=main.motionTrailMdRealTime)
btn_MotionTrail_Me_Web = Button(frame_motion_trail, text="ME", bg="gray", fg="white",
                                command=main.motionTrailMeRealTime)
btn_MotionTrail_Jd_Web = Button(frame_motion_trail, text="JD", bg="gray", fg="white",
                                command=main.motionTrailJdRealTime)
btn_MotionTrail_Je_Web = Button(frame_motion_trail, text="JE", bg="gray", fg="white",
                                command=main.motionTrailJeRealTime)
btn_MotionTrail_Qd_Web = Button(frame_motion_trail, text="QD", bg="gray", fg="white",
                                command=main.motionTrailQdRealTime)
btn_MotionTrail_Qe_Web = Button(frame_motion_trail, text="QE", bg="gray", fg="white",
                                command=main.motionTrailQeRealTime)
btn_MotionTrail_Pd_Web = Button(frame_motion_trail, text="PD", bg="gray", fg="white",
                                command=main.motionTrailPdRealTime)
btn_MotionTrail_Pe_Web = Button(frame_motion_trail, text="PE", bg="gray", fg="white",
                                command=main.motionTrailPeRealTime)
btn_MotionTrail_Nda_Web = Button(frame_motion_trail, text="NDA", bg="gray", fg="white",
                                 command=main.motionTrailNdaRealTime)

btn_atualizarAngle = Button(frame_angles, text="Atualizar", bg="gray", fg="white", command=main.atualizarAngles)

btn_Menos2x = Button(frame_manipul_frames, text="-2x", bg="gray", fg="white", command=main.speedMenos2x)
btn_1x = Button(frame_manipul_frames, text="1x", bg="gray", fg="white", command=main.speed1x)
btn_2x = Button(frame_manipul_frames, text="2x", bg="gray", fg="white", command=main.speed2x)

btn_resetParams = Button(win, text="Reset", bg="gray", fg="white", command=main.resetParams)

btn_saveFoto = Button(frame_label_angles, text="Salvar", bg="gray", fg="white", command=main.saveFoto)

btn_colorPoints = Button(frame_colors, text="Pontos", bg="gray", fg="white", command=main.attColorPoints)
btn_colorLines = Button(frame_colors, text="Linhas", bg="gray", fg="white", command=main.attColorLines)
btn_colorMt = Button(frame_colors, text="MT", bg="gray", fg="white", command=main.attColorMt)

v = tkinter.IntVar()
btn_radio_foto_original = Radiobutton(frame_label_angles, variable=v, value=0, text="Original",
                                      command=main.selectOriginalPhoto, bg='#EEEED5')

btn_radio_foto_linesPoints = Radiobutton(frame_label_angles, variable=v, value=1, text="Linhas e Pontos",
                                         command=main.selectLinesAndPointsPhoto, bg='#EEEED5')

btn_gravar_video = Button(frame_label_gravacao, text="Gravar", bg="gray", fg="white", command=main.gravarVideo)
btn_pararGravacao_video = Button(frame_label_gravacao, text="Parar", bg="gray", fg="white", command=main.pararGravacaoVideo)
btn_visualizar_video = Button(frame_label_gravacao, text="Visualizar", bg="gray", fg="white", command=main.newSecondWindow)
# --------------------------------------- PROGRESS BAR --------------------------------
varBarra = DoubleVar()
varBarra.set(0)

# ----------------------------------------- SCALES ----------------------------------------------
scale_hori_Red = Scale(frame_colors, from_=0, to=255, bg='red', orient="horizontal", width=10, length=260,
                       command=main.update_corRed)
scale_hori_Green = Scale(frame_colors, from_=0, to=255, bg='green', orient="horizontal", width=10, length=260,
                         command=main.update_corGreen)
scale_hori_Blue = Scale(frame_colors, from_=0, to=255, bg='blue', orient="horizontal", width=10, length=260,
                        command=main.update_corBlue)

scale_threshold = Scale(frame_thr, from_=0, to=1, digits=3, resolution=0.01, bg='yellow', orient="vertical",
                        width=10, length=120, command=main.update_thr)

# ---------------------------------------- STRINGSVAR -------------------------------------------
varAngleOd = tkinter.StringVar()
varAngleOe = tkinter.StringVar()
varAngleCd = tkinter.StringVar()
varAngleCe = tkinter.StringVar()
varAngleQd = tkinter.StringVar()
varAngleQe = tkinter.StringVar()
varAngleJd = tkinter.StringVar()
varAngleJe = tkinter.StringVar()

varAngleOdValue = tkinter.StringVar()
varAngleOeValue = tkinter.StringVar()
varAngleCdValue = tkinter.StringVar()
varAngleCeValue = tkinter.StringVar()
varAngleQdValue = tkinter.StringVar()
varAngleQeValue = tkinter.StringVar()
varAngleJdValue = tkinter.StringVar()
varAngleJeValue = tkinter.StringVar()

# ---------------------------------------- ENTRYS -------------------------------------------
entryAngleOd = tkinter.Entry(frame_angles, textvariable=varAngleOd)
entryAngleOe = tkinter.Entry(frame_angles, textvariable=varAngleOe)
entryAngleCd = tkinter.Entry(frame_angles, textvariable=varAngleCd)
entryAngleCe = tkinter.Entry(frame_angles, textvariable=varAngleCe)
entryAngleQd = tkinter.Entry(frame_angles, textvariable=varAngleQd)
entryAngleQe = tkinter.Entry(frame_angles, textvariable=varAngleQe)
entryAngleJd = tkinter.Entry(frame_angles, textvariable=varAngleJd)
entryAngleJe = tkinter.Entry(frame_angles, textvariable=varAngleJe)
# labelAngleMd = tkinter.Label(win, textvariable = varAngleMd)


# --------------------------------------- LABELS -----------------------------------------
label_frameAngles_ombros = Label(frame_angles, text="Ombros", bg='#CDCDB7')
label_frameAngles_cotovelos = Label(frame_angles, text="Cotovelos", bg='#CDCDB7')
label_frameAngles_quadris = Label(frame_angles, text="Quadris", bg='#CDCDB7')
label_frameAngles_joelhos = Label(frame_angles, text="Joelhos", bg='#CDCDB7')

label_frameAngles_ombroD = Label(frame_angles, text="D:", bg='#CDCDB7')
label_frameAngles_ombroE = Label(frame_angles, text="E:", bg='#CDCDB7')
label_frameAngles_cotoveloD = Label(frame_angles, text="D:", bg='#CDCDB7')
label_frameAngles_cotoveloE = Label(frame_angles, text="E:", bg='#CDCDB7')
label_frameAngles_quadrilD = Label(frame_angles, text="D:", bg='#CDCDB7')
label_frameAngles_quadrilE = Label(frame_angles, text="E:", bg='#CDCDB7')
label_frameAngles_joelhoD = Label(frame_angles, text="D:", bg='#CDCDB7')
label_frameAngles_joelhoE = Label(frame_angles, text="E:", bg='#CDCDB7')

label_anguloOmbroD = Label(frame_angulos_separados, text="Ombro D:", bg='#EEEED5')
label_anguloOmbroE = Label(frame_angulos_separados, text="Ombro E:", bg='#EEEED5')
label_anguloCotoveloD = Label(frame_angulos_separados, text="Cotovelo D:", bg='#EEEED5')
label_anguloCotoveloE = Label(frame_angulos_separados, text="Cotovelo E:", bg='#EEEED5')
label_anguloQuadrilD = Label(frame_angulos_separados, text="Quadril D:", bg='#EEEED5')
label_anguloQuadrilE = Label(frame_angulos_separados, text="Quadril E:", bg='#EEEED5')
label_anguloJoelhoD = Label(frame_angulos_separados, text="Joelho D:", bg='#EEEED5')
label_anguloJoelhoE = Label(frame_angulos_separados, text="Joelho E:", bg='#EEEED5')

label_anguloOmbroD_value = Label(frame_angulos_separados, textvariable=varAngleOdValue, bg='#EEEED5')
label_anguloOmbroE_value = Label(frame_angulos_separados, textvariable=varAngleOeValue, bg='#EEEED5')
label_anguloCotoveloD_value = Label(frame_angulos_separados, textvariable=varAngleCdValue, bg='#EEEED5')
label_anguloCotoveloE_value = Label(frame_angulos_separados, textvariable=varAngleCeValue, bg='#EEEED5')
label_anguloQuadrilD_value = Label(frame_angulos_separados, textvariable=varAngleQdValue, bg='#EEEED5')
label_anguloQuadrilE_value = Label(frame_angulos_separados, textvariable=varAngleQeValue, bg='#EEEED5')
label_anguloJoelhoD_value = Label(frame_angulos_separados, textvariable=varAngleJdValue, bg='#EEEED5')
label_anguloJoelhoE_value = Label(frame_angulos_separados, textvariable=varAngleJeValue, bg='#EEEED5')

label_frameAngles_valor_ombroD = tkinter.Label(frame_angles, textvariable=varAngleCd,
                                               bg='#CDCDB7', width=3)

label_frameVideo_gravando = Label(frame_label_gravacao, font=("Arial", 8), text="Gravando...", bg='#EEEED5')

# --------------------------------------- TELAS FRAMES -------------------------------------------
lblVideo = Label(frame_video_ArrayOriginal)
lblVideo.grid()

lblVideo2 = Label(frame_video_Array)
lblVideo2.grid()

lblFoto = Label(frame_video_foto)
lblFoto.grid()





win.mainloop()
