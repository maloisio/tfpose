import tkinter

import main
from tkinter import *
from tkinter import ttk

btn_next_frame = None
btn_run_frames = None
btn_previous_frame = None
btn_start_frames = None
btn_stop = None
btn_manipul_frames = None
pb = 0


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

    if btn_next_frame and btn_previous_frame and btn_run_frames is not NONE:
        btn_next_frame.place_forget()
        btn_run_frames.place_forget()
        btn_previous_frame.place_forget()
        btn_1x.place_forget()
        btn_2x.place_forget()

        win.update()

    btn_start_frames.place(x=30, y=400, width=40)
    # btn_stop = Button(win, text="⏸", bg="gray", fg="white", command=main.parar)
    # btn_stop.place(x=100, y=400, width=40)
    win.update()


def guiManipulLoadedFrames():
    btn_manipul_frames.place(x=30, y=450, width=40)
    win.update()


def guiManipulLoadedFramesDelete():
    btn_manipul_frames.place_forget()
    win.update()


def guiManipulCommands():
    global btn_next_frame, btn_run_frames, btn_previous_frame, btn_start_frames, btn_stop

    btn_start_frames.place_forget()
    btn_run_frames.place(x=30, y=500, width=40)
    btn_next_frame.place(x=130, y=500, width=40)
    btn_previous_frame.place(x=80, y=500, width=40)
    btn_2x.place(x=30, y=550, width=40)
    btn_1x.place(x=80, y=550, width=40)

    btn_MotionTrail_Md_Web.place(x=300, y=550, width=40)
    entryAngleMd.place(x=480, y=500, width=30, height=20)
    win.update()


def guiWebcamDelete():
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

    btn_MotionTrail_Md_Web.place(x=30, y=500, width=40)
    btn_MotionTrail_Me_Web.place(x=80, y=500, width=40)
    btn_MotionTrail_Jd_Web.place(x=130, y=500, width=40)
    btn_MotionTrail_Je_Web.place(x=180, y=500, width=40)
    btn_MotionTrail_Qd_Web.place(x=230, y=500, width=40)
    btn_MotionTrail_Qe_Web.place(x=280, y=500, width=40)
    btn_MotionTrail_Pd_Web.place(x=330, y=500, width=40)
    btn_MotionTrail_Pe_Web.place(x=380, y=500, width=40)
    btn_MotionTrail_Nda_Web.place(x=430, y=500, width=40)

    entryAngleOd.place(x=480, y=300, width=30, height=20)
    entryAngleOe.place(x=480, y=340, width=30, height=20)
    entryAngleCd.place(x=480, y=380, width=30, height=20)
    entryAngleCe.place(x=480, y=420, width=30, height=20)
    entryAngleQd.place(x=480, y=460, width=30, height=20)
    entryAngleQe.place(x=480, y=500, width=30, height=20)
    entryAngleJd.place(x=480, y=500, width=30, height=20)
    entryAngleJe.place(x=480, y=500, width=30, height=20)

    btn_atualizarAngle.place(x=510, y=500, width=30, height=20)
    #labelAngleMd.place(x=480, y=550)
    win.update()


win = Tk()
win.geometry("1125x600")

my_menu = Menu(win)
win.config(menu=my_menu)
add_video_menu = Menu(my_menu)
my_menu.add_cascade(label="Abrir", menu=add_video_menu)
add_video_menu.add_command(label="Arquivo", command=main.addVideo)
add_video_menu.add_command(label="Webcam", command=main.addWebcam)

btn_manipul_frames = Button(win, text="⏏", bg="gray", fg="white", command=main.reprise)
btn_start_frames = Button(win, text="▶", bg="gray", fg="white", command=main.start)
btn_run_frames = Button(win, text="⏯", bg="gray", fg="white", command=main.seguido)
btn_next_frame = Button(win, text="▶", bg="gray", fg="white", command=main.frente)
btn_previous_frame = Button(win, text="◀", bg="gray", fg="white", command=main.volta)

btn_MotionTrail_Md = Button(win, text="MD", bg="gray", fg="white", command=main.motionTrailMd)

btn_MotionTrail_Md_Web = Button(win, text="MD", bg="gray", fg="white", command=main.motionTrailMdRealTime)
btn_MotionTrail_Me_Web = Button(win, text="ME", bg="gray", fg="white", command=main.motionTrailMeRealTime)
btn_MotionTrail_Jd_Web = Button(win, text="JD", bg="gray", fg="white", command=main.motionTrailJdRealTime)
btn_MotionTrail_Je_Web = Button(win, text="JE", bg="gray", fg="white", command=main.motionTrailJeRealTime)
btn_MotionTrail_Qd_Web = Button(win, text="QD", bg="gray", fg="white", command=main.motionTrailQdRealTime)
btn_MotionTrail_Qe_Web = Button(win, text="QE", bg="gray", fg="white", command=main.motionTrailQeRealTime)
btn_MotionTrail_Pd_Web = Button(win, text="PD", bg="gray", fg="white", command=main.motionTrailPdRealTime)
btn_MotionTrail_Pe_Web = Button(win, text="PE", bg="gray", fg="white", command=main.motionTrailPeRealTime)
btn_MotionTrail_Nda_Web = Button(win, text="NDA", bg="gray", fg="white", command=main.motionTrailNdaRealTime)

btn_atualizarAngle = Button(win, text="Atualizar", bg="gray", fg="white", command=main.atualizarAngles)

btn_2x = Button(win, text="2x", bg="gray", fg="white", command=main.speed2x)
btn_1x = Button(win, text="1x", bg="gray", fg="white", command=main.speed1x)

varBarra = DoubleVar()
varBarra.set(0)


varAngleOd = tkinter.StringVar()
varAngleOe = tkinter.StringVar()
varAngleCd = tkinter.StringVar()
varAngleCe = tkinter.StringVar()
varAngleQd = tkinter.StringVar()
varAngleQe = tkinter.StringVar()
varAngleJd = tkinter.StringVar()
varAngleJe = tkinter.StringVar()


entryAngleOd = tkinter.Entry(win, textvariable = varAngleOd)
entryAngleOe = tkinter.Entry(win, textvariable = varAngleOe)
entryAngleCd = tkinter.Entry(win, textvariable = varAngleCd)
entryAngleCe = tkinter.Entry(win, textvariable = varAngleCe)
entryAngleQd = tkinter.Entry(win, textvariable = varAngleQd)
entryAngleQe = tkinter.Entry(win, textvariable = varAngleQe)
entryAngleJd = tkinter.Entry(win, textvariable = varAngleJd)
entryAngleJe = tkinter.Entry(win, textvariable = varAngleJe)
#labelAngleMd = tkinter.Label(win, textvariable = varAngleMd)

lblVideo = Label(win)
lblVideo.grid(column=0, row=2, columnspan=2)

win.mainloop()
