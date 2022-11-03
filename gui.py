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

        win.update()

    btn_start_frames.place(x=30, y=400, width=40)
    # btn_stop = Button(win, text="⏸", bg="gray", fg="white", command=main.parar)
    # btn_stop.place(x=100, y=400, width=40)
    win.update()


def guiWhenPlayingFrames():
    #btn_previous_frame.place_forget()
    #btn_next_frame.place_forget()
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
    btn_next_frame.place(x=120, y=500, width=40)
    btn_previous_frame.place(x=70, y=500, width=40)

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
btn_previous_stop = Button(win, text="⏯", bg="gray", fg="white", command=main.stop)

varBarra = DoubleVar()
varBarra.set(0)

lblVideo = Label(win)
lblVideo.grid(column=0, row=2, columnspan=2)

win.mainloop()
