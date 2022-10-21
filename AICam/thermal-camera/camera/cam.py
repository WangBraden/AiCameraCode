# import the opencv library
import cv2
import threading
import copy
import numpy as np
import thermal
import gpiozero as GZ
  
cam_frame = None
cam_lock = threading.Lock()

cam_width = 0
cam_height = 0

thermal_frame = None
thermal_lock = threading.Lock()

frame = None
frame_lock = threading.Lock()
frame_avail = threading.Event()
quit = threading.Event()

def cam_main():
    global cam_frame, quit, cam_width, cam_height
    # define a video capture object
    vid = cv2.VideoCapture(0)
    cam_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(cam_width, cam_height)

    while not quit.wait(0.03):
        ret, f = vid.read()
        f = cv2.flip(f, 1)
        cam_lock.acquire()
        cam_frame = f
        cam_lock.release()

    vid.release()

def thermal_main():
    global thermal_frame, cam_width, cam_height
    while cam_height == 0:
        quit.wait(0.1)

    mlx = thermal.init()

    while not quit.wait(0.03):
        h = cam_height
        w = cam_width
        f, _ = thermal.capture(mlx, (w, h))
        thermal_lock.acquire()
        thermal_frame = f
        thermal_lock.release()

def compositor_main():
    global frame
    while not quit.wait(0.03):
        f = None
        cam_lock.acquire()
        if cam_frame is not None:
            f = copy.deepcopy(cam_frame)
        cam_lock.release()
        if f is None:
            continue

        f2 = None
        thermal_lock.acquire()
        if thermal_frame is not None:
            f2 = copy.deepcopy(thermal_frame)
        thermal_lock.release()

        if f2 is not None:
           f = cv2.addWeighted(f, 0.5, f2, 0.5, 0)

        frame_avail.set()
        frame_lock.acquire()
        frame = f
        frame_lock.release()

funcs = [cam_main, thermal_main, compositor_main]
threads = []

for f in funcs:
    th = threading.Thread(target=f)
    threads.append(th)
    th.start()

save_button = GZ.Button(27)
save_button.when_deactivated = lambda b: quit.set()

while not quit.is_set():

    if frame_avail.wait(0.1):
        frame_avail.clear()
        frame_lock.acquire()
        f = copy.deepcopy(frame)
        frame_lock.release()
        cv2.imshow('frame', f)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit.set()
  
for th in threads:
    th.join()

# After the loop release the cap object
# Destroy all the windows
cv2.destroyAllWindows()
