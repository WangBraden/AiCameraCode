import RPi.GPIO as GPIO
import errno
import fnmatch
import os
import os.path
import signal
import pygame
import stat
import threading
import gpiozero as GZ
import cv2
import numpy as np
import copy
from time import sleep
from pygame.locals import *
from thermal import ThermalCam
from RpiMotorLib import RpiMotorLib

# Global stuff -------------------------------------------------------------
powerButtonPin = 27
captureButtonPin = 17
backlightPin = 18

screenMode      =  3      # Current screen mode; default = viewfinder
screenModePrior = -1      # Prior screen mode (for detecting changes)
sizeMode = 2
saveIdx         = -1      # Image index for saving (-1 = none set yet)
loadIdx         = -1      # Image index for loading
scaled          = None    # pygame Surface w/last-loaded image

sizeData = [ # Camera parameters for different size settings
 # Full res      Viewfinder  Crop window
 [(2592, 1944), (320, 240), (0.0   , 0.0   , 1.0   , 1.0   )], # Large
 [(1920, 1080), (320, 180), (0.1296, 0.2222, 0.7408, 0.5556)], # Med
 [(1440, 1080), (320, 240), (0.2222, 0.2222, 0.5556, 0.5556)]] # Small

savePath = '/home/pi/Photos'

camera = None

thermal_frame = None
thermal_raw = None
thermal_lock = threading.Lock()

motor = RpiMotorLib.BYJMotor("MyMotor", "28BYJ")
motor_pins = [16, 19, 20, 26]
motor_pos = 0
motor_steps = 32
motor_actions = [
  True,
  False,
  False,
  True
]

bumper = GZ.Button(13)

quit_event = threading.Event()

def fix_perm(filename):
  # Set image file ownership to pi user, mode to 644
  os.chown(filename, uid, gid) # Not working, why?
  os.chmod(filename,
    stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

# Scan files in a directory, locating JPEGs with names matching the
# software's convention (IMG_XXXX.JPG), returning a tuple with the
# lowest and highest indices (or None if no matching files).
def imgRange(path):
    min = 9999
    max = 0
    try:
      for file in os.listdir(path):
        if fnmatch.fnmatch(file, 'PHOTO_[0-9][0-9][0-9][0-9][0-9].JPG'):
          i = int(file[6:-4])
          if(i < min): min = i
          if(i > max): max = i
    finally:
      return None if min > max else (min, max)


def takePicture():
    global busy, gid, loadIdx, saveIdx, scaled, sizeMode, savePath, uid

    print(f'taking picture #{saveIdx}')
    # Scan for next available image slot
    template = savePath + '/{}_{:05d}.{}'
    while True:
      filename = template.format('PHOTO', saveIdx, 'JPG')
      if not os.path.isfile(filename): break
      saveIdx += 1
      if saveIdx > 9999: saveIdx = 0

    scaled = None
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, sizeData[sizeMode][0][0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, sizeData[sizeMode][0][1])
    #camera.crop       = sizeData[sizeMode][2]
    try:
      ret, frame = camera.read()
      #frame = cv2.flip(frame, 0)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      cv2.imwrite(filename, frame)
      fix_perm(filename)
      img    = pygame.image.load(filename)
      scaled = pygame.transform.scale(img, sizeData[sizeMode][1])

      # save the thermal raw data
      filename = template.format('THERM', saveIdx, 'RAW')
      raw = None
      tf = None
      thermal_lock.acquire()
      if thermal_raw is not None:
        raw = copy.deepcopy(thermal_raw)
      if thermal_frame is not None:
        tf = copy.deepcopy(thermal_frame)
      thermal_lock.release()

      if raw is not None:
        with open(filename, "wb") as f:
          np.save(f, raw)
        fix_perm(filename)

      filename = template.format('THERM', saveIdx, 'JPG')
      if tf is not None:
        cv2.imwrite(filename, tf)
        fix_perm(filename)
     
    finally:
      # Add error handling/indicator (disk full, etc.)
      camera.set(cv2.CAP_PROP_FRAME_WIDTH, sizeData[sizeMode][1][0])
      camera.set(cv2.CAP_PROP_FRAME_HEIGHT, sizeData[sizeMode][1][1])

    if scaled:
      if scaled.get_height() < 240: # Letterbox
        screen.fill(0)
      screen.blit(scaled,
        ((320 - scaled.get_width() ) / 2,
         (240 - scaled.get_height()) / 2))
      pygame.display.update()
      #time.sleep(2.5)
      loadIdx = saveIdx     

def calibrate_motor():
    # goes to far left until bumper is triggered
    steps = 0
    while not bumper.is_pressed and steps < 80:
      motor.motor_run(motor_pins, .001, 1, True, False, "half", .01)
      steps += 1
    # return to center
    motor.motor_run(motor_pins, .001, motor_steps, False, False, "half", .05)
    return

def thermal_main():
    global thermal_frame, thermal_raw
    global motor_actions, motor_pos, motor_steps

    mlx = ThermalCam()

    while not quit_event.wait(0.3):
        # move motor
        ccw = motor_actions[motor_pos]
        # (GPIOPins, stepdelay, steps, counterclockwise, verbose, steptype, initdelay)
        motor.motor_run(motor_pins, .001, motor_steps, ccw, False, "half", .05)
        motor_pos = (motor_pos + 1) % len(motor_actions)
        sleep(0.05)

        h = 240
        w = 320
        f, raw = mlx.capture((w, h))
        thermal_lock.acquire()
        thermal_frame = f
        thermal_raw = raw
        thermal_lock.release()


# Initialization -----------------------------------------------------------

# Init framebuffer/touchscreen environment variables
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.putenv('SDL_FBDEV'      , '/dev/fb1')
#os.putenv('SDL_MOUSEDRV'   , 'TSLIB')
#os.putenv('SDL_MOUSEDEV'   , '/dev/input/touchscreen')
os.putenv('DISPLAY'   , '')

# Get user & group IDs for file & folder creation
# (Want these to be 'pi' or other user, not root)
s = os.getenv("SUDO_UID")
uid = int(s) if s else os.getuid()
s = os.getenv("SUDO_GID")
gid = int(s) if s else os.getgid()

print("==> init pygame")
# Init pygame and screen
pygame.init()
print("==> setup display")
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
pygame.display.update()

print("==> open camera")
# Init camera and set up default values
camera            = cv2.VideoCapture(0)
if camera is None:
  print('failed to open camera')
  os.exit(-1)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, sizeData[sizeMode][1][0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, sizeData[sizeMode][1][1])

print("==> prepare save folder")
r = imgRange(savePath)
if r is None:
  saveIdx = 1
else:
  saveIdx = r[1] + 1
  if saveIdx > 9999: saveIdx = 0

CAPTURE_EVENT = pygame.USEREVENT + 1
SHOW_NUM_PICS_EVENT = pygame.USEREVENT + 2
QUIT_EVENT = pygame.USEREVENT + 4

button_was_held = False

def button_pressed(b):
  global button_was_held
  button_was_held = False

def button_on_hold(b: GZ.Button):
  global button_was_held
  button_was_held = True
  if b.held_time < 3:
    pygame.event.post(pygame.event.Event(SHOW_NUM_PICS_EVENT))

def button_released(b):
  if not button_was_held:
    pygame.event.post(pygame.event.Event(CAPTURE_EVENT))

print("==> setup GPIO pins")
# Init the pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

cb = GZ.Button(captureButtonPin, bounce_time=0.05, hold_time=1)
cb.when_pressed = button_pressed
cb.when_held = button_on_hold
cb.when_released = button_released

pb = GZ.Button(powerButtonPin, bounce_time=0.05)
pb.when_pressed = lambda: pygame.event.post(pygame.event.Event(QUIT_EVENT))

# turn the backlight on.
GPIO.setup(backlightPin, GPIO.IN)

# calibrate the camera motor
print("==> calibrate motor")
calibrate_motor()

# start the thermal capture thread
print("==> starting thermal camera")
thermal_thread = threading.Thread(target=thermal_main)
thermal_thread.start()

print("==> install signal handler")
def quit_signal(signum, f):
  pygame.event.post(pygame.event.Event(QUIT_EVENT))

# capture signals
signal.signal(signal.SIGINT, quit_signal)
signal.signal(signal.SIGTERM, quit_signal)
 
# Main loop ----------------------------------------------------------------
print ("==> start main loop")
try:
  show_num_pics = False
  running = True
  while running:

    for event in pygame.event.get():
      if event.type == QUIT_EVENT:
        running = False
        break
      elif event.type == SHOW_NUM_PICS_EVENT:
        show_num_pics = not show_num_pics
      elif event.type == CAPTURE_EVENT:
        takePicture()
      else:
        pass
    
    # Refresh display
    if screenMode >= 3: # Viewfinder or settings modes
      ret, frame = camera.read()
      frame = cv2.resize(frame, (320, 240), interpolation =cv2.INTER_AREA)
      #frame = cv2.flip(frame, 0)

      tf = None
      thermal_lock.acquire()
      if thermal_frame is not None:
        tf = copy.deepcopy(thermal_frame)
      thermal_lock.release()
      if tf is not None:
        frame = cv2.addWeighted(frame, 0.5, tf, 0.5, 0)
      
      if show_num_pics:
        numPics = saveIdx - 1
        text = f'Total pics: {numPics}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.putText(frame, text, (0, textSize[1]), font,
          1, (255, 255, 255), 2)

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      img = pygame.image.frombuffer(frame, frame.shape[1::-1], 'RGB')
    elif screenMode < 2: # Playback mode or delete confirmation
      img = scaled       # Show last-loaded image
    else:                # 'No Photos' mode
      img = None         # You get nothing, good day sir

    if img is None or img.get_height() < 240: # Letterbox, clear background
      screen.fill(0)
    if img:
      screen.blit(img,
        ((320 - img.get_width() ) / 2,
        (240 - img.get_height()) / 2))
      
    pygame.display.update()

    screenModePrior = screenMode
except KeyboardInterrupt:
  pass

quit_event.set()

screen.fill(0)
pygame.display.update()
pygame.quit()

thermal_thread.join()

GPIO.setup(backlightPin, GPIO.OUT)
GPIO.output(backlightPin, 0)
