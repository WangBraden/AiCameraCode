import numpy as np
import cv2
import adafruit_mlx90640
import board, busio
import time

frequency = 1000000
width = 32
height = 24

def init():
    # normal I2C bus 1
    i2c = busio.I2C(board.SCL, board.SDA, frequency=frequency)
    ###
    # use I2C bus 0
    #i2c = busio.I2C(board.D1, board.D0, frequency=frequency)
    addrs = i2c.scan()
    if 0x33 not in addrs:
        raise RuntimeError('MLX90640 not found on i2c bus')

    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    return mlx


def capture(mlx, dim):
    frame = np.zeros((width * height, ))
    while True:
        try:
            mlx.getFrame(frame)
        except IOError as e:
            print(e)
            time.sleep(0.05)
            continue
        except Exception as e:
            print(e)
            time.sleep(2)
            continue
        else:
            break

    # tempratures: min, max, and center
    t_min = np.min(frame)
    t_max = np.max(frame)
    t_center = frame[height//2*width + width//2]

    # normalize the tempratures and convert to 0-255
    image = np.uint8((frame - t_min) * 255 / (t_max-t_min))
    image.shape = (height, width)
    # apply JET color map to convert to RGB image
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    # resize the image to the required dimension
    image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    # draw a cross in the center and show the temprature of the center.
    text = f"{t_center:.1f}C"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textSize = cv2.getTextSize(text, font, .4, 1)[0]
    textPos = (dim[0]//2 - textSize[0], (dim[1] - textSize[1])//2)
    cv2.putText(image, text, textPos, font, .4, (255,255,255), 1)

    cv2.drawMarker(image, (dim[0]//2, dim[1]//2), (255, 255, 255),
        markerType=cv2.MARKER_CROSS, markerSize=30)

    return image, frame
