import cv2

index = 1

while(True):
    fileName = '/home/pi/Photos/PHOTO_{:05d}.JPG'.format(index)
    photo = cv2.imread(fileName)
    cv2.imshow(str(index), photo)
    key = cv2.waitKey(0)
    if key & 0xff == ord('q'):
        break
    index+=1

cv2.destroyAllWindows()
    