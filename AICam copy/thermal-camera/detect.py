import models
import torch
from torchvision import transforms
from PIL import Image

def gen_image(data):
    # convert to grayscale image
    data = data * 10 - 100
    data[data < 0] = 0
    data[data > 255] = 255
    img = Image.new('L', (32, 24))
    img.putdata(data)
    return img

class ThermalAI:
    def __init__(self):
        self.model = models.ThermalCNN(2)
        self.model.eval()
        self.model.load_state_dict(torch.load('/home/pi/thermal-camera/model.pt'))
        self.txfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.44,),(0.1,),)])
        return

    def detect(self, data):
        # convert to grayscale image
        img = gen_image(data)
        img = self.txfm(img)
        # add a dummy batch dimension to fit the model which accepts a batch of images.
        img = img[None, :]
        with torch.no_grad():
            output = self.model(img)
            return torch.argmax(output)
