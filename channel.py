import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

url = './test.jpg'

image = Image.open(url)
#T.Compose([T.GaussianBlur(3, sigma=(0.1, 2.0))])
#plt.ion()
#plt.imshow(image)
#plt.draw()
#plt.pause(0.001)

convert = T.ToTensor()
ten = convert(image)
print(ten)

#plt.imshow(ten.permute(1, 2, 0))
#plt.draw()
#plt.pause(0.001)

blur = T.Compose([T.Resize((224,224)), T.GaussianBlur(151, sigma=8.0)])
blrd = blur(ten[None, :, :])[0]

print(blrd)
print('now')

plt.imshow(blrd.permute(1, 2, 0))
plt.show()
#plt.draw()

print('now')

#time.sleep(12)

