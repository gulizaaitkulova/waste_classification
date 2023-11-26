import time
import json

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image


RECYCLE = [440, 441,504, 509, 510, 511, 521, 523, 529, 531, 540, 548, 549, 553, 563, 572, 574, 575, 580, 582, 592, 598, 
           599, 605, 620, 622, 634, 635, 646, 649, 651, 688, 696, 703, 704, 705, 706, 707, 709, 710, 711, 712, 713, 719, 720, 
           721, 726, 728, 734, 739, 740, 742, 743, 745, 746, 756, 760, 761, 768, 772, 773, 777, 782, 783, 784, 785, 786, 
           792, 798, 799, 800,801, 802, 803, 804, 805, 806, 807, 809, 811, 813, 814, 816, 821, 822, 823, 827, 828, 831,
           834, 839, 841, 843, 844, 845, 846, 849, 850, 852, 859, 860, 861, 864, 866, 868, 870, 872, 873, 875, 878, 879,
           882, 883, 886, 898, 899, 900, 901, 902, 906, 909, 910]
COMPOST = [924, 925, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963,
           964, 965, 966, 967, 969, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998]

# Specify the path to Imagenet labels
json_file_path = 'classes.json'

# Open the JSON file and load its content into a Python dictionary
with open(json_file_path, 'r') as file:
    classes = json.load(file)


torch.backends.quantized.engine = 'qnnpack'

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

# load image
image = cv2.imread('images/banana.jpeg')

# convert opencv output from BGR to RGB
image = image[:, :, [2, 1, 0]]
permuted = image

# preprocess
input_tensor = preprocess(image)

# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# run model
output = net(input_batch)
# do something with output ...

# log model performance
frame_count += 1
now = time.time()
# if now - last_logged > 1:
#     print(f"{frame_count / (now-last_logged)} fps")
#     last_logged = now
#     frame_count = 0

top = list(enumerate(output[0].softmax(dim=0)))
top.sort(key=lambda x: x[1], reverse=True)
for idx, val in top[:1]:
    if val.item() > 0.5:
        result = "Non Recyclable"
        if idx in RECYCLE:
            result = "Recyclable"
        if idx in COMPOST:
            result = "Compost"
        print(f"{result} {val.item()*100:.2f}%")
