import time
import json

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image


RECYCLE = [401, 402, 404, 405, 407, 409, 411, 413, 414, 417, 418, 419, 420, 421, 423, 426, 427, 428, 431, 433, 
           434, 435, 436, 437, 438, 439, 441, 442, 447, 448, 449, 450, 451, 455, 457, 459, 460, 466, 467, 469, 
           473, 474, 477, 479, 480, 481, 482, 484, 485, 487, 494, 496, 501, 503, 504, 505, 506, 508, 509, 510, 
           511, 512, 514, 516, 517, 518, 520, 522, 523, 524, 526, 527, 528, 532, 533, 534, 535, 536, 538, 539, 
           540, 542, 544, 545, 546, 548, 550, 551, 554, 556, 558, 559, 563, 564, 566, 567, 568, 572, 576, 578, 
           580, 581, 582, 583, 585, 586, 588, 590, 591, 593, 594, 595, 596, 599, 601, 602, 603, 604, 606, 608, 
           609, 610, 611, 612, 613, 614, 616, 618, 619, 620, 621, 622, 624, 627, 628, 631, 633, 634, 635, 637, 
           638, 640, 641, 642, 647, 648, 649, 650, 651, 654, 655, 656, 660, 661, 662, 663, 664, 665, 670, 671, 
           672, 678, 680, 681, 683, 684, 686, 687, 689, 693, 694, 695, 696, 700, 701, 702, 704, 707, 709, 710, 
           711, 712, 713, 714, 719, 720, 721, 723, 725, 726, 728, 730, 731, 732, 739, 741, 744, 745, 746, 748, 
           750, 756, 758, 759, 762, 765, 766, 768, 771, 777, 780, 784, 789, 795, 797, 800, 804, 827, 849, 851, 
           882, 883, 887, 898, 899, 901, 902, 904, 905, 907, 908, 909, 910, 911, 912, 914, 915, 916, 917, 918, 
           919, 920, 921, 922, 923]
COMPOST = [416, 432, 446, 453, 454, 456, 458, 471, 475, 481, 492, 493, 495, 521, 530, 531, 537, 541, 543, 545, 
           552, 553, 561, 565, 574, 577, 579, 584, 589, 596, 597, 600, 626, 630, 642, 646, 655, 657, 658, 663, 
           664, 666, 667, 669, 671, 674, 675, 676, 678, 679, 680, 682, 683, 684, 688, 691, 697, 705, 706, 708, 
           716, 717, 718, 722, 724, 727, 729, 734, 735, 736, 737, 738, 740, 742, 743, 747, 749, 751, 754, 755, 
           757, 760, 761, 763, 767, 770, 772, 773, 774, 775, 776, 778, 779, 781, 785, 787, 790, 791, 792, 793, 
           798, 799, 801, 802, 803, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 
           820, 821, 822, 823, 824, 825, 826, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 
           841, 842, 843, 844, 845, 846, 848, 850, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 
           865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 879, 880, 881, 884, 885, 886, 889, 
           890, 891, 893, 894, 895, 896, 897, 926, 927, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 
           940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 
           960, 961, 962, 963, 964, 965, 966, 967, 968, 969]

# Specify the path to Imagenet labels
json_file_path = 'classes.json'

# Open the JSON file and load its content into a Python dictionary
with open(json_file_path, 'r') as file:
    classes = json.load(file)


torch.backends.quantized.engine = 'qnnpack'

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

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
