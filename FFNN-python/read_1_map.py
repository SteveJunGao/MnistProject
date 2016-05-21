from read_map import *
import numpy as np
import matplotlib.pyplot as plt


train_image = get_train_images()
train_label = get_train_labels()
test_image = get_test_images()
test_label = get_test_labels()

for x in range(100):
    print train_label[x]
    im = train_image[x]
    im = im.reshape(28,28)
    fig = plt.figure
    plt.imshow(im,cmap = "binary")
    plt.show()

