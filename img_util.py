import numpy as np
import matplotlib.pyplot as plt

def visualize_image(image_array):
    image_to_visualize = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.figure()
    plt.imshow(image_to_visualize)
    plt.show()
