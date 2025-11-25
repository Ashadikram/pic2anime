import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Original image")
    #plt.axes("off")
    plt.show()
    return img

filename = "Anime Character project using operncv/ashad_picture.jpg"

img = read_file(filename)

#crea edge mask

def edge_mask(img, line_size, blur_value):
    """
    input: Gray Scale image
    output: Edges off image
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)

    edges = cv2.adaptiveThreshold(gray_blur,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)

    return edges

line_size, blur_value = 5, 7
edges = edge_mask(img, line_size, blur_value)

plt.imshow(edges, cmap= "binary")
plt.title("After edge masking")
plt.axis('off')
plt.show()

#Reduce thw Color palette

def color_quantization(img , k):

    #Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Datemine Criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implreminting k-means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result

img = color_quantization(img , k=9)

plt.imshow(img)
plt.title("After color quantization")
plt.show()

#Reduce the noisc
blurred = cv2.bilateralFilter(img , d= 3, sigmaColor= 200,sigmaSpace= 200)

plt.imshow(blurred)
plt.title("After clearing a blur")

plt.show()

#Combine Edge Mask with the quantize img

def cartoon():
    c = cv2.bitwise_and(blurred, blurred, mask=edges)

    plt.imshow(c)
    plt.title("After merging a two pictures")
    plt.show()

cartoon()