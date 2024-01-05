# Четверта лабораторна робота

# Імпортуємо необхідні бібліотеки та зчитаємо зображення:

import numpy as np
import random
from PIL import Image
import cv2 as cv
import random
import matplotlib.pyplot as plt

# Перше зображення
bear = cv.imread('bear.jpg', 0)
# Друге зображення
kyiv = cv.imread('kyiv.jpg', 0)
#Третє зображення
fuji = cv.imread('fuji.jpg', 0)
#Четверте зображення
buildings = cv.imread('buildings.jpg', 0)

fig, ((img1, img2), (img3, img4)) =  plt.subplots(2, 2)
fig.suptitle('Оригінальні зображення')
img1.set_title('bear')
img2.set_title('fuji')
img3.set_title('kyiv')
img4.set_title('buildings')
img1.imshow(bear, cmap='gray')
img2.imshow(fuji, cmap='gray')
img3.imshow(kyiv, cmap='gray')
img4.imshow(buildings, cmap='gray')


# Функції для шуму:

# Iмпульсний шум (всі функції для шуму взяті з noise.py)
def sp_noise_gray(image, prob=0.03):
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                image[i,j] = 0
            elif rdn > thres:
                image[i,j] = 255
    return image

#Нормальний шум
def norm_noise_gray(image, mean=0, var=0.1, a=0.5):
    sigma = var**0.5
    row,col= image.shape[:2]
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = a*image + (1-a)*gauss
    noisy = noisy-np.min(noisy)
    noisy = 255*(noisy/np.max(noisy))
    return noisy.astype(np.uint8)


noise_bear = sp_noise_gray(bear,0.07)
img1 = Image.fromarray(noise_bear)
img1


noise_bear1 = norm_noise_gray(bear, mean=0, var=10, a=0.1)
img10 = Image.fromarray(noise_bear1.astype(np.uint8))
img10


noise_fuji = sp_noise_gray(fuji,0.07)
img2 = Image.fromarray(noise_fuji)
img2

noise_fuji1 = norm_noise_gray(fuji, mean=0, var=10, a=0.1)
img20 = Image.fromarray(noise_fuji1.astype(np.uint8))
img20


noise_kyiv = sp_noise_gray(kyiv,0.07)
img3 = Image.fromarray(noise_kyiv)
img3


noise_kyiv1 = norm_noise_gray(kyiv, mean=0, var=10, a=0.1)
img30 = Image.fromarray(noise_kyiv1.astype(np.uint8))
img30


noise_buildings = norm_noise_gray(buildings, mean=0, var=10, a=0.1)
img4 = Image.fromarray(noise_buildings.astype(np.uint8))
img4


noise_buildings1 = norm_noise_gray(buildings, mean=0, var=10, a=0.1)
img40 = Image.fromarray(noise_buildings1.astype(np.uint8))
img40

def apply_mask(image, mask):
    result_rows = image.shape[0] - mask.shape[0]
    result_columns = image.shape[1] - mask.shape[1]
    result = np.zeros([result_rows, result_columns])
    for r in range(result_rows):
        for c in range(result_columns):
            val = np.multiply(image[r:r+mask.shape[0], c:c+mask.shape[1]], mask)
            val = np.sum(np.ravel(val))
            result[r][c] = val / np.prod(mask.shape)
    plt.figure()
    plt.imshow(result, cmap='gray')
    plt.show()


k = np.ones((5,5),np.float32)/(9)
apply_mask(noise_bear, k)


apply_mask(noise_bear1, k)


apply_mask(noise_fuji1, k)


apply_mask(noise_fuji1, k)


apply_mask(noise_kyiv, k)


apply_mask(noise_kyiv1, k)


apply_mask(noise_buildings, k)


apply_mask(noise_buildings1, k)


def median_filter(image, mask=5):
    bd = int(mask/2)
    median_img = np.zeros_like(image)
    for i in range(bd, image.shape[0] - bd):
        for j in range(bd, image.shape[1] - bd):
            kernel = np.ravel(image[i - bd : i + bd + 1, j - bd : j + bd + 1])
            median = np.sort(kernel)[np.int8(np.divide((np.multiply(mask, mask)), 2) + 1)]
            median_img[i, j] = median
    plt.figure()
    plt.imshow(median_img, cmap='gray')
    plt.show()


median_filter(noise_bear, mask=5)

median_filter(noise_bear1, mask=5)

median_filter(noise_fuji, mask=5)

median_filter(noise_fuji1, mask=5)

median_filter(noise_kyiv, mask=5)

median_filter(noise_kyiv1, mask=5)

median_filter(noise_buildings, mask=5)

median_filter(noise_buildings1, mask=5)
# Висновки:
# Найкращі результати показує медіанний фільтр для імпульсного шуму, особливо, якщо зображення велике і шуму на ньому було небагато. Трохи гірші результати для медіанного фільтру на нормальному шумі, і найгірші результати для згладжування коробкою.
