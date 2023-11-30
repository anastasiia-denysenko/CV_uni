import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
#Зчитуємо зображення в вигляді двовимірного масиву
img = cv2.imread("C:\\Users\\Anastasiia\\Desktop\\img.jpg", 0)
#Стфорюємо структурний елемент
kernal = np.matrix('1, 0, 1; 0, 0, 0; 1, 0, 1')
#Функція ерозії
def erosian(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    y = kernal.shape[0] // 2
    x = kernal.shape[1] // 2
    processed_image = np.copy(img)
    for i in range(y, img.shape[0] - y):
        for j in range(x, img.shape[1] - x):
            local_window = img[i-y:i+y+1, j-x:j+x+1]
            processed_image[i][j] = np.min(local_window[kernal])
    return processed_image 
#Функція нарощування
def dilate(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    y = kernal.shape[0] // 2
    x = kernal.shape[1] // 2
    processed_image = np.copy(img)
    for i in range(y, img.shape[0] - y):
        for j in range(x, img.shape[1] - x):
            local_window = img[i-y:i+y+1, j-x:j+x+1]
            processed_image[i][j] = np.max(local_window[kernal])
    return processed_image 
#Функція розмикання (викликаємо дві попередні по черзі)
def opening(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    temp = erosian(img, kernal)
    output = dilate(temp, kernal)
    return output
#Функція замикання (викликаємо дві перші в зворотньому порядку)
def closing(img: np.ndarray, kernal: np.ndarray) -> np.ndarray:
    temp = dilate(img, kernal)
    output = erosian(temp, kernal)
    return output
#Показуємо зображення, структурний елемент і результат застосування кожної з функцій
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
plt.figure()
plt.imshow(kernal, cmap='gray')
plt.show()
plt.figure()
plt.imshow(erosian(img, kernal), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(dilate(img, kernal), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(opening(img, kernal), cmap = 'gray')
plt.show()
plt.figure()
plt.imshow(closing(img, kernal), cmap = 'gray')
plt.show()
