import cv2
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
#ПЕРЕВЕДЕННЯ В ЧБ
input_image = imread("C:\\Users\\Anastasiia\\Desktop\\iris_0.jpg")#Зчитуємо зображення
r, g, b = input_image[:,:,0], input_image[:,:,1], input_image[:,:,2]#Розділяємо на кольорові канали
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722#Визначаємо константи (В презентації на лекції були інші, але в Вікіпедії написано,
#що такі значення підходять краще
grayscale_image = r*r_const+g*g_const+b*b_const#Створюємо напівтонове зображення
fig = plt.figure(1)#Створюємо фігуру для виведення графіку
img1, img2 = fig.add_subplot(121), fig.add_subplot(122)
img1.imshow(input_image)
img2.imshow(grayscale_image, cmap = plt.cm.get_cmap('gray'))#В matplotlib зображення за замовчуванням представляється в неонових кольорах,
#тому вживаєму функцію plt.cm.get_cmap
#Якщо застосувати функцію imwrite з cv2, то завантажене зображення буде чорно-білим
fig.show()
plt.show()
#БІНАРИЗАЦІЯ (наскільки я зрозуміла,
#не можна використовувати сторонні модулі
#тільки в переведенні до чб)
mask = cv2.imread("C:\\Users\\Anastasiia\\Desktop\\iris_1.jpg")#Зчитуємо зображення
threshold_value = 200#Встановлюємо порогове значення
max_val = 255#Встановлюємо максимальне значення
ret, image = cv2.threshold(mask, threshold_value, max_val, cv2.THRESH_BINARY_INV)#Використовуємо функцію threshold з модуля cv2
#НАКЛАДАННЯ МАСКИ
coloured = cv2.imread("C:\\Users\\Anastasiia\\Desktop\\iris_0.jpg")#Зчитуємо кольорове зображення 
mask_1 = cv2.imread("C:\\Users\\Anastasiia\\Desktop\\iris_2.jpg")#Зчитуємо бінаризоване зображення
masked = cv2.bitwise_and(coloured, mask_1, mask=None)#Використовуємо функцію cv2 для того, щоб накласти одне зображення на інше
imS = cv2.resize(masked, (960, 800))#Змінюємо розмір зображення, щоб його було зручніше переглядати
ims = cv2.resize(image, (960, 800))#Змінюємо розмір другого зображення, щоб його було зручніше переглядати
cv2.imshow('InverseBinaryThresholding', ims)#Показуємо зображення
cv2.imshow("Mask Applied to Image", imS)#Показуємо зображення
cv2.waitKey(0)#Функція, яка потрібна, щоб зображення відобразилося одразу

