import numpy as np
import imageio as image
import matplotlib.pyplot as plt

#1. membuat representasi citra 3x3 dengan nilai RGB
image_data = np.array([[[123,234,45],[67,89,12],[200,100,50]],
                       [[145,205,255],[30,60,90],[10,150,75]],
                       [[255,255,0],[80,90,100],[220,50,70]]],dtype=np.uint8)
plt.imshow(image_data)
plt.title("RGB Image")
plt.show()

#2. mengubah gambar menjadi brightness
brightness_factor = 100 
image_bright = np.clip(image_data + brightness_factor,0,255).astype(np.uint8)
plt.imshow(image_bright)
plt.title("Brightness Image")
plt.show()

#3. mengubah gambar brightness menjadi greyscale
image_greyscale = np.dot(image_bright[...,:3],[0.2989, 0.587, 0.114]).astype(np.uint8)
plt.imshow(image_greyscale,cmap = 'gray')
plt.title("greyscale Image")
plt.show()

#4. melakukan kontras dengan faktor 1.5 pada gambar greyscale
contrast_factor = 1.5
mean_grey = np.mean(image_greyscale)
image_contrast = np.clip((image_greyscale - mean_grey)*contrast_factor + mean_grey,0,225).astype(np.uint8)
plt.imshow(image_contrast,cmap='gray')
plt.title("contrast with factor 1.5")
plt.show()

#5. mengubah gambar greyscale menjadi threshold level 50
thresthold_level = 50
image_thresthold = (image_greyscale > thresthold_level) * 255
plt.imshow(image_thresthold, cmap='gray')
plt.title("Threshold level 50")
plt.show()

#6. mengubah gambar RGB menjadi negative
image_negative = 255 - image_data
plt.imshow(image_negative)
plt.title("image_negative")
plt.show()

#7. histogram
def histogram_equalization(image):
    histogram, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = histogram.cumsum() 
    cdf_normalized = cdf * 255 / cdf[-1] 
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return image_equalized.reshape(image.shape).astype(np.uint8)

image_hist_eq = histogram_equalization(image_greyscale)

plt.imshow(image_hist_eq, cmap='gray')
plt.title("Histogram Equalization")
plt.show()
print(image_hist_eq)