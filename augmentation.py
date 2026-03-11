import numpy as np
import cv2

def add_salt_and_pepper_noise(image, amount=0.04, salt_vs_pepper=0.5):
    noisy_img = np.copy(image)
    
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))
    
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_img[tuple(coords_salt)] = 255
    
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_img[tuple(coords_pepper)] = 0
    
    return noisy_img

def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)

def histogram_equalization(image):
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    
    elif len(image.shape) == 3:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)