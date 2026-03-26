import numpy as np

def add_salt_and_pepper_noise(image, amount=0.04):
    noisy_img = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_img[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_img[tuple(coords)] = 0
    return noisy_img

def custom_gaussian_kernel(size, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel) 

def apply_convolution(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    padded_img = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    
    for y in range(i_h):
        for x in range(i_w):
            region = padded_img[y:y+k_h, x:x+k_w]
            output[y, x] = np.sum(region * kernel)
            
    return np.clip(output, 0, 255).astype(np.uint8)

def custom_median_filter(image, size=3):
    i_h, i_w = image.shape
    pad = size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)
    
    for y in range(i_h):
        for x in range(i_w):
            window = padded_img[y:y+size, x:x+size].flatten()
            output[y, x] = np.median(window)
            
    return output

def custom_histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    cdf = hist.cumsum()
    
    cdf_m = np.ma.masked_equal(cdf, 0) 
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    return cdf[image]