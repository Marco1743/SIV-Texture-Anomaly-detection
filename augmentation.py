import numpy as np

def add_salt_and_pepper_noise(image, amount=0.04):
    """Aggiunge rumore impulsivo (Salt & Pepper)."""
    noisy_img = np.copy(image)
    # Sale
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy_img[tuple(coords)] = 255
    # Pepe
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy_img[tuple(coords)] = 0
    return noisy_img

def custom_gaussian_kernel(size, sigma=1.0):
    """
    A. GENERAZIONE KERNEL GAUSSIANO (LSI System)
    Implementa la formula: h(m,n) = (1/2pi*sigma^2) * exp(-(m^2+n^2)/(2*sigma^2))
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel) # Normalizzazione (fondamentale!)

def apply_convolution(image, kernel):
    """
    A. CONVOLUZIONE DISCRETA 2D
    Implementa l'operazione di filtraggio spaziale FIR.
    """
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    # Padding per gestire i bordi
    padded_img = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image, dtype=np.float32)
    
    for y in range(i_h):
        for x in range(i_w):
            # Estrazione regione e prodotto scalare (convoluzione)
            region = padded_img[y:y+k_h, x:x+k_w]
            output[y, x] = np.sum(region * kernel)
            
    return np.clip(output, 0, 255).astype(np.uint8)

def custom_median_filter(image, size=3):
    """
    B. FILTRO MEDIANO (Rank-Order Filter)
    Implementa l'ordinamento dei pixel in una finestra locale.
    """
    i_h, i_w = image.shape
    pad = size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)
    
    for y in range(i_h):
        for x in range(i_w):
            # Estrazione finestra, flattening e calcolo mediana
            window = padded_img[y:y+size, x:x+size].flatten()
            output[y, x] = np.median(window)
            
    return output

def custom_histogram_equalization(image):
    """
    C. EQUALIZZAZIONE ISTOGRAMMA (Statistical Manipulation)
    Usa la Funzione di Distribuzione Cumulativa (CDF).
    """
    # 1. Calcolo istogramma h(p)
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # 2. Calcolo Istogramma Cumulativo (CDF)
    cdf = hist.cumsum()
    
    # 3. Normalizzazione CDF (mappatura su [0, 255])
    cdf_m = np.ma.masked_equal(cdf, 0) # Maschera gli zeri
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 4. Uso della CDF come Look-Up Table (LUT)
    return cdf[image]