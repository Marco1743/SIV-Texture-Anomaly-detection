import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def compute_lbp(image, radius=2, n_points=16):
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp_image

def extract_glcm_features(patch): 
    if patch.dtype != np.uint8:
        patch = patch.astype(np.uint8)
    
    glcm = graycomatrix(patch, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    return np.array([contrast, homogeneity, energy, correlation])