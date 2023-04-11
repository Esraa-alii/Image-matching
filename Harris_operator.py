import matplotlib.pyplot as plt
from skimage.io import imshow
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image
import time as Time

def gauss_derivative_kernels(size, sizey=None):
    """ 
    returns x and y derivatives of a 2D 
    gauss kernel array for convolutions
     """
    
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * np.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    return gx,gy

def gauss_kernel(size, sizey = None):
    """ 
    Returns a normalized 2D gauss kernel array for convolutions 
    """

    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def harris(uploaded_file, threshold, min_distance = 10):
    """
    Computes corners from a Harris response image and plots them

    Parameters-->
        filename     : Path of image file
        threshold    : (optional)Threshold for corner detection
        min_distance : (optional)Minimum number of pixels separating 
                       corners and image boundary
    Returns-->
        harris_path  : Plotted image path
    """

    start_time = Time.time()
    harris_path = "./images/output/harris_output.jpeg"
    image = np.array(Image.open(uploaded_file).convert("L"))
    

    ######################Computing Harris Operator Response##########################
    #derivatives
    gx, gy = gauss_derivative_kernels(3)
    imx = signal.convolve(image, gx, mode='same')
    imy = signal.convolve(image, gy, mode='same')
    #kernel for blurring
    gauss = gauss_kernel(3)
    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx, gauss, mode='same')
    Wxy = signal.convolve(imx*imy, gauss, mode='same')
    Wyy = signal.convolve(imy*imy, gauss, mode='same')   
    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy   
    harris_response = Wdet / Wtr

    ######################Compute Harris Operator Points##############################
    #find top corner candidates above a threshold
    corner_threshold = max(harris_response.ravel()) * threshold
    harrisim_t = (harris_response > corner_threshold) * 1    
    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harris_response[c[0]][c[1]] for c in coords]    
    #sort candidates
    index = np.argsort(candidate_values)   
    #store allowed point locations in array
    allowed_locations = np.zeros(harris_response.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1   
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0     
                      
    ###############################Plotting###########################################
    plt.figure()
    plt.gray()
    imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'r*')
    plt.axis('off')
    plt.savefig(harris_path)

    end_time = Time.time()
    final_time = end_time - start_time
    
    return harris_path, final_time