import skimage.morphology as morph
import matplotlib.pyplot as plt
import numpy as np
import cv2


def rangescale(frame, rescale):
    '''
    Rescale image values to be within range

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    '''
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame-np.min(frame))/np.ptp(frame))*np.ptp(rescale)+rescale[0]
    else:
        frame = np.ones_like(frame)*(rescale[0]+rescale[1])/2
    return frame


def tracking_weights(track, segall, halo_distance = 50):
    '''
    Compute weights for tracking training sets

    Parameters
    ----------
    track : 2D array
        Tracking output mask.
    segall : 2D array
        Segmentation mask of all cells in current image.
    halo_distance : int, optional
        Distance in pixels to emphasize other cells from tracked cell.
        The default is 50.

    Returns
    -------
    weights : 2D array
        Tracking weights map.

    '''

    # Cell escaped / disappeared:
    if np.max(track)==0:
        weights = ((segall>0).astype(np.uint8)*20)+1
        return weights

    # Distance from the tracked cell:
    _, dist_from = morph.medial_axis(track==0,return_distance=True)
    dist_from = halo_distance-dist_from
    dist_from[dist_from<1] = 1

    # Distance from border within cell:
    _, dist_in = morph.medial_axis(track>0,return_distance=True)

    # Tracked cell skeleton:
    skel = morph.skeletonize(track,method='lee')

    # Tracked Cell weights are distance from edges + skeleton at 255 in the center
    weights = rangescale(dist_in+1,(1,42))
    weights[skel>0] = 255

    # Rest of the image is weighed according to distance from tracked cell:
    weights+= 63*(dist_from/halo_distance) \
        *(segall>0).astype(np.float32)\
            *(track==0).astype(np.float32)

    weights*=100
    weights[dist_from<1] = 1

    return weights

# Normalize image between 0 and 255
def range_0255(mat):
    mat_01 = (mat - np.min(mat))/(np.max(mat) - np.min(mat))
    mat_0255 = mat_01*255
    return(mat_0255.astype('uint16'))

# Script to reindex cell label with integers
def reindex_cell_labels(img):
    idx = np.unique(img)
    img_c = img.copy()
    for i, x in enumerate(idx):
        img_c[img_c == x] = i
    return(img_c.astype('uint16'))

# 3x3 square kernel for morphological operations:
kernel_full33 = np.array(
    [[1,1,1],
     [1,1,1],
     [1,1,1]],
    np.uint8
    )

# DELTA script 
def labels2binary(labels):

    # Empty binary mask:
    binary = labels>0

    # Cell indexes in image:
    cell_nbs = np.unique(labels[binary])

    # Run through cells:
    for cell in cell_nbs:

        # single cell mask:
        single_cell = labels == cell
        single_cell = single_cell.astype(np.uint8)

        # Contour mask dilated by 1 pixel:
        single_cell_contour = cv2.dilate(
            single_cell,kernel_full33,iterations = 1
            )
        single_cell_contour -= single_cell

        # remove borders where cells are touching:
        binary[single_cell_contour>0]=False

    return binary.astype(np.uint8)

def kernel(n):
    '''
    

    Parameters
    ----------
    n : Int
        Determine size of kernel.

    Returns
    -------
    kernel : Array of unit8
        Returns a kernal with size of n.

    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    return kernel


def seg_weights_2D(mask, classweights=(1,1)):
    '''
    This function computes custom weightmaps designed for bacterial images where borders are difficult to distinguish
    
    Parameters
    ----------
    mask : 2D array
        Training output segmentation mask.
    classweights : tuple of 2 int/floats, optional
        Weights to apply to cells and border
        The default is (1,1)
      

    Returns
    -------
    weightmap : 2D array
        Weights map image.

    '''
    if mask.ndim == 3:
        mask = mask[:,:,0]
    if np.max(mask) == 255:
        mask = mask / 255
        
    # Extract all pixels that include the cells and it's border 
    border = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel(20)) 
    # Set all pixels that include the cells to zero to leave behind the border only
    border[mask>0] = 0   
    
    # Erode the segmentation to avoid putting high emphasiss on edges of cells 
    mask_erode = cv2.erode(mask,kernel(2))   
    
    # Get the skeleton of the segmentation and border
    mask_skel = morph.skeletonize(mask_erode > 0)  
    border_skel = morph.skeletonize(border > 0)  
    
    # Find the distances from the skeleton of the segmention and border
    s , border_dist = morph.medial_axis(border_skel < 1, return_distance=True)  
    s , mask_dist = morph.medial_axis(mask_skel < 1, return_distance=True)      
    
    # Use the distance from the skeletons to create a gradient towards the skeleton
    border_gra = border * (classweights[1]) / (border_dist + 1)**2             
    mask_gra = (mask/ (mask_dist + 1)**2)
        
    # Set up weights array
    weightmap = np.zeros((mask.shape),dtype=np.float32)
    
    # Add the gradients for the segmentation and border into the weights array
    weightmap[mask_erode>0] = mask_gra[mask_erode>0] 
    weightmap[border>0] = border_gra[border>0] 
    
    # Set the skeletons of the segmentation and borders to the maximum values
    weightmap[mask_skel>0] = classweights[0]
    weightmap[border_skel>0] = classweights[1]
        
    # Keep the background zero and set the erode values in the seg/border to a minimum of 1
    bkgd = np.ones((mask.shape)) - mask - border
    weightmap[((weightmap == 0) * (bkgd < 1))] = 1 / 255
    
    return weightmap

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
