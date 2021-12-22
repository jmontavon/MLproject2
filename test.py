import skimage.morphology as morph
import numpy as np
import cv2

# 3x3 square kernel for morphological operations:
kernel_full33 = np.array(
    [[1,1,1],
     [1,1,1],
     [1,1,1]],
    np.uint8
    )

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

