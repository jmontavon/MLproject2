'''
This file contains functions and class definitions that are used in pipeline.py
A lot of functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

@author: jblugagne
'''
import cv2, os, re, ffmpeg, time, sys, importlib
import numpy as np
from scipy.io import savemat
import matplotlib.cm as cm
from threading import Thread
from skimage.morphology import skeletonize

#%% Importing the config.py file

def import_module_by_path(path):
    '''
    
    Import a module from a different directory than the current working directory
    
    Parameters
    ----------
    path : str
        Path to local config.py file.

    Returns
    -------
    mod : module
        Returns the config.py file as a module.

    '''
    name = os.path.splitext(os.path.basename(path))[0]
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod

# Customized config.py should be stored in your home directory within a '.delta' folder
# If there is no config.py at this location, the local config.py will be use
hd_path = os.path.expanduser('~/.delta/config.py')
if os.path.exists(hd_path):
    cfg = import_module_by_path(hd_path)
else:
    cfg = import_module_by_path('config.py')

def gen_cfg(config_file=None):
    # If we provide a custom config file, load it :
    if config_file is not None:
        cfg = import_module_by_path(config_file)
    else:
        cfg = import_module_by_path('config.py')
    return(cfg)


#%% Image correction

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


def deskew(image):
    '''
    Compute rotation angle of chambers in image for rotation correction.
    From: https://gist.github.com/panzerama/beebb12a1f9f61e1a7aa8233791bc253
    Not extensively tested. You can skip rotation correction if your chambers
    are about +/- 1 degrees of horizontal

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    rotation_number : float
        Rotation angle of the chambers for correction.

    '''
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line


    # canny edges in scikit-image
    edges = canny(image)

    # hough lines
    hough_lines = probabilistic_hough_line(edges)

    # hough lines returns a list of points, in the form ((x1, y1), (x2, y2))
    # representing line segments. the first step is to calculate the slopes of
    # these lines from their paired point values
    slopes = [(y2 - y1)/(x2 - x1) if (x2-x1) else 0 for (x1,y1), (x2, y2) in hough_lines]

    # it just so happens that this slope is also y where y = tan(theta), the angle
    # in a circle by which the line is offset
    rad_angles = [np.arctan(x) for x in slopes]

    # and we change to degrees for the rotation
    deg_angles = [np.degrees(x) for x in rad_angles]

    # which of these degree values is most common?
    histo = np.histogram(deg_angles, bins=180)
    
    # correcting for 'sideways' alignments
    rotation_number = histo[1][np.argmax(histo[0])]

    if rotation_number > 45:
        rotation_number = -(90-rotation_number)
    elif rotation_number < -45:
        rotation_number = 90 - abs(rotation_number)

    return rotation_number


def imrotate(frame, rotate):
    '''
    Rotate image

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rotate : float
        Rotation angle, in degrees.

    Returns
    -------
    2D numpy array of floats
        Rotated image

    '''
    M = cv2.getRotationMatrix2D((frame.shape[1]/2,frame.shape[0]/2), rotate, 1)
    frame = cv2.warpAffine(frame,M,(frame.shape[1],frame.shape[0]),borderMode=cv2.BORDER_REPLICATE)
    
    return frame


def driftcorr(img,template=None,box=None, drift=None):
    '''
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drfit for.
    template : None or 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        default is None.
    box : None or dictionary, optional
        A cropping box to extract the part of the frame to compute drift 
        correction over (see cropbox()).
        default is None.
    drift : None or tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        default it None.

    Returns
    -------
    2D/3D numpy array, tuple of len 2
        Drift-corrected image and drift.

    '''
    
    if len(img.shape)==2:
        twoDflag=True
        img = np.expand_dims(img,axis=0)
    else:
        twoDflag=False
        
    if drift is None:
        if template is None: # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0,0)
        template = rangescale(template,(0,255)).astype(np.uint8) # Making sure its the right format
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]],[drift[1]])
    else:
        (xcorr, ycorr) = drift
        
    for i in range(img.shape[0]):
        if drift is None:
            frame = rangescale(img[i],(0,255)).astype(np.uint8) # Making sure its the right format
            driftcorrimg = cropbox(frame,box)
            res = cv2.matchTemplate(driftcorrimg,template,cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            ycorr[i] = max_loc[0] - res.shape[1]/2
            xcorr[i] = max_loc[1] - res.shape[0]/2
        T = np.float32([[1, 0, -ycorr[i]], [0, 1, -xcorr[i]]])
        img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])
    
    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    else:
        return img, (xcorr, ycorr)


def getDriftTemplate(chamberboxes,img,whole_frame=False):
    '''
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getROIBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.
    whole_frame : bool, optional
        Whether to use the whole frame as reference instead of the area above
        the chambers.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    '''
    
    if len(chamberboxes) == 0 and not whole_frame:
        return None
    (y_cut,x_cut) = [round(i*.025) for i in img.shape] # Cutting out 2.5% of the image on eahc side as drift margin
    
    box = dict(
        xtl = x_cut,
        xbr = -x_cut,
        ytl = y_cut,
        ybr = -y_cut if whole_frame else max(chamberboxes,
                                             key=lambda elem: elem['ytl']
                                             )['ytl']-y_cut
        )
        
    return cropbox(img,box)


def opencv_areafilt(I, min_area=20, max_area=None):
    '''
    Area filtering using openCV instead of skimage

    Parameters
    ----------
    I : 2D array
        Segmentation mask.
    min_area : int or None, optional
        Minimum object area.
        The default is 20
    max_area : int or None, optional
        Maximum object area.
        The default is None.

    Returns
    -------
    I : 2D array
        Filtered mask.

    '''
    
    # Get contours:
    contours ,_ = cv2.findContours(I, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Loop through contours, flag them for deletion:
    to_remove = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (min_area is not None and area < min_area) or \
            (max_area is not None and area > max_area):
            to_remove += [cnt]
    
    # Delete all at once:
    if len(to_remove)>0:
        I = cv2.drawContours(I, to_remove, -1, 0, thickness=-1)
    
    return I


#%% Image cropping & stitching

def cropbox(img,box):
    '''
    Crop image

    Parameters
    ----------
    img : 2D numpy array
        Image to crop.
    box : Dictionary
        Dictionary describing the box to cut out, containing the following 
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    '''
    if box is None or all([v is None for v in box.values()]):
        return img
    else:
        return img[box['ytl']:box['ybr'],box['xtl']:box['xbr']]


def create_windows(img, target_size = (512,512), min_overlap = 24):
    '''
    Crop input image into windows of set size.

    Parameters
    ----------
    img : 2D array
        Input image.
    target_size : tuple, optional
        Dimensions of the windows to crop out.
        The default is (512,512).
    min_overlap : int, optional
        Minimum overlap between windows in pixels.
        Defaul is 24.
        

    Returns
    -------
    
    windows: 3D array
        Cropped out images to feed into U-Net. Dimensions are
        (nb_of_windows, target_size[0], target_size[1])
    loc_y : list
        List of lower and upper bounds for windows over the y axis
    loc_x : list
        List of lower and upper bounds for windows over the x axis

    '''
    # Make sure image is minimum shape (bigger than the target_size) 
    if img.shape[0] < target_size[0] :
        img = np.concatenate((img,np.zeros((target_size[0]-img.shape[0],img.shape[1]))),axis=0)
    if img.shape[1] < target_size[1]:
        img = np.concatenate((img,np.zeros((img.shape[0],target_size[1]-img.shape[1]))),axis=1)
    
    # Decide how many images vertically the image is split into
    ny = int(
        1+float(img.shape[0]-min_overlap)/float(target_size[0]-min_overlap)
        ) 
    nx = int(
        1+float(img.shape[1]-min_overlap)/float(target_size[1]-min_overlap)
        ) 
    # If image is 512 pixels or smaller, there is no need for anyoverlap
    if img.shape[0] == target_size[0]:
        ny = 1
    if img.shape[1] == target_size[1]:
        nx = 1
        
    # Compute y-axis indices:
    ovlp_y = -(img.shape[0]-ny*target_size[0])/(ny-1) if ny > 1 else 0
    loc_y = []
    for n in range(ny-1):
        loc_y += [
            (int(target_size[0]*n-ovlp_y*n),int(target_size[0]*(n+1)-ovlp_y*n))
            ]
    loc_y += [(img.shape[0]-target_size[0],img.shape[0])]
        
    # Compute x-axis indices:
    ovlp_x = -(img.shape[1]-nx*target_size[1])/(nx-1) if nx > 1 else 0
    loc_x = []
    for n in range(nx-1):
        loc_x += [
            (int(target_size[1]*n-ovlp_x*n),int(target_size[1]*(n+1)-ovlp_x*n))
            ]
    
    loc_x += [(img.shape[1]-target_size[1],img.shape[1])] 
        
    # Store all cropped images into one numpy array called windows
    windows = np.zeros(((nx*ny,)+target_size),dtype=img.dtype)
    for i in range(len(loc_y)):
        for j in range(len(loc_x)):
            windows[i*len(loc_x)+j,:,:] = img[
                loc_y[i][0]:loc_y[i][1],
                loc_x[j][0]:loc_x[j][1]
                ]

    return windows, loc_y, loc_x


def stitch_pic(results,loc_y,loc_x):
    '''
    Stitch segmentation back together from the windows of create_windows()

    Parameters
    ----------
    results : 3D array
        Segmentation outputs from the seg model with dimensions 
        (nb_of_windows, target_size[0], target_size[1])
    loc_y : list
        List of lower and upper bounds for windows over the y axis
    loc_x : list
        List of lower and upper bounds for windows over the x axis

    Returns
    -------
    stitch_norm : 2D array
        Stitched image.

    '''
    
    # Create an array to store segmentations into a format similar to how the image was cropped
    stitch = np.zeros((loc_y[-1][1],loc_x[-1][1]),dtype=results.dtype)
    index = 0
    y_end = 0
    for i in range(len(loc_y)):
        
        # Compute y location of window:
        y_start = y_end
        if i+1==len(loc_y):
            y_end = loc_y[i][1]
        else:
            y_end = int((loc_y[i][1]+loc_y[i+1][0])/2)
        
        x_end = 0
        for j in range(len(loc_x)):
            
            # Compute x location of window:
            x_start = x_end
            if j+1==len(loc_x):
                x_end = loc_x[j][1]
            else:
                x_end = int((loc_x[j][1]+loc_x[j+1][0])/2)
            
            # Add to array:
            res_crop_y = -(loc_y[i][1]-y_end) if loc_y[i][1]-y_end>0 else None
            res_crop_x = -(loc_x[j][1]-x_end) if loc_x[j][1]-x_end>0 else None
            stitch[y_start:y_end,x_start:x_end] = results[
                index,
                y_start-loc_y[i][0]:res_crop_y,
                x_start-loc_x[j][0]:res_crop_x
                ]
            
            index+=1

    return stitch


def gettrackingboxes(cell, frame_shape=None, target_size=cfg.target_size_track):
    '''
    Get a crop box and a fill box around a cell that fits the tracking target 
    size

    Parameters
    ----------
    cell : 2D array of uint8
        Mask of the cell to track.
    frame_shape : tuple of 2 ints or None, optional
        Original dimensions of the  image. If None, cell.shape is used.
        The default is None.
    target_size : tuple of 2 ints, optional
        Target dimensions of the cropped image. 
        The default is cfg.target_size_track.

    Returns
    -------
    cropbox : dict
        Crop box in the cropbox() input format.
        The crop box determines which part of the full-size frame to crop out.
    fillbox : dict
        Fill box in the cropbox() input format.
        The fill box determines which part of the target-size input to fill with
        the cropped out image.

    '''
    
    if frame_shape is None:
        frame_shape = cell.shape
    
    cx, cy = getcentroid(cell)
    
    xtl = int(max(cx-target_size[1]/2,0))
    xbr = int(min(cx+target_size[1]/2,frame_shape[1]))
    
    ytl = int(max(cy-target_size[0]/2,0))
    ybr = int(min(cy+target_size[0]/2,frame_shape[0]))
    
    cropbox = {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    xtl = int(max(target_size[1]/2-cx,0))
    xbr = int(min(target_size[1]/2+frame_shape[1]-cx,target_size[1]))
    
    ytl = int(max(target_size[0]/2-cy,0))
    ybr = int(min(target_size[0]/2+frame_shape[0]-cy,target_size[0]))
    
    fillbox = {'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr}
    
    return cropbox, fillbox

def getshiftvalues(shift,img_shape,cb):
    ''' 

    Parameters
    ----------
    shift : int
        Max amount of pixels cropbox to be shifted in the y / x direction.
    img_shape : tuple
        Shape of the image / input that will be cropped.
    cropbox : dict
        Crop box in the cropbox() input format.
        The crop box determines which part of the full-size frame to crop out.

    Returns
    -------
    shiftY : int
        Number of pixels to shift cropbox in the y irection
    shiftX : int
        Number of pixels to shift cropbox in the x direction.

    '''
    upperY = np.min((shift,img_shape[0] - cb['ybr']))
    lowerY = np.min((shift,cb['ytl']))
    shiftY = int(np.random.uniform(-lowerY,upperY))
    
    upperX = np.min((shift,img_shape[1] - cb['xbr']))
    lowerX = np.min((shift,cb['xtl']))
    shiftX = int(np.random.uniform(-lowerX,upperX))

    return shiftY, shiftX
#%% Masks, labels, objects identification

def getROIBoxes(chambersmask):
    '''
    Extract the bounding boxes of the chambers in the binary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of dictionaries
        List of cropping box dictionaries (see cropbox()).

    '''
    chamberboxes = []
    if chambersmask.dtype == bool:
        chambersmask = chambersmask.astype(np.uint8)
    else:
        chambersmask = cv2.threshold(chambersmask,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(chambersmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamberboxes.append(dict(
                xtl = xtl,
                ytl = int(ytl-(boxheight*.1)), # -10% of height to make sure the top isn't cropped
                xbr = xtl+boxwidth,
                ybr = ytl+boxheight)) # tl = top left, br = bottom right.
    chamberboxes.sort(key=lambda elem: elem['xtl'])# Sorting by top-left X (normally sorted by Y top left)
    return chamberboxes


def label_seg(seg,cellnumbers=None,return_contours=False,background=0):
    '''
    Label cells in segmentation mask

    Parameters
    ----------
    seg : numpy 2D array of float/uint8/uint16/bool
        Cells segmentation mask. Values >0.5 will be considered cell pixels
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is 
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    label : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels 
        with values given by cellnumbers

    '''
    if seg.dtype == bool:
        seg = seg.astype(np.uint8)
    else:
        seg = cv2.threshold(seg,.5,1,cv2.THRESH_BINARY)[1].astype(np.uint8)
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
    label = np.full(seg.shape,background,dtype=np.uint16)
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(label,[contour],c+1 if cellnumbers is None else cellnumbers[c])
    if return_contours:
        return label, contours
    else:
        return label


def getcellsinframe(labels, return_contours=False, background=0):
    '''
    Get numbers of cells present in frame, sorted along Y axis

    Parameters
    ----------
    labels : 2D numpy array of ints
        Single frame from labels stack.
    return_contours : bool, optional
        Flag to get cv2 contours.

    Returns
    -------
    cells : list
        Cell numbers (0-based indexing).
    contours : list
        List of cv2 contours for each cell. Returned if return_contours==True.

    '''
    
    cells, ind = np.unique(labels,return_index=True)
    cells = [cell-1 for _,cell in sorted(zip(ind,cells))] # Sorting along Y axis, 1-based to 0-based, & removing 1st value (background)
    cells = [cell for cell in cells if cell!=background-1] # Remove background
    
    # Get opencv contours:
    if return_contours:
        contours = []
        for c, cell in enumerate(cells):
            # Have to do it this way to avoid cases where border is shrunk out
            cnt, _ = cv2.findContours(
                (labels==cell+1).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
                )
            contours += cnt
        return cells, contours
    else:
        return cells


def getcentroid(contour):
    '''
    Get centroid of cv2 contour

    Parameters
    ----------
    contour : 3D numpy array
        Blob contour generated by cv2.findContours().

    Returns
    -------
    cx : int
        X-axis coordinate of centroid.
    cy : int
        Y-axis coordinate of centroid.

    '''
    
    if contour.shape[0]>2: # Looks like cv2.moments treats it as an image
        # Calculate moments for each contour
        M = cv2.moments(contour)
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
    else:
        cx = int(np.mean(contour[:,:,0]))
        cy = int(np.mean(contour[:,:,1]))
        
    return cx,cy


#%% Poles

def getpoles(seg, labels=None, scaling=None):
    '''
    Get cell poles

    Parameters
    ----------
    seg : 2D array of uint8
        Cell segmentation mask.
    labels : 2D array of int, optional
        Cell labels map. If None, label_seg() will be applied to seg
        The default is None.

    Returns
    -------
    poles : list
        List of poles per cell. Each cell in the list has exactly 2 poles.

    '''
    
    # No label provided:
    if labels is None:
        labels = label_seg(seg)
    
    # Get poles using skeleton method:
    skel = skeletonize(seg, method='lee')
    ends_map = skeleton_poles(skel)
    poles = extract_poles(ends_map, labels)
    
    # Make sure cells have 2 poles each:
    for p in range(len(poles)):
        if len(poles[p])>2:
            poles[p] = two_poles(poles[p])
        if len(poles[p])<2: # Sometimes skeletonize fails
            poles[p] = extrema_poles(labels==p+1, scaling=scaling)
    
    # Apply scaling:
    if scaling:
        for c in poles:
            for p in c:
                p[:] = [int(yx*scale) for yx, scale in zip(p,scaling)]
    
    return poles


def skeleton_poles(skel):
    """
    This function was adapted from stackoverflow
    #https://stackoverflow.com/questions/26537313/how-can-i-find-endpoints-of-binary-skeleton-image-in-opencv

    It uses a kernel to filter out the poles

    Parameters
    ----------
    skel : 2D numpy array of bool
        Contains skeletons of single cells from the segmentation

    Returns
    -------
    out : 2D numpy array 
        Contains the poles of the skeletons from the input (skel).
        
    """

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the poles
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 1
    
    return out


def extrema_poles(cellmask, scaling=None):
    '''
    A slower and more rudimentary version of poles finding but that is 
    guaranteed to find exactly 2 poles

    Parameters
    ----------
    cellmask : 2D array of int
        Segmentation mask of a single cell.

    Returns
    -------
    poles : list of 2 1D arrays
        List of two poles, each being a 2-element array of y-x coordinates.

    '''
    
    scaling = scaling or (1,1)
    
    y,x = cellmask.nonzero()
    if np.ptp(y)*scaling[0]>np.ptp(x)*scaling[1]: # Vertical cell
        i = np.lexsort((x,y))
    else: # Horizontal cell
        i = np.lexsort((y,x))    
    poles = [np.array((y[i[0]],x[i[0]])),np.array((y[i[-1]],x[i[-1]]))]
    
    return poles


def two_poles(poles):
    '''
    Sometimes the skeleton produces more than 2 poles.
    Thus function selects only 2 poles in the skeleton.

    Parameters
    ----------
    poles : list
        Coordinates for the mother/daughter poles for a single cell
        Dimension is (#_of_poles, [y_coordiante, x_coordinate])
        

    Returns
    -------
    poles : List of 2 1D array
        List of two poles, each being a 2-element array of y-x coordinates.

    '''

    dist = 0
    # Measure the distance from the first endpoint to all the other poles
    for i in range(1,len(poles)):
        d = eucl(poles[0],poles[i])
        if d > dist:
            dist = d
            i_max = i
    
    # Redo the same thing starting from the furthest endpoint:
    dist = 0
    for j in range(len(poles)):
        if j != i_max: # skip same point
            d = eucl(poles[i_max],poles[j])
            if d > dist:
                dist = d
                j_max = j

    return [poles[i_max],poles[j_max]]
    
def extract_poles(end_img, labels):
    '''
    Extract poles per cell from ends image

    Parameters
    ----------
    end_img : 2D array of bool
        'Mask' of the poles in the image.
    labels : 2D array of int
        Cell labels map.

    Returns
    -------
    poles : list of tuples of 2 1D numpy arrays
        For each cell, tuple of 2 numpy arrays with the y-x coordinates 
        of the poles.

    '''
    locations = end_img.nonzero()
    poles = [[] for _ in range(np.max(labels))]
    for p in range(len(locations[0])):
        
        cell = labels[locations[0][p],locations[1][p]]-1
        poles[cell]+=[np.array([locations[0][p],locations[1][p]])]
    
    return poles


def eucl(p1,p2):
    '''
    Euclidean point to point distance

    Parameters
    ----------
    p1 : 1D array
        Coordinates of first point.
    p2 : 1D array
        Coordinates of second point.

    Returns
    -------
    float
        Euclidean distance between p1 and p2.

    '''
    return np.linalg.norm(p1-p2)


#%% Lineage

def getTrackingScores(labels, outputs, boxes=None):
    '''
    Get overlap scores between input/target cells and tracking outputs

    Parameters
    ----------
    inputs : 2D array of floats
        Segmentation mask of input/target cells that the tracking U-Net is 
        tracking against. (ie segmentation mask of the 'current'/'new' frame)
    outputs : 4D array of floats
        Tracking U-Net output.

    Returns
    -------
    scores : 2D array of floats
        Overlap scores matrix between tracking predictions and current 
        segmentation mask for each new-old cell.

    '''

    total_cell = np.max(labels)
    if total_cell == 0:
        return None
    
    # Get areas for each cell:
    _, areas = np.unique(labels, return_counts=True)
    areas = areas[1:]
    
    # Compile scores:
    scores = np.zeros([outputs.shape[0],total_cell],dtype=np.float32)
    for o, output in enumerate(outputs):
        
        # Find pixels with a score > .05:
        nz = list((output>.05).nonzero())
        if len(nz[0])==0:
            continue
        
        # Find cells that are "hit" by those pixels in the labels image:
        if boxes is None or\
            boxes[o] is None or\
                all([v is None for v in boxes[o][0].values()]):
            cells, counts = np.unique(labels[tuple(nz)], return_counts=True)
            
        else:
            # Clean nz hits outside of fillbox:
            cb, fb = boxes[o][:]
            to_remove = np.logical_or.reduce(
                (
                    nz[0]<=fb['ytl'],
                    nz[0]>=fb['ybr'],
                    nz[1]<=fb['xtl'],
                    nz[1]>=fb['xbr']
                    )
                )
            nz[0] = np.delete(nz[0],to_remove.nonzero())
            nz[1] = np.delete(nz[1],to_remove.nonzero())
            
            # Offset hits by cropbox-fillbox:
            nz[0] = nz[0]+cb['ytl']-fb['ytl']
            nz[1] = nz[1]+cb['xtl']-fb['xtl']
            
            # Compute number of hits per cell:
            cells, counts = np.unique(labels[nz[0],nz[1]], return_counts=True)
        
        # Compile score for these cells:
        for c, cell in enumerate(cells):
            if cell > 0:
                scores[o,cell-1]=counts[c]/areas[cell-1]
    
    return scores


def getAttributions(scores):
    '''
    Get attribution matrix from tracking scores

    Parameters
    ----------
    scores : 2D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : 2D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each old cell can be attributed to 1 
        or 2 new cells.

    '''
    
    attrib = np.zeros(scores.shape, dtype=int)
    
    for n in range(scores.shape[1]):
        worst_old = np.argsort(scores[:,n]) # Worst-to-best score of old cells for n-th new cell
        # Run through old cells from best to worst:
        for o in range(-1,-scores.shape[0]-1,-1):
            o_best = worst_old[o] # o-th best old cell 
            s = scores[o_best,n]
            # If score gets too low, stop:
            if s < .2:
                break
            # Check if new cell is at least 2nd best for this old cell:
            worst_new = np.argsort(scores[o_best,:]) # Worst-to-best score of new cells for o_best old cell
            if n in worst_new[-2:]:
                attrib[o_best,n]=1
                break
                
    return attrib


class Lineage:
    '''
    Class for cell lineages contained in each ROI
    '''
    
    def __init__(self):
        '''
        Initialize object.

        Returns
        -------
        None.

        '''
        self.cells = []
        self.stack = []
        self.cellnumbers = []
    
    def update(self, cell, frame, attrib=[], poles=[]):
        '''
        Attribute cell from previous frame to cell(s) in new frame.

        Parameters
        ----------
        cell : int or None
            Cell index in previous frame to update. If None, the cells in the
            new frame are treated as orphans.
        frame : int
            Frame / timepoint number.
        attrib : list of int, optional
            Cell index(es) in current frame to attribute the previous cell to.
            The default is [].
        poles : list of lists of 1D numpy arrays, optional
            List of poles for the cells in the new frame.
            The default is [].

        Returns
        -------
        None.

        '''
        
        # Check cell numbers list:
        if len(self.cellnumbers)<=frame:
            self.cellnumbers+=[[] for _ in range(frame+1-len(self.cellnumbers))] # Initialize
        if len(attrib)>0 and len(self.cellnumbers[frame])<=max(attrib):
            self.cellnumbers[frame]+=[-1 for _ in range(max(attrib)+1-len(self.cellnumbers[frame]))] # Initialize
        
        # If no attribution: (new/orphan cell)
        if cell is None and len(attrib)>0:
            poles[0].sort(key=lambda p: p[0], reverse=True)
            cellnum = self.createcell(frame,new_pole=poles[0][0],old_pole=poles[0][1])
            self.cellnumbers[frame][attrib[0]]=cellnum
        
        # Simple tracking event:
        elif len(attrib)==1:
            cellnum = self.cellnumbers[frame-1][cell] # Get old cell number
            self.cellnumbers[frame][attrib[0]]=cellnum
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(cellnum, frame-1, 'old_pole')
            prev_new = self.getvalue(cellnum, frame-1, 'new_pole')
            # Find which pole is new vs old:
            if eucl(poles[0][0],prev_old)+eucl(poles[0][1],prev_new)\
                < eucl(poles[0][0],prev_new)+eucl(poles[0][1],prev_old):
                self.updatecell(
                    cellnum,frame,old_pole=poles[0][0],new_pole=poles[0][1]
                    )
            else:
                self.updatecell(
                    cellnum,frame,old_pole=poles[0][1],new_pole=poles[0][0]
                    )
            
        
        # Division event:
        elif len(attrib)==2:
            mothernum = self.cellnumbers[frame-1][cell] # Get old cell number
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(mothernum, frame-1, 'old_pole')
            prev_new = self.getvalue(mothernum, frame-1, 'new_pole')
            # Find new new poles (2 closest of the poles of the new cells):
            min_dist = np.inf
            for c1 in range(2):
                for c2 in range(2):
                    dist = eucl(poles[0][c1],poles[1][c2])
                    if dist<min_dist:
                        min_dist=dist
                        c1_new = c1
                        c2_new = c2
            # Find poles closest to old and new pole from previous cell:
            if eucl(poles[0][int(not c1_new)],prev_old) \
                + eucl(poles[1][int(not c2_new)],prev_new) \
                    < eucl(poles[0][int(not c1_new)],prev_new) \
                        + eucl(poles[1][int(not c2_new)],prev_old):
                # cell 0 is mother, cell 1 is daughter
                # Attribute daughter:
                daughternum = self.createcell(
                    frame,
                    new_pole=poles[1][c2_new],
                    old_pole=poles[1][int(not c2_new)],
                    mother=mothernum
                    )
                self.cellnumbers[frame][attrib[1]]=daughternum
                #Attribute mother:
                self.updatecell(
                    mothernum,
                    frame,
                    new_pole=poles[0][c1_new],
                    old_pole=poles[0][int(not c1_new)],
                    daughter=daughternum
                    )
                self.cellnumbers[frame][attrib[0]]=mothernum
            else:
                # cell 1 is mother, cell 0 is daughter
                # Attribute daughter:
                daughternum = self.createcell(
                    frame,
                    new_pole=poles[0][c1_new],
                    old_pole=poles[0][int(not c1_new)],
                    mother=mothernum
                    )
                self.cellnumbers[frame][attrib[0]]=daughternum
                #Attribute mother:
                self.updatecell(
                    mothernum,
                    frame, 
                    new_pole=poles[1][c2_new], 
                    old_pole=poles[1][int(not c2_new)],
                    daughter=daughternum
                    )
                self.cellnumbers[frame][attrib[1]]=mothernum
            
            
    def createcell(self, frame, new_pole=None, old_pole=None, mother=None):
        '''
        Create cell to append to lineage list
    
        Parameters
        ----------
        frame : int
            Frame that the cell first appears (1-based indexing).
        mother : int or None, optional
            Number of the mother cell in the lineage (1-based indexing).
            The default is None. (ie unknown mother)
    
        Returns
        -------
        int
            new cell number
    
        '''
        
        new_cell = {
            'id':len(self.cells),
            'mother':mother,
            'frames':[frame],
            'daughters':[None],
            'new_pole': [new_pole],
            'old_pole': [old_pole]
            }
        self.cells.append(new_cell)
        
        return new_cell['id']
    
    def updatecell(
            self, cell, frame, daughter=None, new_pole=None, old_pole=None
            ):
        '''
        Update cell lineage values

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        daughter : int or None, optional
            Daughter cell number if division just happened.
            The default is None.
        new_pole : list, optional
            New pole location. The default is None.
        old_pole : list, optional
            Old pole location. The default is None.

        Returns
        -------
        None.

        '''
        
        self.cells[cell]['frames']+=[frame] # Update frame number list
        self.cells[cell]['daughters']+=[daughter]
        self.cells[cell]['old_pole']+=[old_pole]
        self.cells[cell]['new_pole']+=[new_pole]
    
    def setvalue(self,cell,frame,feature,value):
        '''
        Set feature value for specific cell and frame/timepoint

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to set value for.
        value : int, float, str
            Value to assign.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        None.

        '''
        
        try:
            i = self.cells[cell]['frames'].index(frame)
        except ValueError:
            raise ValueError('Cell %d is not present in frame %f'%(cell,frame))
        
        # Get cell dict
        cell = self.cells[cell]
        
        # If feature doesn't exist yet, create list:
        if feature not in cell:
            cell[feature]=[None for _ in range(len(cell['frames']))]
        
        # Add value:
        cell[feature][i]=value
    
    def getvalue(self,cell,frame,feature):
        '''
        Get feature value for specific timepoint/frame

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to get value for.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        int, float, str
            Value for feature at frame.

        '''
        
        try:
            i = self.cells[cell]['frames'].index(frame)
        except ValueError:
            raise ValueError('Cell %d is not present in frame %d'%(cell,frame))
        
        return self.cells[cell][feature][i]


#%% Image files

def getxppathdialog(ask_folder=False):
    '''
    Pop up window to select experiment file or folder.

    Parameters
    ----------
    ask_folder : bool, optional
        Folder selection window will pop up instead of file selection.
        The default is False.

    Returns
    -------
    file_path : str
        Path to experiment file or folder.

    '''
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    # root.withdraw() # For some reason this doesn't work with askdirectory?
    if ask_folder:
        file_path = filedialog.askdirectory(title='Please select experiment folder(s)',
                                              mustexist=True)
        
    else:
        file_path = filedialog.askopenfilename(title='Please select experiment files(s)')
    
    root.destroy()
        
    return file_path


class xpreader:
    
    def __init__(
            self, 
            filename=None,
            channelnames=None,
            use_bioformats=False,
            prototype=None,
            fileorder='pct',
            filenamesindexing=1,
            watchfiles = False,
            ):
        '''
        Initialize experiment reader

        Parameters
        ----------
        filename : String or None
            Path to experiment file or directory. If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If None, an interactive selection dialog
            will be used. If no prototype is provided, the filenames are
            expected to be of the following C-style format: 
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded 
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
                Pos01_Cha3_Fra0005.tif
                p3c1t034.tif
                xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager 
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same 
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing = int
            Selects between 0-based or 1-based indexing in the filename. If 
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1
            

        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the 
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        '''

        # Set default parameters
        self.filename = filename
        self.use_bioformats = use_bioformats
        self.fileorder = fileorder
        self.filenamesindexing = filenamesindexing
        self.prototype = prototype
        self.resfolder = None
        
        # Retrieve command line arguments (if any)
        cmdln_arguments = sys.argv
        
        if len(cmdln_arguments) >= 2: 
            # If command line arguments were passed
            self._command_line_init(cmdln_arguments)
        elif filename is None:
            # Interactive selection and pop up window:
            self._interactive_init()
        
        self.channelnames = channelnames
        self.watchfiles = watchfiles
        
        _, file_extension = os.path.splitext(self.filename)
        
        if self.use_bioformats:
            import bioformats
            import javabridge
            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(self.filename)
            md = bioformats.OMEXML(
                bioformats.get_omexml_metadata(path=self.filename)
                )
            self.positions = md.get_image_count()
            self.timepoints = md.image(0).Pixels.SizeT # Here I'm going to assume all series have the same number of timepoints
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=False,c=0).dtype
        
        elif os.path.isdir(self.filename): # Experiment is stored as individual image TIFF files in a folder
            self.filetype = 'dir'
            self.filehandle = self.filename
            # If filename prototype is not provided, guess it from the first file:
            if self.prototype is None:
                imgfiles = [x for x in os.listdir(self.filename) if os.path.splitext(x)[1].lower() in ('.tif','.tiff')]
                # Here we assume all images in the folder follow the same naming convention:
                numstrs = re.findall("\d+", imgfiles[0]) # Get digits sequences in first filename
                charstrs = re.findall("\D+", imgfiles[0]) # Get character sequences in first filename
                if len(numstrs)!=3 or len(charstrs)!=4:
                    raise ValueError('Filename formatting error. See documentation for image sequence formatting')
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype =    charstrs[0]+'%0'+str(len(numstrs[0]))+'d'+\
                                    charstrs[1]+'%0'+str(len(numstrs[1]))+'d'+\
                                    charstrs[2]+'%0'+str(len(numstrs[2]))+'d'+\
                                    charstrs[3]
            # Get experiment settings by testing if relevant files exist: 
            # Get number of positions:
            if 'p' in self.fileorder:
                self.positions = 0
                while(os.path.exists(self.getfilenamefromprototype(self.positions,0,0))): self.positions+=1
            else:
                self.positions = 1
            # Get number of channels:
            if 'c' in self.fileorder:
                self.channels = 0
                while(os.path.exists(self.getfilenamefromprototype(0,self.channels,0))): self.channels+=1
            else:
                self.channels = 1
            # Get number of frames/timepoints:
            if 't' in self.fileorder:
                self.timepoints = 0
                while(os.path.exists(self.getfilenamefromprototype(0,0,self.timepoints))): self.timepoints+=1
            else:
                self.timepoints = 1 # I guess this shouldn't really happen
            # Get image specs:
            if self.watchfiles:
                # Start file watcher thread:
                self.watcher = files_watcher(self)
                self.watcher.start()
            else:
                # Load first image, get image data from it
                I = cv2.imread(self.getfilenamefromprototype(0,0,0),cv2.IMREAD_ANYDEPTH)
                self.x = I.shape[1]
                self.y = I.shape[0]
                self.dtype = I.dtype
            
        elif (file_extension.lower() == '.tif' or file_extension.lower() == '.tiff'): # Works with single-series tif & mutli-series ome.tif
            from skimage.external.tifffile import TiffFile
            self.filetype = 'tif'
            self.filehandle = TiffFile(self.filename)
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[0] # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find('T')]
            self.channels = s.shape[s.axes.find('C')]
            self.x = s.shape[s.axes.find('X')]
            self.y = s.shape[s.axes.find('Y')]
            self.dtype = s.pages[0].asarray().dtype
    
    def _command_line_init(self, cmdln_arguments):
        '''
        Initialization routine if command line arguments were passed

        Parameters
        ----------
        cmdln_arguments : list of str
            List of command line arguments that were passed.

        Returns
        -------
        None.

        '''

        self.filename = cmdln_arguments[1]
        
        if len(cmdln_arguments) > 2 and cmdln_arguments[2][0] != '-':
            self.resfolder = cmdln_arguments[2]
            i = 3 # i =3 means there was a results folder specified
        else:
            i = 2 # i = 2 means there was no results folder specified
            
        if importlib.util.find_spec("bioformats"):   
            import bioformats 
            self.use_bioformats = True if os.path.splitext(self.filename)[-1][1:] in bioformats.READABLE_FORMATS else False

        while i<len(cmdln_arguments):
            if cmdln_arguments[i]=='--bio-formats':
                self.use_bioformats = bool(int(cmdln_arguments[i+1]))
                i+=2
            if cmdln_arguments[i]=='--order':
                self.fileorder = cmdln_arguments[i+1]
                i+=2
            if cmdln_arguments[i]=='--index':
                self.filenamesindexing = int(cmdln_arguments[i+1])
                i+=2
            if cmdln_arguments[i]=='--proto':
                self.prototype = cmdln_arguments[i+1]
                i+=2
                
    def _interactive_init(self):
        '''
        Interactive initialization routine.

        Raises
        ------
        ValueError
            If a non-valid experiment type was passed.

        Returns
        -------
        None.

        '''
        
        # Get xp settings:
        print(('Experiment type?\n'
              '1 - Bio-Formats compatible (.nd2, .oib, .czi, .ome.tif...)\n'
              '2 - bioformats2sequence (folder)\n'
              '3 - micromanager (folder)\n'
              '4 - high-throughput (folder)\n'
              '0 - other (folder)\n'
              'Enter a number: '),end='')
        answer = int(input())
        print()
        
        # If bioformats file(s):
        if answer is None or answer == 1:
            print('Please select experiment file(s)...')
            self.filename = getxppathdialog(ask_folder=False)
            self.use_bioformats = True
            self.prototype = None
            self.filenamesindexing = 1
            self.fileorder = 'pct'

        # If folder:
        else:
            print('Please select experiment folder...')
            self.filename = getxppathdialog(ask_folder=True)
            self.use_bioformats = False
            if answer is None or answer == 2:
                self.prototype = None
                self.fileorder = 'pct'
                self.filenamesindexing=1
            elif answer == 3:
                self.prototype = 'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
                self.fileorder = 'pcpt'
                self.filenamesindexing=0
            elif answer == 4:
                self.prototype = 'chan%02d_img/Position%06d_Frame%06d.tif'
                self.fileorder = 'cpt'
                self.filenamesindexing=1
            elif answer == 0:
                print('Enter files prototype: ', end='')
                self.prototype = input()
                print()
                print('Enter files order: ', end='')
                self.fileorder = input()
                print()
                print('Enter files indexing: ', end='')
                self.filenamesindexing = int(input())
                print()
            else:
                raise ValueError('Invalid experiment type')
            print()
                
    def close(self):
        # Close bioformats or tif reader
        if self.use_bioformats:
            self.filehandle.close()
            import javabridge
            javabridge.kill_vm()
        elif self.filetype == 'tif': # Nothing to do if sequence directory
            self.filehandle.close()
            
    def getfilenamefromprototype(self,position,channel,frame):
        '''
        Generate full filename for specific frame based on file path, 
        prototype, fileorder, and filenamesindexing

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        '''
        filenumbers = []
        
        for i in self.fileorder:
            if i=='p':
                filenumbers.append(position+self.filenamesindexing)
            if i=='c':
                filenumbers.append(channel+self.filenamesindexing)
            if i=='t':
                filenumbers.append(frame+self.filenamesindexing)
        return os.path.join(self.filehandle,self.prototype % tuple(filenumbers))
        
        
        
    def getframes(self, positions=None, channels=None, frames=None,
                  squeeze_dimensions=True, 
                  resize=None, 
                  rescale=None,
                  globalrescale=None,
                  rotate=None):
        '''
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer 
            or a tuple/list will be returned. If None is passed, all positions 
            are returned. 
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer 
            or a tuple/list will be returned. If the channel names have been 
            defined, the channel(s) can be passed as a string or tuple/list of 
            strings. If an empty list is passed, None is returned. If None is 
            passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list 
            will be returned. If None is passed, all frames are returned. If -1
            is passed and the file watcher is activated, only new frames are
            read. Works only for one position and channel at a time.
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        globalrescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in all frames to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        '''
        
        # Handle options:
        if positions is None: # No positions specified: load all
            positions = list(range(self.positions))
        elif type(positions) is not list and type(positions) is not tuple:
            positions = [positions]
          
        if channels is None: # No channel specified: load all
            channels = list(range(self.channels))
        elif type(channels) is str: # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError('Set channel names first')
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels+' is not a valid channel name.')
        elif (type(channels) is list or type(channels) is tuple):
            # If list of ints, nothing to do
            if len(channels)==0: # Empty list of channels: Return None
                return None
            elif all([type(c) is str for c in channels]): # refer to channels by list/tuple of names:
                if self.channelnames is None:
                    raise ValueError('Set channel names first')
                for i, c in enumerate(channels):
                    if c in self.channelnames:
                        channels[i] = self.channelnames.index(c)
                    else:
                        raise ValueError(c+' is not a valid channel name.')   
        elif type(channels) is not list and type(channels) is not tuple:
            channels = [channels]
        
        if frames is None: # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            if frames == -1: # Read new frames from files watcher:
                if self.watchfiles:
                    if len(positions) > 1 or len(channels) > 1:
                        raise ValueError('Can not load latest frames for more than one position/channel')
                    new = self.watcher.new[positions[0]][channels[0]]
                    frames = list(
                        range(
                            self.watcher.old[positions[0]][channels[0]]+1,
                            new+1
                            )
                        )
                else:
                    frames = list(range(self.timepoints))
            else:
                frames = [frames]
        
        # If files watcher, update the old frames array:
        if self.watchfiles:
            for p in positions:
                for c in channels:
                    self.watcher.old[p][c] = frames[-1]
            
        # Allocate memory:
        if rescale is None:
            dt = self.dtype
        else:
            dt = np.float32
        if resize is None:
            output = np.empty([len(positions),len(frames),len(channels),self.y,self.x],dtype=dt)
        else:
            output = np.empty([len(positions),len(frames),len(channels),resize[0],resize[1]],dtype=dt)
        
        # Load images:
        for p, pos in enumerate(positions):
            for c, cha in enumerate(channels):
                for f, fra in enumerate(frames):
                    # Read frame:
                    if self.use_bioformats:
                        frame = self.filehandle.read(series=pos,c=cha,t=fra,rescale=False)
                    elif self.filetype == 'dir':
                        frame = cv2.imread(self.getfilenamefromprototype(pos, cha, fra),cv2.IMREAD_ANYDEPTH)
                    elif self.filetype == 'tif':
                        frame = self.filehandle.series[pos].\
                                                    pages[fra*self.channels+cha].\
                                                    asarray()
                    # Optionally resize and rescale:
                    if rotate is not None:
                        frame = imrotate(frame, rotate)
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1]) #cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame,rescale)
                    # Add to output array:
                    output[p,f,c,:,:] = frame
                    
        # Rescale all images:
        if globalrescale is not None:
            output = rangescale(output,globalrescale)

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output


class files_watcher(Thread):
    '''
    Daemon to watch experiment files and signal new ones
    '''
    
    def __init__(self, reader):
        super().__init__()
        self.daemon = True
        self.reader = reader
        self.old = []
        self.new = []
    
    def run(self):
        while(True):
            
            # Run through monitored positions:
            for p in range(self.reader.positions):
                
                if len(self.old) <= p: # Position added / not done yet
                    # Append empty lists to the end:
                    self.old+=[[] for _ in range(p+1-len(self.old))]
                    self.new+=[[] for _ in range(p+1-len(self.new))]
                
                # Run through monitored channels:
                for channel in range(self.reader.channels):
                    
                    if len(self.old[p]) <= channel:
                        # Append empty lists to the end:
                        self.old[p]+=[
                            -1 for _ in range(channel+1-len(self.old[p]))
                            ]
                        self.new[p]+=[-1]*(channel+1-len(self.new[p]))
                    
                    # Check if new files have been written:
                    i = self.new[p][channel]
                    while(
                            os.path.exists(
                                self.reader.getfilenamefromprototype(
                                    p,
                                    channel,
                                    i+1
                                    )
                                )
                            ): i+=1
                    
                    # Store new timepoint number:
                    self.new[p][channel] = i
            time.sleep(.01)
            
    def newfiles(self):
        '''
        Get list of new files position and channel

        Returns
        -------
        newfiles : list of tuples
            List containing all new files position and channel.

        '''
        
        newfiles = []
        for p, pos_old in enumerate(self.old):
            for c, latest_read in enumerate(pos_old):
                if latest_read < self.new[p][c]:
                    newfiles += [(p,c)]
        return newfiles


#%% Saving & Loading results

def loadmodels(toload=cfg.models):
    '''
    Load models (as specified in config.py)

    Parameters
    ----------
    toload : list of str, optional
        Which of the 3 models to load. 
        The default is ['rois','segmentation','tracking'].

    Returns
    -------
    models : dict
        Dictionary containing the models specified.

    '''
    
    
    from delta.model import unet_rois, unet_seg, unet_track
    models = dict()
    
    if 'rois' in toload:
        models['rois'] = unet_rois(input_size = cfg.target_size_rois + (1,))
        models['rois'].load_weights(cfg.model_file_rois)
    
    if 'segmentation' in toload:
        models['segmentation'] = unet_seg(input_size = cfg.target_size_seg + (1,))
        models['segmentation'].load_weights(cfg.model_file_seg)
    
    if 'tracking' in toload:
        models['tracking'] = unet_track(input_size = cfg.target_size_track + (4,))
        models['tracking'].load_weights(cfg.model_file_track)
    
    return models


def legacysave(position, res_file):
    '''
    Save pipeline data in the legacy Matlab format

    Parameters
    ----------
    position : pipeline.Position object
        Position object to save data for.
    res_file : str
        Path to save file.

    Returns
    -------
    None.

    '''
    
    # File reader info
    moviedimensions = [
                position.reader.y,
                position.reader.x,
                position.reader.channels,
                position.reader.timepoints
                ]
    xpfile = position.reader.filename
    
    # If No ROIs detected for position:
    if len(position.rois)==0:
        savemat(
            res_file,
            {'res': [],
             'tiffile': xpfile,
             'moviedimensions': moviedimensions,
             'proc': {'rotation': position.rotate,
                      'chambers': [],
                      'XYdrift': []}
             }
            )
        return
    
    # Initialize data structure/dict:
    data = dict(moviedimensions=moviedimensions,tifffile=xpfile)
    
    # Proc dict/structure:
    data['proc'] = dict(
        rotation=position.rotate,
        XYdrift=np.array(position.drift_values, dtype=np.float64),
        chambers=np.array(
            [[r.box['xtl'],
              r.box['ytl'],
              r.box['xbr']-r.box['xtl'],
              r.box['ybr']-r.box['ytl']] for r in position.rois],
            dtype=np.float64)
        )
    
    # Lineages:
    data['res'] = []
    for r in position.rois:
        res = dict()
        
        # Resized labels stack: (ie original ROI size)
        res['labelsstack_resized'] = np.array(r.label_stack,dtype=np.uint16)
        
        # Not resized stack: (ie U-Net seg target size)
        label_stack = []
        for f, cellnbs in enumerate(r.lineage.cellnumbers):
            label_stack += [label_seg(r.seg_stack[f],[c+1 for c in cellnbs])]
        res['labelsstack'] = np.array(label_stack,dtype=np.uint16)
        
        # Run through cells, update to 1-based indexing
        cells = r.lineage.cells
        lin = []
        for c in cells:
            lin += [dict()]
            # Base lineage:
            lin[-1]['mothernb'] = c['mother']+1 if c['mother'] is not None else 0
            lin[-1]['framenbs'] = np.array(c['frames'],dtype=np.float32)+1
            lin[-1]['daughters'] = np.array(c['daughters'],dtype=np.float32)+1
            lin[-1]['daughters'][np.isnan(lin[-1]['daughters'])]=0
            if 'edges' in c:
                lin[-1]['edges'] = c['edges']
            # Morphological features:
            if 'area' in c:
                lin[-1]['area'] = np.array(c['area'],dtype=np.float32)
            if 'width' in c:
                lin[-1]['width'] = np.array(c['width'],dtype=np.float32)
            if 'length' in c:
                lin[-1]['length'] = np.array(c['length'],dtype=np.float32)
            if 'perimeter' in c:
                lin[-1]['perimeter'] = np.array(c['perimeter'],dtype=np.float32)
            # Loop through potential fluo channels:
            fluo = 0
            while True:
                fluo+=1
                fluostr = 'fluo%d'%fluo
                if fluostr in c:
                    lin[-1][fluostr] = np.array(c[fluostr],dtype=np.float32)
                else:
                    break
        # Store into res dict:
        res['lineage'] = lin
        
        # Store into data structure:
        data['res'] += [res]
    
    # Finally, save to disk:
    savemat(res_file, data)
        

def results_movie(position, frames=None):
    '''
    Generate movie illustrating segmentation and tracking

    Parameters
    ----------
    position : pipeline.Position object
        Position object to save data for.
    frames : list of int or None, optional
        Frames to generate the movie for. If None, all frames are run. 
        The default is None.

    Returns
    -------
    movie : TYPE
        DESCRIPTION.

    '''

    # Re-read trans frames:
    trans_frames = position.reader.getframes(
        positions=position.position_nb,
        channels=0,
        frames=frames,
        rescale=(0, 1),
        squeeze_dimensions=False,
        rotate=position.rotate
        )
    trans_frames = trans_frames[0,:,0]
    if position.drift_correction:
        trans_frames, _ = driftcorr(
            trans_frames, drift=position.drift_values
            )
    movie = []
    
    # Run through frames, compile movie:
    for f, fnb in enumerate(frames):
        
        frame = trans_frames[f]
        
        #RGB-ify:
        frame = np.repeat(frame[:,:,np.newaxis],3,axis=-1)
        
        # Add frame number text:
        frame = cv2.putText(
            frame,
            text='frame %06d' %(fnb,),
            org=(int(frame.shape[0]*.05),int(frame.shape[0]*.97)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(1, 1, 1, 1),
            thickness=2
            )
        
        for r, roi in enumerate(position.rois):
                
                # Get chamber-specific variables:
                colors = getrandomcolors(len(roi.lineage.cells), seed=r)
                cells, contours = getcellsinframe(
                    roi.label_stack[fnb],
                    return_contours=True
                    )
                
                if roi.box is None:
                    xtl, ytl = (0, 0)
                else:
                    xtl, ytl = (roi.box['xtl'], roi.box['ytl'])
                
                # Run through cells in labelled frame:
                for c, cell in enumerate(cells):
                    
                    # Draw contours:
                    frame = cv2.drawContours(
                        frame,
                        contours,
                        c,
                        color=colors[cell],
                        thickness=1,
                        offset=(xtl,ytl)
                        )
                    
                    # Draw poles:
                    oldpole = roi.lineage.getvalue(cell,fnb,'old_pole')
                    frame = cv2.drawMarker(
                        frame, 
                        (oldpole[1]+xtl,oldpole[0]+ytl),
                        color=colors[cell],
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=3,
                        thickness=1
                        )
                    
                    daughter = roi.lineage.getvalue(cell,fnb,'daughters')
                    bornago = roi.lineage.cells[cell]['frames'].index(fnb)
                    mother = roi.lineage.cells[cell]['mother']
                    
                    if daughter is None and (bornago>0 or mother is None):
                        newpole = roi.lineage.getvalue(cell,fnb,'new_pole')
                        frame = cv2.drawMarker(
                            frame,
                            (newpole[1]+xtl,newpole[0]+ytl),
                            color=[1,1,1], 
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=3,
                            thickness=1
                            )
                    
                    # Plot division arrow:
                    if daughter is not None:
                        
                        newpole = roi.lineage.getvalue(cell,fnb,'new_pole')
                        daupole = roi.lineage.getvalue(daughter,fnb,'new_pole')
                        # Plot arrow:
                        frame = cv2.arrowedLine(
                            frame,
                            (newpole[1]+xtl,newpole[0]+ytl),
                            (daupole[1]+xtl,daupole[0]+ytl),
                            color=(1,1,1),
                            thickness=1
                            )
        
        # Add to movie array:
        movie+= [(frame*255).astype(np.uint8)]
    
    return movie


def getrandomcolors(num_colors, seed=0):
    '''
    Pseudo-randomly generate list of random hsv colors.

    Parameters
    ----------
    num_colors : int
        Number of desired colors.
    seed : None or int, optional
        Seed to use with numpy.random.seed(). 
        The default is 0.

    Returns
    -------
    colors : list
        List of RGB values (0-1 interval).

    '''
    
    # Get colors:
    cmap = cm.get_cmap('hsv', lut=num_colors)
    colors = [cmap(i)[0:3] for i in range(num_colors)]
    
    # Pseudo randomly shuffle colors:
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(colors)
    
    return colors


def vidwrite(images, filename, crf=20, verbose=1):
    '''
    Write images stack to video file with h264 compression.

    Parameters
    ----------
    images : 4D numpy array
        Stack of RGB images to write to video file.
    filename : str
        File name to write video to. (Overwritten if exists)
    crf : int, optional
        Compression rate. 'Sane' values are 17-28. See 
        https://trac.ffmpeg.org/wiki/Encode/H.264
        The default is 20.
    verbose : int, optional
        Verbosity of console output. 
        The default is 1.

    Returns
    -------
    None.

    '''
    
    # Initialize ffmpeg parameters:
    height,width,_ = images[0].shape
    if height%2==1:
        height-=1
    if width%2==1:
        width-=1
    quiet = [] if verbose else ['-loglevel', 'error', '-hide_banner']
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height),r=7)
            .output(filename, pix_fmt='yuv420p', vcodec='libx264', crf=crf, preset='veryslow')
            .global_args(*quiet)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    
    # Write frames:
    for frame in images:
        process.stdin.write(
            frame[:height,:width]
                .astype(np.uint8)
                .tobytes()
        )
    
    # Close file stream:
    process.stdin.close()
    
    # Wait for processing + close to complete:
    process.wait()


def load_position(filename):
    '''
    Load position object from pickle file

    Parameters
    ----------
    filename : str
        Path to saved pickle file.

    Returns
    -------
    p : pipeline.Position object
        Reloaded Position object. (without reader and models)

    '''
    
    import pickle
    f = open(filename, 'rb')
    p = pickle.load(f)
    f.close()
    return p


#%% Misc

def findfirst(mylist):
    '''
    Find first non-zero element of list

    Parameters
    ----------
    mylist : list
        List of elements to scan through.

    Returns
    -------
    int
        index of first non-zero element (or None).

    '''
    return next((i for i, x in enumerate(mylist) if x>0), None)
