'''
This file contains function definitions for data manipulations and input/output
operations.

@author: jblugagne
'''
from __future__ import print_function
import numpy as np 
import os, glob, re, random, warnings, copy, importlib, cv2
import skimage.io as io
import tifffile
import skimage.transform as trans
from skimage.measure import label
import skimage.morphology as morph
from scipy import interpolate
import delta.utilities as utils


# Try to import elastic deformations, issue warning if not found:
if importlib.util.find_spec("elasticdeform") is None:
    warnings.warn("Could not load elastic deformations module.")
else:
    import elasticdeform
        



#%% UTILITIES:

def binarizerange(i):
    '''
    This function will binarize a numpy array by thresholding it in the middle
    of its range

    Parameters
    ----------
    i : 2D numpy array
        Input array/image.

    Returns
    -------
    newi : 2D numpy array
        Binarized image.

    '''
    
    newi = i
    newi[i > (np.amin(i)+np.amax(i))/2] = 1
    newi[i <= (np.amin(i)+np.amax(i))/2] = 0
    return newi

def readreshape(filename, target_size = (256,32), binarize = False, order = 1, rangescale = True, crop = False):
    '''
    Read image from disk and format it

    Parameters
    ----------
    filename : string
        Path to file. Only PNG, JPG or single-page TIFF files accepted
    target_size : tupe of int or None, optional
        Size to reshape the image. 
        The default is (256,32).
    binarize : bool, optional
        Use the binarizerange() function on the image.
        The default is False.
    order : int, optional
        interpolation order (see skimage.transform.warp doc). 
        0 is nearest neighbor
        1 is bilinear
        The default is 1.
    rangescale : bool, optional
        Scale array image values to 0-1 if True. 
        The default is True.
    mode : str, optional
        Resize the 
    Raises
    ------
    ValueError
        Raised if image file is not a PNG, JPEG, or TIFF file.

    Returns
    -------
    i : numpy 2d array of floats
        Loaded array.

    '''
    fext = os.path.splitext(filename)[1].lower()
    if fext in ('.png', '.jpg', '.jpeg'):
        i = io.imread(filename,as_gray = True)
    elif fext in ('.tif', '.tiff'):
        i = tifffile.imread(filename)
    else:
        raise ValueError('Only PNG, JPG or single-page TIF files accepted')
    if i.ndim == 3:
        i = i[:,:,0]
    # For DeLTA mothermachine, all images are resized in 256x32
    if not crop:
        img = trans.resize(i, target_size, anti_aliasing=True, order=order)
    # For DeLTA 2D, black space is added if img is smaller than target_size
    else: 
        fill_shape = [target_size[j] if i.shape[j] < target_size[j] else i.shape[j] for j in range(2)]
        img = np.zeros((fill_shape[0],fill_shape[1]))
        img[0:i.shape[0],0:i.shape[1]] = i 

    if binarize:
        img = binarizerange(img)
    if rangescale:
        if np.ptp(img) != 0:
            img = (img-np.min(img))/np.ptp(img)
    if np.max(img) == 255:
        img = img / 255
    return img

def postprocess(images,square_size=5,min_size=None, crop = False):
    '''
    A generic binary image cleaning function based on mathematical morphology.

    Parameters
    ----------
    images : 2D or 3D numpy array
        Input image or images stacked along axis=0.
    square_size : int, optional
        Size of the square structuring element to use for morphological opening
        The default is 5.
    min_size : int or None, optional
        Remove objects smaller than this minimum area value. If None, the 
        operation is not performed.
        The default is None.

    Returns
    -------
    images : 2D numpy array
        Cleaned, binarized images. Note that the dimensions are squeezed before 
        return (see numpy.squeeze doc)

    '''
    
    # Expand dims if 2D:
    if images.ndim == 2:
        images = np.expand_dims(images,axis=0)
    
    # Generate struturing element:
    selem = morph.square(square_size)
    
    # Go through stack:
    for index, I in enumerate(images):
        I = binarizerange(I)
        if not crop:
            I = morph.binary_opening(I,selem=selem)
        if min_size is not None:
            I = morph.remove_small_objects(I,min_size=min_size)
        images[index,:,:] = I
        
    return np.squeeze(images)


#%% DATA AUGMENTATION
def data_augmentation(images_input, aug_par, order=0, time=0):
    '''
    Data augmentation function

    Parameters
    ----------
    images_input : list of 2D numpy arrays of floats
        Images to apply augmentation operations to.
    aug_par : dict
        Augmentation operations parameters. Accepted key-value pairs:
            illumination_voodoo: bool. Whether to apply the illumination 
                voodoo operation.
            histogram_voodoo: bool. Whether to apply the histogram voodoo 
                operation.
            elastic_deformation: dict. If key exists, the elastic deformation
                operation is applied. The parameters are given as key-value 
                pairs. sigma values are given under the sigma key, deformation
                points are given under the points key. See elasticdeform doc.
            gaussian_noise: float. Apply gaussian noise to the image. The 
                sigma value of the gaussian noise is uniformly sampled between
                0 and +gaussian_noise.
            gaussain_blur: float. Apply gaussian blur to the image. The sigma
                value is the standard deviation of the kernel in the x and y
                direction.
            horizontal_flip: bool. Whether to flip the images horizontally. 
                Input images have a 50% chance of being flipped
            vertical_flip: bool. Whether to flip the images vertically. 
                Input images have a 50% chance of being flipped
            rotations90d: bool. Whether to randomly rotate the images in 90°
                increments. Each 90° rotation has a 25% chance of happening
            rotation: int/float. Range of random rotation to apply. The angle 
                is uniformly sampled in the range [-rotation, +rotation]
            zoom: float. Range of random "zoom" to apply. The image
                is randomly zoomed by a factor that is sampled from an 
                exponential distribution with a lamba of 3/zoom. The random
                factor is clipped at +zoom.
            shiftX: int/float. The range of random shifts to apply along X. A
                uniformly sampled shift between [-shiftX, +shiftX] is applied
            shiftY: int/float. The range of random shifts to apply along Y. A
                uniformly sampled shift between [-shiftY, +shiftY] is applied
            timeshiftX: int/float. The range of random shifts to apply along X
                to inputs from separate time points.
            timeshiftY: int/float. The range of random shifts to apply along Y
                to inputs from separate time points.
            timeshift_prob: float. Probability of time shift to be applied.
                
            Note that the same operations are applied to all inputs except for 
            the timeshift ones.
    order : int or list/tuple of ints, optional
        Interpolation order to use for each image in the input stack. If order
        is a scalar, the same order is applied to all images. If a list of
        orders is provided, each image in the stack will have its own operaiton
        order. See skimage.transform.wrap doc.
        Note that the histogram voodoo operation is only applied to images with
        a non-zero order.
        The default is 0.
    time : int or list of ints, optional
        Timepoint of each input. If a list is provided, inputs belonging to the
        same timepoint (e.g. 0 for previous timepoint images and 1 for current)
        will be treated the same in time-related transformations (e.g. 
        timeshift, where current frames are shifted relative to previous frames)

    Returns
    -------
    output : list of 2D numpy arrays of floats
        Augmented images array.

    '''
    
    #processing inputs / initializing variables::
    output = list(images_input)
    if np.isscalar(order) or len(order)==1:
        orderlist = [order] * len(images_input)
    else:
        orderlist = list(order)
    
    # Apply augmentation operations:
    
    if "illumination_voodoo" in aug_par:
        if aug_par["illumination_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = illumination_voodoo(item)
                    
    
                    
    if "histogram_voodoo" in aug_par:
        if aug_par["histogram_voodoo"]:
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    output[index] = histogram_voodoo(item)
                    
    
    if 'gaussian_noise' in aug_par:
        if aug_par['gaussian_noise']:
            sigma = np.random.rand()*aug_par['gaussian_noise']
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    item = item + np.random.normal(0,sigma,item.shape) # Add Gaussian noise
                    output[index] = (item-np.min(item))/np.ptp(item) # Rescale to 0-1
                
    if 'gaussian_blur' in aug_par:
        if aug_par['gaussian_blur']:
            sigma = np.random.rand()*aug_par['gaussian_blur']
            for index, item in enumerate(output):
                if order[index] > 0: # Not super elegant, but tells me if binary or grayscale image
                    item = cv2.GaussianBlur(item,(5,5),sigma) # blur image
                    output[index] = item 
    
    if "elastic_deformation" in aug_par:
        output = elasticdeform.deform_random_grid(output,
                                                  sigma=aug_par["elastic_deformation"]["sigma"],
                                                  points=aug_par["elastic_deformation"]["points"],
                                                  order=[i*3 for i in orderlist], # Using bicubic interpolation instead of bilinear here
                                                  mode='nearest',
                                                  axis=(0,1),
                                                  prefilter=False)
    
    if "horizontal_flip" in aug_par:
        if aug_par["horizontal_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.fliplr(item)
                    
    if "vertical_flip" in aug_par:
        if aug_par["vertical_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.flipud(item)
                    
    if "rotations_90d" in aug_par: # Only works with square images right now!
        if aug_par["rotations_90d"]:
            rot = random.randint(0,3)*90
            if rot>0: 
                for index, item in enumerate(output):
                    output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    if "rotation" in aug_par:
        rot = random.uniform(-aug_par["rotation"],aug_par["rotation"])
        for index, item in enumerate(output):
            output[index] = trans.rotate(item,rot,mode='edge',order=orderlist[index])
    
    # Zoom and shift operations are processed together:
    if "zoom" in aug_par:
        zoom = random.expovariate(3*1/aug_par["zoom"]) # I want most of them to not be too zoomed
        zoom = aug_par["zoom"] if zoom > aug_par["zoom"] else zoom
    else:
        zoom = 0
        
    if "shiftX" in aug_par:
        shiftX = random.uniform(-aug_par["shiftX"],aug_par["shiftX"])
    else:
        shiftX = 0
        
    if "shiftY" in aug_par:
        shiftY = random.uniform(-aug_par["shiftY"],aug_par["shiftY"])
    else:
        shiftY = 0
    
    # Apply zoom & shifts:
    if any([abs(x)>0 for x in [zoom,shiftX,shiftY]]):
        for index, item in enumerate(output):
            output[index] = zoomshift(
                item,
                zoom+1,
                shiftX,
                shiftY,
                order=orderlist[index]
                )
    
    return output

def zoomshift(I,zoomlevel,shiftX,shiftY, order=0):
    '''
    This function zooms and shifts images.

    Parameters
    ----------
    I : 2D numpy array
        input image.
    zoomlevel : float
        Additional zoom to apply to the image.
    shiftX : int
        X-axis shift to apply to the image, in pixels.
    shiftY : int
        Y-axis shift to apply to the image, in pixels.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    I : 2D numpy array
        Zoomed and shifted image of same size as input.

    '''
    
    oldshape = I.shape
    I = trans.rescale(I,zoomlevel,mode='edge',multichannel=False, order=order)
    shiftX = shiftX * I.shape[0]
    shiftY = shiftY * I.shape[1]
    I = shift(I,(shiftY, shiftX),order=order)
    i0 = (round(I.shape[0]/2 - oldshape[0]/2), round(I.shape[1]/2 - oldshape[1]/2))
    I = I[i0[0]:(i0[0]+oldshape[0]), i0[1]:(i0[1]+oldshape[1])]
    return I
    
def shift(image, vector, order=0):
    '''
    Image shifting function

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    vector : tuple of ints
        Translation/shit vector.
    order : int, optional
        Interpolation order. The default is 0.

    Returns
    -------
    shifted : 2D numpy image
        Shifted image.

    '''
    transform = trans.AffineTransform(translation=vector)
    shifted = trans.warp(image, transform, mode='edge',order=order)

    return shifted

def histogram_voodoo(image,num_control_points=3):
    '''
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the histogram conversion curve. 
        The default is 3.

    Returns
    -------
    2D numpy array
        Modified image.

    '''
    control_points = np.linspace(0,1,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)
    
    return mapping(image)

def illumination_voodoo(image,num_control_points=5):
    '''
    This function inspired by the one above.
    It simulates a variation in illumination along the length of the chamber

    Parameters
    ----------
    image : 2D numpy array
        Input image.
    num_control_points : int, optional
        Number of inflection points to use on the illumination multiplication
        curve. 
        The default is 5.

    Returns
    -------
    newimage : 2D numpy array
        Modified image.

    '''
    
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0,image.shape[0]-1,num=num_control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    # Apply this curve to the image intensity along the length of the chamebr:
    newimage = np.multiply(
        image,
        np.reshape(
            np.tile(np.reshape(curve,curve.shape + (1,)), (1, image.shape[1])),
            image.shape
            )
        )
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (image.min(), image.max()))
    
    return newimage



#%% SEGMENTATION FUNCTIONS:

def trainGenerator_seg(
        batch_size,
        img_path,
        mask_path,
        weight_path,
        target_size = (256,32),
        augment_params = {},
        preload = False,
        seed = 1,
        crop_windows = False
        ):
    '''
    Generator for training the segmentation U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images.
    mask_path : string
        Path to folder containing training segmentation groundtruth.
    weight_path : string
        Path to folder containing weight map images.
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    preload : bool, optional
        Flag to load all training inputs in memory during intialization of the
        generator.
        The default is False.
    seed : int, optional
        Seed for numpy's random generator. see numpy.random.seed() doc
        The default is 1.

    Yields
    ------
    image_arr : 4D numpy array of floats
        Input images for the U-Net training routine. Dimensions of the tensor 
        are (batch_size, target_size[0], target_size[1], 1)
    mask_wei_arr : 4D numpy array of floats
        Output masks and weight maps for the U-Net training routine. Dimensions
        of the tensor are (batch_size, target_size[0], target_size[1], 2)

    '''
    
    preload_mask = []
    preload_img = []
    preload_weight = []
    
    # Get training image files list:
    image_name_arr =    glob.glob(os.path.join(img_path,"*.png")) +\
                        glob.glob(os.path.join(img_path,"*.tif"))
    
    # If preloading, load the images and compute weight maps:
    if preload:
        for filename in image_name_arr:
            preload_img.append(readreshape(filename, target_size = target_size, order = 1, crop = crop_windows))
            preload_mask.append(readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False, crop = crop_windows))
            if weight_path is not None:
                preload_weight.append(readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0), rangescale = False, crop = crop_windows)
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    image_arr = np.empty((batch_size,)+target_size+(1,), dtype=np.float32)
    if weight_path is None:
        mask_wei_arr = np.empty((batch_size,)+target_size+(1,), dtype=np.float32)
    else:
        mask_wei_arr = np.empty((batch_size,)+target_size+(2,), dtype=np.float32)
    
    while True:
        # Reset image arrays:
        
        for b in range(batch_size):
            # Pick random image index:
            index = random.randrange(0,len(image_name_arr))
            
            if preload:
                # Get from preloaded arrays:
                img = preload_img[index]
                mask = preload_mask[index]
                weight = preload_weight[index]
            else:
                # Read images:
                filename = image_name_arr[index]
                img = readreshape(filename, target_size = target_size, order = 1, crop = crop_windows)
                mask = readreshape(os.path.join(mask_path,os.path.basename(filename)), target_size = target_size, binarize = True, order = 0, rangescale = False, crop = crop_windows)
                if weight_path is not None:
                    weight = readreshape(os.path.join(weight_path,os.path.basename(filename)), target_size = target_size, order = 0, rangescale = False, crop = crop_windows)
                else:
                    weight = []
                    
            if crop_windows:
                y0 = np.random.randint(0,img.shape[0] - (target_size[0] - 1))
                y1 = y0 + target_size[0]
                x0 = np.random.randint(0,img.shape[1] - (target_size[1] - 1))
                x1 = x0 + target_size[1]
                
                img = img[y0:y1,x0:x1]
                mask = mask[y0:y1,x0:x1]
                if weight_path is not None:
                    weight = weight[y0:y1,x0:x1]
                else:
                    weight = []
            
            # Data augmentation:
            if weight_path is not None:
                [img, mask, weight] = data_augmentation([img, mask, weight], augment_params, order=[1,0,0])
            else:
                [img, mask] = data_augmentation([img, mask], augment_params, order=[1,0])
            
            # Compile into output arrays:
            image_arr[b,:,:,0] = img
            mask_wei_arr[b,:,:,0] = mask
            if weight_path is not None:
                mask_wei_arr[b,:,:,1] = weight

        yield (image_arr, mask_wei_arr)


def saveResult_seg(save_path,npyfile, files_list = [], multipage=False):
    '''
    Saves an array of segmentation output images to disk

    Parameters
    ----------
    save_path : string
        Path to save folder.
    npyfile : 3D or 4D numpy array
        Array of segmentation outputs to save to individual files. If 4D, only 
        the images from the first index of axis=3 will be saved.
    files_list : list of strings, optional
        Filenames to save the segmentation masks as. png, tif or jpg extensions
        work.
        The default is [].
    multipage : bool, optional
        Flag to save all output masks as a single, multi-page TIFF file. Note
        that if the file already exists, the masks will be appended to it.
        The default is False.

    Returns
    -------
    None.

    '''
    
    for i,item in enumerate(npyfile):
        if item.ndim == 3:
            img = item[:,:,0]
        else:
            img = item
        if multipage:
            filename = os.path.join(save_path,files_list[0])
            io.imsave(filename,(img*255).astype(np.uint8),plugin='tifffile',append=True)
        else:
            if files_list:
                filename = os.path.join(save_path,files_list[i])
            else:
                filename = os.path.join(save_path,"%d_predict.png"%i)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(filename,(img*255).astype(np.uint8))


def predictGenerator_seg(files_path, files_list = None, target_size = (256,32),crop=False):
    '''
    Get a generator for predicting segmentation on new image files 
    once the segmentation U-Net has been trained.

    Parameters
    ----------
    files_path : string
        Path to image files folder.
    files_list : list/tuple of strings, optional
        List of file names to read in the folder. If empty, all 
        files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Size for the images to be resized.
        The default is (256,32).

    Returns
    ------
    mygen : generator
        Generator that will yield single image files as 4D numpy arrays of
        size (1, target_size[0], target_size[1], 1).

    '''
    
    
    files_list = files_list or sorted(os.listdir(files_path))

        
    def generator(files_path, files_list, target_size):
        for index, fname in enumerate(files_list):
            img = readreshape(os.path.join(files_path,fname),
                              target_size=target_size,
                              order=1,
                              crop = crop)
            img = np.reshape(img,(1,)+img.shape) # Tensorflow needs one extra single dimension (so that it is a 4D tensor)
            
            yield img

    mygen = generator(files_path, files_list, target_size)
    return mygen
    

def seg_weights(mask, classweights=(1,1), w0=12, sigma=2):
    '''
    This function computes the weight map as described in the original U-Net
    paper to force the model to learn borders 
    (Slow, best to run this offline before training)

    Parameters
    ----------
    mask : 2D array
        Training output segmentation mask.
    classweights : tuple of 2 int/floats
        Weights to apply to background, foreground.
        The default is (1,1)
    w0 : int or float, optional
        Base weight to apply to smallest distance (1 pixel). 
        The default is 12.
    sigma : int or float, optional
        Exponential decay rate to apply to distance weights.
        The default is 2.

    Returns
    -------
    weightmap : 2D array
        Weights map image.

    '''

    # Get cell labels mask:
    lblimg,lblnb = label(mask,connectivity=1,return_num=True)
    
    # Compute cell-to-cell distance masks:
    distance_array = float('inf')*np.ones(mask.shape[0:2] + (max(lblnb,2),))
    for i in range(0,lblnb):
        _, distance_array[:,:,i] = morph.medial_axis(
            (lblimg==i+1)==0,return_distance=True
            )
    
    # Keep 2 smallest only:
    distance_array = np.sort(distance_array,axis=-1)[:,:,0:2]
    
    # Compute weights map:
    weightmap = w0*np.exp(-np.square(np.sum(distance_array,-1))/(2*sigma^2))
    weightmap = np.multiply(weightmap,mask==0)
    weightmap = np.add(weightmap,(mask[:,:,0]==0)*classweights[0])
    weightmap = np.add(weightmap,(mask[:,:,0]==1)*classweights[1])
    
    return weightmap

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

def estimate_seg2D_classweights(mask_path, sample_size = None):
    '''
    
    Parameters
    ----------
    mask_path : str
        Path to folder containing segmentations.
    sample_size : int, optional
        Determines the size of the training set that will be used to calcualte class weights. The default is None.

    Returns
    -------
    class1 : Int
        weight of class 1.
    class2 : Int
        weight of class 2.

    '''
    mask_name_arr = glob.glob(os.path.join(mask_path,'*.png')) + \
                    glob.glob(os.path.join(mask_path,'*.tif'))
                    
    # Take a sample size of training set to reduce computation time. 
    if sample_size:
        mask_name_arr = random.sample(mask_name_arr,sample_size)
        
    c1 = 0
    c2 = 0
    
    for mask_name in mask_name_arr:
        
        mask = cv2.imread(mask_name,0) / 255
        # Extract all pixels that include the cells and it's border (no background)
        border = (cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel(20))) 
        # Set all pixels that include the cells to zero to leave behind the border only
        border[mask>0] = 0   
             
        # Bkgd (background) is just the image minus the cells + border
        bkgd = np.ones((mask.shape))
        bkgd = bkgd - (mask + border)   
        
        # Erode the segmentation to avoid putting high emphasiss on edges of cells 
        mask_erode = cv2.erode(mask,kernel(2))   
        mask_dil = cv2.dilate(mask,kernel(3))
        border_erode = (mask_dil < 1) * (border > 0) * 1 
        
        skel = morph.skeletonize(mask_erode>0) * 1
        skel_border = morph.skeletonize(border_erode>0) * 1
            
        c1 = c1 + np.sum(np.sum(skel.astype(np.float64)))    
        c2 = c2 + np.sum(np.sum(skel_border.astype(np.float64)))  
        
    if c1 > c2:
        class1 = c2 / c1
        class2 = 1
    else:
        class1 = 1
        class2 = 1 * c1 / c2
    
    return class1, class2
    
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

def estimateClassweights(gene, num_samples = 30):
    '''
    Estimate the class weights to use with the weighted 
    categorical cross-entropy based on the output of the trainGenerator_track
    output.

    Parameters
    ----------
    gene : generator
        Tracking U-Net training generator. (output of trainGenerator_seg/track)
    num_samples : int, optional
        Number of batches to use for estimation. The default is 30.

    Returns
    -------
    class_weights : tuple of floats
        Relative weights of each class. Note that, if 0 elements of a certain 
        class are present in the samples, the weight for this class will be set 
        to 0.

    '''
    
    sample = next(gene)
    if sample[1].shape[-1]==1:
        class_counts = [0, 0]
    else:
        class_counts = [0] * sample[1].shape[-1] # List of 0s
    
    # Run through samples and classes/categories:
    for _ in range(num_samples):
        if sample[1].shape[-1]==1:
            class_counts[1] += np.mean(sample[1]>0)
            class_counts[0] += np.mean(sample[1]==0)
        else:
            for i in range(sample[1].shape[-1]):
                class_counts[i] += np.mean(sample[1][..., i])
        sample = next(gene)
        
    # Warning! If 0 elements of a certain class are present in the samples, the
    # weight for this class will be set to 0. This is for the tracking case
    # (Where there are only empty daughter images in the training set)
    # Try changing the num_samples value if this is a problem
    class_weights = [(x/num_samples)**-1 if x!= 0 else 0 for x in class_counts] # Normalize by nb of samples and invert to get weigths, unless x == 0 to avoid Infinite weights or errors
    
    return class_weights
        
#%% TRACKING FUNCTIONS
    
def trainGenerator_track(
        batch_size,
        img_path, 
        seg_path, 
        previmg_path, 
        segall_path,
        track_path,
        weights_path = None,
        augment_params = {},
        crop_windows = False,
        target_size = (256,32),
        shiftcropbox = 0,
        seed = 1
        ):
    '''
    Generator for training the tracking U-Net.

    Parameters
    ----------
    batch_size : int
        Batch size, number of training samples to concatenate together.
    img_path : string
        Path to folder containing training input images (current timepoint).
    seg_path : string
        Path to folder containing training 'seed' images, ie mask of 1 cell 
        in the previous image to track in the current image.
    previmg_path : string
        Path to folder containing training input images (previous timepoint).
    segall_path : string
        Path to folder containing training 'segall' images, ie mask of all 
        cells in the current image.
    track_path : string
        Path to folder containing tracking groundtruth, ie mask of 
        the seed cell and its potential daughter tracked in the current frame.
    weights_path : string or None, optional
        Path to folder containing pixel-wise weights to apply to the tracking 
        groundtruth. If None, the same weight is applied to all pixels. If the
        string is 'online', weights will be generated on the fly (not 
        recommended, much slower)
        The default is None.
    augment_params : dict, optional
        Data augmentation parameters. See data_augmentation() doc for more info
        The default is {}.
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).
    shiftcropbox: int
        Determine the max number of pixels that the cropbox for the current frames (img,segall,mot_dau,wei) inputs will be shifted
    seed : int, optional
        Seed for numpy's random generator. see numpy.random.seed() doc
        The default is 1.

    Yields
    ------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the U-Net training routine. Dimensions of 
        the tensor are (batch_size, target_size[0], target_size[1], 4)
    outputs_arr : 4D numpy array of floats
        Output masks for the U-Net training routine. Dimensions of the tensor 
        are (batch_size, target_size[0], target_size[1], 3). The third index
        of axis=3 contains 'background' masks, ie the part of the tracking 
        output groundtruth that is not part of the mother or daughter masks
    '''
    
    
    # Initialize variables and arrays:
    image_name_arr =    glob.glob(os.path.join(img_path,"*.png")) +\
                        glob.glob(os.path.join(img_path,"*.tif"))
    X = np.empty((batch_size,) + target_size + (4,), dtype=np.float32)
    Y = np.empty((batch_size,) + target_size + (2,), dtype=np.float32)
    
    # Reset the pseudo-random generator:
    random.seed(a=seed)
    
    while True:
        
        X[:,:,:,:] = 0
        Y[:,:,:,:] = 0

        for b in range(batch_size):
            # Pick random image file name:
            index = random.randrange(0,len(image_name_arr))
            filename = image_name_arr[index]
            
            # Read images:
            img = readreshape(filename, target_size = target_size, order = 1, crop=True)
            seg = readreshape(
                os.path.join(seg_path,os.path.basename(filename)),
                target_size = target_size, 
                binarize = True, 
                order = 0, 
                rangescale = False,
                crop=True
                )
            previmg = readreshape(
                os.path.join(previmg_path,os.path.basename(filename)), 
                target_size = target_size, 
                order = 1,
                crop=True
                )
            segall = readreshape(
                os.path.join(segall_path,os.path.basename(filename)), 
                target_size = target_size, 
                binarize = True, 
                order = 0, 
                rangescale = False,
                crop=True
                )

            track = readreshape(
                os.path.join(track_path,os.path.basename(filename)),
                target_size = target_size, 
                binarize = True, 
                order = 0, 
                rangescale = False,
                crop=True
                )
            if weights_path is not None:
                if weights_path == 'online':
                    weights = tracking_weights(track, segall)
                else:
                    weights = readreshape(
                        os.path.join(weights_path,os.path.basename(filename)),
                        target_size = target_size, 
                        binarize = False, 
                        order = 0, 
                        rangescale = False,
                        crop=True
                        )
            else:
                weights = np.ones_like(track)
                        
            # Data augmentation:
            [img, seg, previmg, segall, track, weights] = data_augmentation(
                [img, seg, previmg, segall, track, weights],
                augment_params, 
                order=[1,0,1,0,0,0]
                )
            
            if crop_windows:
                cb, fb = utils.gettrackingboxes(seg)
                shiftY, shiftX = utils.getshiftvalues(shiftcropbox,seg.shape,cb)
            else:
                cb = fb = dict(ytl=0, xtl=0, ybr=seg.shape[0], xbr=seg.shape[1])
                shiftY, shiftX = 0,0
            
            # Add into arrays:

            X[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],1] = utils.cropbox(
                seg, cb
                )
            X[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],2] = utils.cropbox(
                previmg, cb
                )
            
            # Shift crop box for current frame inputs
            cb['xtl'] += shiftX; cb['xbr'] += shiftX
            cb['ytl'] += shiftY; cb['ybr'] += shiftY

            X[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],0] = utils.cropbox(
                img, cb
                )
            X[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],3] = utils.cropbox(
                segall, cb
                )
            
            Y[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],0] = utils.cropbox(
                track, cb
                )
            Y[b,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],1] = utils.cropbox(
                weights, cb
                )
        
        # Yield batch:
        yield X, Y

def getfnfromprototype(prototype,fileorder,position,chamber,frame,cellnb = None):
    '''
    Generate full filename for specific frame based on file path, 
    prototype, fileorder, and filenamesindexing

    Parameters
    ----------
    position : int
        Position/series index (0-based indexing).
    channel : int
        Imaging chamber index (0-based indexing).
    frame : int
        Frame/timepoint index (0-based indexing).

    Returns
    -------
    string
        Filename.

    '''
    filenumbers = []
    
    for i in fileorder:
        if i=='p':
            filenumbers.append(position)
        if i=='c':
            filenumbers.append(chamber)
        if i=='t':
            filenumbers.append(frame)
        if i=='n':
            filenumbers.append(cellnb)
    
    if cellnb is not None:
        prototype = os.path.splitext(prototype)[0] + 'Cell%06d' + os.path.splitext(prototype)[1]
    
    filename = prototype % tuple(filenumbers)
    
    return filename

def predictCompilefromseg_track(
        img_path,seg_path, files_list = [], target_size = (256,32), crop = False
        ):
    '''
    Compile an inputs array for tracking prediction with the tracking U-Net, 
    directly from U-Net segmentation masks saved to disk.

    Parameters
    ----------
    img_path : string
        Path to original single-chamber images folder. The filenames are 
        expected in the printf format Position%02d_Chamber%02d_Frame%03d.png
    seg_path : string
        Path to segmentation output masks folder. The filenames must be the 
        same as in the img_path folder.
    files_list : tuple/list of strings, optional
        List of filenames to compile in the img_path and seg_path folders. 
        If empty, all files in the folder will be read.
        The default is [].
    target_size : tuple of 2 ints, optional
        Input and output image size. 
        The default is (256,32).

    Returns
    -------
    inputs_arr : 4D numpy array of floats
        Input images and masks for the tracking U-Net training routine. 
        Dimensions of the tensor are (cells_to_track, target_size[0], 
        target_size[1], 4), with cells_to_track the number of segmented cells
        in all segmentation masks of the files_list.
    seg_name_list : list of strings
        Filenames to save the tracking outputs as. The printf format is 
        Position%02d_Chamber%02d_Frame%03d_Cell%02d.png, with the '_Cell%02d'
        string appended to signal which cell is being seeded/tracked (from top
        to bottom)
        

    '''
    
    seg_name_list = []
    if not files_list:
        files_list = sorted(os.listdir(img_path))
    
    ind = 0
    
    numstrs = re.findall("\d+", files_list[0]) # Get digits sequences in first filename
    charstrs = re.findall("\D+", files_list[0]) # Get character sequences in first filename
    
    # Create the string prototype to be used to generate filenames on the fly:
    if not (len(numstrs)==3 and len(charstrs)==4 or len(numstrs) == 2 and len(charstrs) == 3):
        raise ValueError('Filename formatting error. See documentation for image sequence formatting')
    elif len(numstrs)==3 and len(charstrs)==4:
    # Order is position, chamber, frame/timepoint
        prototype = charstrs[0]+'%0'+str(len(numstrs[0]))+'d'+\
                   charstrs[1]+'%0'+str(len(numstrs[1]))+'d'+\
                   charstrs[2]+'%0'+str(len(numstrs[2]))+'d'+\
                   charstrs[3] 
        fileorder = 'pct'
    elif len(numstrs)==2 and len(charstrs)==3:
    # Order is position, frame/timepoint
        prototype = charstrs[0]+'%0'+str(len(numstrs[0]))+'d'+\
                    charstrs[1]+'%0'+str(len(numstrs[1]))+'d'+\
                    charstrs[2]
        fileorder = 'pt'
        crop = True
    fileordercell = fileorder + 'n'
    for index, item in enumerate(files_list):
            filename = os.path.basename(item)
            # Get position, chamber & frame numbers:
            if fileorder == 'pct':
                (pos, cha, fra) = list(map(int, re.findall("\d+",filename)))
            elif fileorder == 'pt':
                (pos, fra) = list(map(int, re.findall("\d+",filename)))
                cha = None
            
            if fra > 1:
                
                prevframename = getfnfromprototype(prototype,fileorder,position=pos,chamber=cha,frame=fra-1)
                   
                img = readreshape(os.path.join(img_path,filename), target_size = target_size, order = 1, crop = crop)
                segall = readreshape(os.path.join(seg_path,filename), target_size = target_size, order = 0, binarize = True, rangescale = False, crop = crop)
                previmg = readreshape(os.path.join(img_path,prevframename), target_size = target_size, order = 1, crop = crop)
                prevsegall = readreshape(os.path.join(seg_path,prevframename), target_size = target_size, order = 0, binarize = True, rangescale = False, crop = crop)
                
                lblimg,lblnb = label(prevsegall,connectivity=1,return_num=True)
                
                x = np.zeros(
                    shape=(lblnb,)+target_size+(4,),
                    dtype=np.float32
                    )
                boxes = []
                
                for lbl in range(1,lblnb+1):
                    seg = lblimg == lbl
                    seg = seg.astype(np.uint8) # Output is boolean otherwise
                    
                    # Cell-centered crop boxes:
                    if crop:
                        cb, fb = utils.gettrackingboxes(seg, img.shape)
                    else:
                        cb = fb = dict(ytl=None, xtl=None, ybr=None, xbr=None)
                    boxes += [(cb, fb)]
                    
                    # Current image
                    x[lbl-1,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],0] = utils.cropbox(
                        img, cb
                        )
                    
                    x[lbl-1,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],1] = utils.cropbox(
                        seg, cb
                        )            
                    # Previous image
                    x[lbl-1,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],2] = utils.cropbox(
                        previmg, cb
                        )
                    
                    # Segmentation of all current cells
                    x[lbl-1,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],3] = utils.cropbox(
                        segall, cb
                        )
                    
                    segfilename = getfnfromprototype(prototype,fileordercell,position=pos,chamber=cha,frame=fra-1,cellnb=lbl)

                    seg_name_list.append(segfilename)
                                    
                if ind == 0:
                    inputs_arr = x
                    ind = 1
                else:
                    inputs_arr = np.concatenate((inputs_arr,x),axis=0)
    return inputs_arr, seg_name_list, boxes


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
    weights = utils.rangescale(dist_in+1,(1,42))
    weights[skel>0] = 255
    
    # Rest of the image is weighed according to distance from tracked cell:
    weights+= 63*(dist_from/halo_distance) \
        *(segall>0).astype(np.float32)\
            *(track==0).astype(np.float32)
    
    weights*=100
    weights[dist_from<1] = 1
        
    return weights
    
    

def saveResult_track(save_path,npyfile, files_list = None):
    '''
    Save tracking output masks to disk

    Parameters
    ----------
    save_path : string
        Folder to save images to.
    npyfile : 4D numpy array
        Array of tracking outputs to save to individual files.
    files_list : tuple/list of strings, optional
        Filenames to save the masks as. Note that the 'mother_' and 'daughter_'
        prefixes will be added to those names. If None, numbers will be used.
        The default is None.

    Returns
    -------
    None.

    '''
    
    mothers = npyfile[:,:,:,0]
    for i,mother in enumerate(mothers):
        if files_list:
            filenameMo = os.path.join(save_path,files_list[i])
        else:
            filenameMo = os.path.join(save_path,"tracking_09d.png"%i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filenameMo,(mother*255).astype(np.uint8))
