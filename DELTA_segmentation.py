'''
This script runs the segmentation U-Net on data that has been pre-processed to
crop out chamber images. See delta-interfacing/preprocessing.m or pipeline.py.

The images are processed by batches of 4096 to prevent memory issues.

@author: jblugagne
'''

from delta.data import saveResult_seg, predictGenerator_seg, postprocess, readreshape
from delta.model import unet_seg
import os
import delta.utilities as utils
from delta.utilities import cfg
import numpy as np

# Files:
inputs_folder = 'testing_curated_seg_1/img'
outputs_folder = 'testing_curated_seg_1/seg_out_Delta_defaults'

if not os.path.exists(outputs_folder):
    os.mkdir(outputs_folder)
    
unprocessed = sorted(os.listdir(inputs_folder))

# Load up model:
model = unet_seg(input_size = cfg.target_size_seg+(1,))
model.load_weights(cfg.model_file_seg)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(4096,len(unprocessed)) #4096 at a time
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Predict:
    predGene = predictGenerator_seg(
                                    inputs_folder, 
                                    files_list = to_process,
                                    target_size = cfg.target_size_seg,
                                    crop = cfg.crop_windows)
    # Use mother machine model
    if not cfg.crop_windows:
        results = model.predict(predGene,verbose=1)[:,:,:,0]
    # Use 2D surfaces model
    else:
        # Create array to store predictions
        img = readreshape(os.path.join(inputs_folder,to_process[0]),target_size=cfg.target_size_seg,crop=True)
        results = np.zeros((len(to_process),img.shape[0],img.shape[1],1))
        # Crop, segment, stitch and store predictions in results
        for i in range(len(to_process)):
            windows, loc_y, loc_x = utils.create_windows(next(predGene)[0,:,:], target_size=cfg.target_size_seg                                                        )
            windows = windows[:,:,:,np.newaxis]
            pred = model.predict(windows,verbose=1,steps=windows.shape[0])
            pred = utils.stitch_pic(pred[:,:,:,0],loc_y,loc_x)[np.newaxis,:,:,np.newaxis]
            
            results[i] = pred
                
    # Post process results:
    results = postprocess(results, crop = cfg.crop_windows)
    
    # Save to disk:
    saveResult_seg(outputs_folder,results, files_list = to_process)
