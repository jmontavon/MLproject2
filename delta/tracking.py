'''
This script runs the tracking U-Net on data out of the segmentation U-Net.

Images are processed in batches of 512, although the number of actual samples
run through the tracking U-Net will depend on the number of cells in each 
image.

Format of images needs to be Position-Number-Chamber-Number-Frame-Number.fext or Position-Number-Frame-Number.fext

If there are chambers, then it is assumed to be run in the mothermachine. 
If there are no chambers, then images will be cropped as done for DeLTA 2D
@author: jblugagne
'''

from delta.utilities import cfg
from delta.data import saveResult_track, predictCompilefromseg_track
from delta.model import unet_track
import os, cv2

images_folder = 'testing_curated_seg_1/img'
segmentation_folder = 'testing_curated_seg_1/seg_output/'
outputs_folder = 'testing_curated_seg_1/track_output'

if not os.path.exists(outputs_folder):
    os.mkdir(outputs_folder)
    
unprocessed = sorted(os.listdir(images_folder))

# Load up model:
model = unet_track(input_size = cfg.target_size_track+(4,))
model.load_weights(cfg.model_file_track)

# Process
while(unprocessed):
    # Pop out filenames
    ps = min(512,len(unprocessed))
    to_process = unprocessed[0:ps]
    del unprocessed[0:ps]
    
    # Get data:
    inputs, seg_filenames, boxes = predictCompilefromseg_track(
        images_folder, 
        segmentation_folder,
        files_list = to_process, 
        target_size = cfg.target_size_track,
        crop = cfg.crop_windows
        )

    # Predict:
    results = model.predict(inputs,verbose=1)
    
    # Save (use the filenames list from the data compiler)
    saveResult_track(outputs_folder,results, files_list = seg_filenames)
