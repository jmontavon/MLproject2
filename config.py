# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:42:05 2021
bla
@author: jblugagne
"""

# Download latest models first:
# https://drive.google.com/drive/folders/1nTRVo0rPP9CR9F6WUunVXSXrLNMT_zCP

#presets = 'mothermachine' # 'mothermachine' or '2D'
presets = '2D'

if presets == 'mothermachine':
    models = ['rois', 'segmentation', 'tracking']
    model_file_rois = 'data/models/chambers_id_tessiechamp.hdf5'
    model_file_seg = 'data/models/unet_moma_seg_multisets.hdf5'
    model_file_track = 'data/models/unet_moma_track_v2.hdf5'
    target_size_rois = (512, 512)
    target_size_seg = (256, 32)
    target_size_track = target_size_seg
    
    # Training sets:
    training_set_rois = 'data/training/chambers_seg_set/train'
    training_set_seg = 'data/training/segmentation_set/train_multisets/'
    training_set_track = 'data/training/tracking_set/train_multisets'
    
    # Pipeline parameters:
    rotation_correction = True # Whether to perform automated image rotation correction
    drift_correction = True # Whether to perform image drift correction
    crop_windows = False # Whether to crop out windows for segmentation and tracking
    min_roi_area = 500 # Minimum size of ROIs/chambers in pixels for area filtering (0 to +Inf)
    
elif presets == '2D':
    models = ['segmentation', 'tracking']
    model_file_seg = '/Users/acoudray/Desktop/tmp_samba/phd_courses/machine_learning/project2/delta_dev_branch/delta/training_curated_seg_1/model_seg_CustomWeights_sig50_q085_pow5.hdf5'
    model_file_track = '/Users/acoudray/Desktop/tmp_samba/phd_courses/machine_learning/project2/delta_dev_branch/delta/training_tracking_n3/model_track.hdf5'

    target_size_seg = (512, 512)
    target_size_track = (256, 256)
    
    # Training sets:
    training_set_seg = '/Users/acoudray/Desktop/tmp_samba/phd_courses/machine_learning/project2/delta_dev_branch/delta/training_curated_seg_1'
    training_set_track = '/Users/acoudray/Desktop/tmp_samba/phd_courses/machine_learning/project2/delta_dev_branch/delta/training_tracking_n3'
    
    # Pipeline parameters:
    rotation_correction = False # Whether to perform automated image rotation correction
    drift_correction = False # Whether to perform image drift correction
    crop_windows = True # Whether to crop out windows for segmentation and tracking

# Other pipeline parameters:
whole_frame_drift = False # Whether to use the whole frame for drift correction
min_cell_area = 20 # Minimum area of cells in pixels
save_format = ('pickle', 'legacy', 'movie') # Format(s)) to save the data to.

#%% Tensorflow technical parameters:
# Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# If running into OOM issues or having trouble with cuDNN loading, try setting
# memory_growth_limit to a value in MB: (eg 1024, 2048...)
memory_growth_limit = None

if memory_growth_limit is not None:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_growth_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
