'''
This script will run the rois identification/segmentation U-Net. 

To see how to extract roi images with this segmentation masks, see the 
preprocessing section of pipeline.py and getROIboxes() in utilities.py

@author: jblugagne
'''
from delta.utilities import cfg
from delta.data import saveResult_seg, predictGenerator_seg, postprocess
from delta.model import unet_rois
from os import listdir

# Files:
inputs_folder = 'D:/DeLTA_data/mother_machine/evaluation/sequence/'
outputs_folder = 'D:/DeLTA_data/mother_machine/evaluation/chambers_masks/'
model_file = cfg.model_file_rois
input_files = listdir(inputs_folder)

# Load up model:
model = unet_rois(input_size = cfg.target_size_rois+(1,))
model.load_weights(model_file)

# Predict:
predGene = predictGenerator_seg(
    inputs_folder, files_list=input_files, target_size=cfg.target_size_rois
    )
results = model.predict_generator(predGene,len(input_files),verbose=1)

# Post process results:
results[:,:,:,0] = postprocess(results[:,:,:,0])

# Save to disk:
saveResult_seg(outputs_folder,results,files_list=input_files)