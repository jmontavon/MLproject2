'''
This script trains the cell segmentation U-Net

@author: jblugagne
'''
from delta.utilities import gen_cfg
from delta.model import unet_seg
from delta.data import trainGenerator_seg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import argparse

parser = argparse.ArgumentParser(description='Train cell segmentation with delta using custom inputs')
parser.add_argument('-i', '--path_to_config', type=str, help='custom config file with path to training set and custom options')
args = parser.parse_args()

cfg = gen_cfg(args.path_to_config)

# Files:
training_set = cfg.training_set_seg
savefile = cfg.model_file_seg
print('training set is : ',training_set)
print('model will be saved to : ',savefile)

# Training parameters:
batch_size = 1
epochs = 3
steps_per_epoch = 300
patience = 50

#Data generator parameters:
data_gen_args = dict(
    rotation = 2,
    rotations_90d = True,
    zoom=.15,
    horizontal_flip=True,
    vertical_flip=True,
    illumination_voodoo=True,
    gaussian_noise = 0.03,
    gaussian_blur = 1
    )

# Generator init:
myGene = trainGenerator_seg(
    batch_size,
    os.path.join(training_set,'img'),
    os.path.join(training_set,'seg'),
    os.path.join(training_set,'wei'),
    augment_params = data_gen_args,
    target_size = cfg.target_size_seg,
    crop_windows = cfg.crop_windows
    )

# Define model:
model = unet_seg(input_size = cfg.target_size_seg+(1,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor='loss', verbose=2, save_best_only=True
    )
early_stopping = EarlyStopping(
    monitor='loss', mode='min', verbose=2, patience=patience
    )

# Train:
history = model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping]
    )
