'''
This script trains the tracking U-Net.

@author: jblugagne
'''
from delta.utilities import cfg
import os
from delta.model import unet_track
from delta.data import trainGenerator_track
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Files:
training_set = cfg.training_set_track
savefile = cfg.model_file_track

# Training parameters:
batch_size = 2
epochs = 500
steps_per_epoch = 300
patience = 50

#Data generator parameters:
data_gen_args = dict(
    rotation = 1,
    zoom=.15,
    horizontal_flip=True,
    histogram_voodoo=True,
    illumination_voodoo=True,
    )

# Generator init:
myGene = trainGenerator_track(
    batch_size,
    os.path.join(training_set,'img'),
    os.path.join(training_set,'seg'),
    os.path.join(training_set,'previmg'),
    os.path.join(training_set,'segall'),
    os.path.join(training_set,'mot_dau'),
    os.path.join(training_set,'wei'),
    data_gen_args,
    target_size = cfg.target_size_track,
    crop_windows=cfg.crop_windows,
    shiftcropbox = 5
    )
   

# Define model:
model = unet_track(input_size = cfg.target_size_track+(4,))
model.summary()

# Callbacks:
model_checkpoint = ModelCheckpoint(
    savefile, monitor='loss', verbose=1, save_best_only=True
    )
early_stopping = EarlyStopping(
    monitor='loss', mode='min', verbose=1, patience=patience
    )

# Train:
history = model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint, early_stopping]
    )
