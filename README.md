### MLproject2
Project for the EPFL Machine Learning course done in the scope of ML4Science.

## Installation of Python Env

First install the environment project-env
`python3 -m venv project-env` 

Connect yourself to this newly created env
`source project-env/bin/activate`

Then install all requirements : 
`pip install -r utils/requirements.txt`

Enjoy !

Tested on MACOSX Catalina. Tensorflow-GPU for DELTA was not tested. 

## Launching the training of DELTA models 

All the path to the training dataset folder and to the model file are in config.py. All the path should be correct. It needs the folder data/ to be in the root of the project folder.

To launch segmentation training : 

`python3 DELTA_train_segmentation.py`

To launch cell tracking training :

`python3 DELTA_train_tracking.py`

# Prediction using already built DELTA models 

All the path to the training dataset folder and to the model file are in config.py. All the path should be correct. It needs the folder data/ to be in the root of the project folder.

To launch segmentation training : 

`python3 DELTA_train_segmentation.py`

