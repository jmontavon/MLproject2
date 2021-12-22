# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:41:45 2021

@author: jeanbaptiste
"""




# Modules:
import time, cv2, os, threading
import numpy as np
from delta.data import postprocess
import delta.utilities as utils
from delta.utilities import cfg

class Pipeline(threading.Thread):
    '''
    Main Pipeline class to process all positions.
    '''
    
    
    def __init__(
            self,
            xpreader,
            resfolder = None,
            on_the_fly = False,
            reload = False,
            verbose = 1
            ):
        '''
        Initialize Pipeline

        Parameters
        ----------
        xpreader : object
            utilities xpreader object.
        resfolder : str, optional
            Path to folder to save results to.
            The default is None.
        reload : bool, optional
            Flag to reload previous position files from resfolder.
            The default is False.
        verbose : int, optional
            Verbosity flag. The default is 1.

        Returns
        -------
        None.

        '''
        
        super().__init__()
        
        self.reader = xpreader
        self.resfolder = resfolder
        self.positions = []
        self.rotation_correction = cfg.rotation_correction
        self.drift_correction = cfg.drift_correction
        self.crop_windows=cfg.crop_windows
        self.save_format=cfg.save_format
        self.daemon=True

        if self.reader.resfolder is not None:
            self.resfolder = self.reader.resfolder
            
        # Load models:
        self.models = utils.loadmodels()
        
        # Create result files folders
        if self.resfolder is None:
            xpfile = self.reader.filename
            if os.path.isdir(xpfile):
                self.resfolder = os.path.join(xpfile, "delta_results")
            else:
                self.resfolder = os.path.splitext(xpfile)[0] + "_delta_results"
        if not os.path.exists(self.resfolder):
            os.mkdir(self.resfolder)

        
        
        # Initialize position processors:
        for p in range(self.reader.positions):
            self.positions += [
                Position(
                    p,
                    self.reader,
                    self.models,
                    drift_correction=self.drift_correction,
                    crop_windows=self.crop_windows
                    )
                ]
        
        # If reload flag, reload positions from pickle files:
        if reload:
            for p in range(self.reader.positions):
                self.positions[p].load(
                    os.path.join(self.resfolder,'Position%06d'%(p))
                    )
    
    
    def preprocess(self, positions=None, references=None, ROIs = 'model'):
        '''
        Pre-process positions (Rotation correction, identify ROIs, 
        initialize drift correction)

        Parameters
        ----------
        positions : list of int or None, optional
            List of positions to pre-process. If None, all will be run.
            The default is None.
        references : 3D array or None, optional
            Reference images to use to perform pre-processing. If None,
            the first image of each position will be used. Dimensions
            are (positions, size_y, size_x)
            The default is None.
        ROIs : None or 'model', optional
            Regions of interest. If None, whole frames are treated as one ROI.
            If 'model', the ROIs model from cfg.model_file_rois will be used
            to detect them. Otherwise, a list of ROIs can be provided in the
            format of the utilities.py cropbox function input box.
            The default is 'model'.

        Returns
        -------
        None.

        '''
        # TODO implement ROIs mode selection here instead of cfg
        
        # Process positions to run:
        if positions is None:
            positions_torun = range(self.reader.positions)
        else:
            positions_torun = positions
        
        # If no reference images provided, pass Nones:
        if references is None:
            references = [None for _ in positions_torun]
        
        # Run preprocessing:
        for p in positions_torun:
            self.positions[p].preprocess(
                reference=references[p],
                rotation_correction=self.rotation_correction
                )


    def process(
            self,
            positions = None,
            frames = None,
            features = None
            ):
        '''
        Run pipeline.

        Parameters
        ----------
        positions : list of int or None, optional
            List of positions to run. If None, all positions are run.
            The default is None.
        frames : list of int or None, optional
            List of frames to run. If None, all frames are run.
            The default is None.
        features : list of str or None, optional
            List of features to extract. If None, all features are extracted.
            The default is None.

        Returns
        -------
        None.

        '''
        
        if frames is None:
            frames = [f for f in range(self.reader.timepoints)]
        
        if positions is None:
            positions = range(self.reader.positions)

        if features is None:
            features = ['length','width','area','perimeter','edges']
            for c in range(1,self.reader.channels):
                features += ['fluo%d'%(c,)]
        
        # Run through positions:
        for p in positions:
            
            # Preprocess is not done already:
            if not self.positions[p]._preprocessed:
                self.positions[p].preprocess(
                    rotation_correction=self.rotation_correction
                    )
            
            # Segment all frames:
            self.positions[p].segment(frames=frames)
            
            # Track cells:
            self.positions[p].track(frames=frames)
            
            # Extract features:
            self.positions[p].features(frames=frames, features=features)
            
            # Save to disk and clear memory:
            self.positions[p].save(
                filename=os.path.join(self.resfolder,'Position%06d'%(p)),
                frames = frames,
                save_format=self.save_format
                )
            self.positions[p].clear()
    
    
    def run(self):
        '''
        On-the-fly processor (not functional yet)

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        if not self.on_the_fly:
            raise RuntimeError(
                'This Pipeline was not initialized for on-the-fly processing'
                )
        
        while True:
            
            # Check if new files are available:
            newfiles = self.reader.watcher.newfiles
            while len(newfiles)==0:
                newfiles = self.reader.watcher.newfiles()
            
            for nf in newfiles:
                pos, chan = nf
                frame = self.reader.watcher.old[pos,chan]
                
                # Segmentation & tracking for trans images:
                if chan == 0:
                    self.positions[pos].segment(frames=[frame])
                    self.positions[pos].track(frames=[frame])
                    self.positions[pos].features(
                        frames=[frame], features=('length','width','area')
                        )

                # Features extraction for fluorescence images:
                else:
                    self.positions[pos].features(
                        frames=[frame], features=['fluo%d'%(chan,)]
                        )
            

class Position:
    '''
    Position processing object
    '''
    
    def __init__(
            self,
            position_nb,
            reader,
            models, 
            drift_correction=True,
            crop_windows=False
            ):
        '''
        Initialize Position

        Parameters
        ----------
        position_nb : int
            Position index.
        reader : object
            utilities xpreader object.
        models : dict
            U-Net models as loaded by utilities loadmodels().
        drift_correction : bool, optional
            Flag to perform drift correction. The default is True.
        crop_windows : bool, optional
            Flag to crop out windows. The default is False.

        Returns
        -------
        None.

        '''
        self.position_nb = position_nb
        self.reader = reader
        self.models = models
        self.rois = []
        self.drift_values = [[],[]]
        self.drift_correction=drift_correction
        self.crop_windows=crop_windows
        self.verbose = 1
        self._preprocessed = False
        self._pickle_skip = ('reader', 'models', '_pickle_skip')
    
    
    def __getstate__(self):
        '''
        For pickle

        Returns
        -------
        state : dict
            Values to store in pickle file.

        '''
        
        state = dict()
        for (k, v) in self.__dict__.items():
            if k in self._pickle_skip:
                state[k] = None
            else:
                state[k] = v
        
        return state
    
    
    def _msg(self, string):
        '''
        Print timestamped messages

        Parameters
        ----------
        string : str
            Message to print.

        Returns
        -------
        None.

        '''
        
        if self.verbose:
            print('%s, Position %d - '%(time.ctime(), self.position_nb)+string)
    
    
    def preprocess(self, reference=None, rotation_correction=True):
        '''
        Pre-process position (Rotation correction, identify ROIs, 
        initialize drift correction)

        Parameters
        ----------
        reference : 2D array, optional
            Reference image to use to perform pre-processing. If None,
            the first image of each position will be used.
            The default is None.
        rotation_correction : bool, optional
            Flag to perform rotation correction. The default is True.

        Returns
        -------
        None.

        '''
        
        self._msg('Starting pre-processing')
        
        # If no reference frame provided, read first frame from reader:
        if reference is None:
            reference = self.reader.getframes(
                positions=self.position_nb,
                frames = 0,
                channels=0,
                rescale=(0, 1),
                squeeze_dimensions=True
                )
        
        # Estimate rotation:
        if isinstance(rotation_correction, bool):
            if rotation_correction:
                self.rotate = utils.deskew(reference)  # Estimate rotation angle
                reference = utils.imrotate(reference, self.rotate)
            else:
                self.rotate = 0
        else:
            self.rotate = rotation_correction
        
        # Find rois, filter results, get bounding boxes:
        if 'rois' in self.models:
            self.detect_rois(reference)
        else:
            self.rois = [
                ROI(
                    roi_nb=0,
                    box=dict(
                        xtl=0,ytl=0,xbr=reference.shape[1],ybr=reference.shape[0]
                        ),
                    crop_windows=self.crop_windows
                    )
                ]
        
        # Get drift correction template and box
        if self.drift_correction:
            self.drifttemplate = utils.getDriftTemplate(
                [r.box for r in self.rois],
                reference,
                whole_frame=cfg.whole_frame_drift
                )
            self.driftcorbox = dict(
                xtl = 0,
                xbr = None,
                ytl = 0,
                ybr = None if cfg.whole_frame_drift else max(
                    self.rois, key=lambda elem: elem.box['ytl']
                    ).box['ytl']
                )
        
        self._preprocessed = True
    
    
    def detect_rois(self, reference):
        '''
        Use U-Net to detect ROIs (chambers etc...)

        Parameters
        ----------
        reference : 2D array
            Reference image to use to perform pre-processing

        Returns
        -------
        None.

        '''
        
        # Predict
        roismask = self.models["rois"].predict(
                utils.rangescale(
                    cv2.resize(
                        reference,
                        cfg.target_size_rois
                        ),
                    (0,1)
                    )[np.newaxis,:,:,np.newaxis],
                verbose=0
                )
        
        # Clean up:
        roismask = postprocess(
            cv2.resize(np.squeeze(roismask), reference.shape[::-1]),
            min_size=cfg.min_roi_area
            )
        
        # Get boxes
        roisboxes = utils.getROIBoxes(roismask)
        
        # Instanciate ROIs:
        self.roismask = roismask
        for b, box in enumerate(roisboxes):
            self.rois += [ROI(roi_nb=b,box=box,crop_windows=self.crop_windows)]


    def segment(self, frames):
        '''
        Segment cells in all ROIs in position

        Parameters
        ----------
        frames : list of int
            List of frames to run.

        Returns
        -------
        None.

        '''
        
        self._msg('Starting segmentation (%d frames)'%(len(frames),))
        
        # Load trans frames:
        trans_frames = self.reader.getframes(
            positions=self.position_nb,
            channels=0,
            frames=frames,
            rescale=(0, 1),
            rotate=self.rotate
            )
        
        # If trans_frames is 2 dimensions, an extra dimension is added at axis=0 for time
        # (1 timepoint may cause this issue)
        if trans_frames.ndim == 2:
            trans_frames = trans_frames[np.newaxis,:,:]
            
        # Correct drift:
        if self.drift_correction:
            trans_frames, self.drift_values = utils.driftcorr(
                trans_frames,
                template=self.drifttemplate,
                box=self.driftcorbox
                )
        
        # Run through frames and ROIs and compile segmentation inputs:
        x = []
        references=[]
        for f, img in enumerate(trans_frames):
            for r, roi in enumerate(self.rois):
                inputs, windows = roi.get_segmentation_inputs(img)
                x+=[inputs]
                references+=[[r,len(x[-1]),f,windows]]
        x = np.concatenate(x)
        
        # Run segmentation model:
        y = self.models['segmentation'].predict(
            x,
            batch_size=4
            )
        
        # Dispatch segmentation outputs to rois:
        i = 0
        for ref in references:
            self.rois[ref[0]].process_segmentation_outputs(
                y[i:i+ref[1]], frame=ref[2], windows=ref[3]
                )
            i+=ref[1]


    def track(self, frames):
        '''
        Track cells in all ROIs in frames

        Parameters
        ----------
        frames : list of int
            List of frames to run.

        Returns
        -------
        None.

        '''
        
        self._msg('Starting tracking (%d frames)'%(len(frames),))
        
        for f in frames:
            
            self._msg('Tracking - frame %d/%d '%(f,len(frames,)))
            
            # Compile inputs and references:
            x=[]
            references=[]
            for r, roi in enumerate(self.rois):
                inputs, boxes = roi.get_tracking_inputs(frame=f)
                if inputs is not None:
                    x+=[inputs]
                    references+=[[r,len(x[-1]),f,boxes]]
            
            # Predict:
            if len(x)>0:
                y = self.models['tracking'].predict(
                    np.concatenate(x),
                    batch_size=16,
                    workers=1,
                    use_multiprocessing=False,
                    verbose=0
                    )
            
            # Dispatch tracking outputs to rois:
            i = 0
            for ref in references:
                self.rois[ref[0]].process_tracking_outputs(
                    y[i:i+ref[1]], frame=ref[2], boxes=ref[3]
                    )
                i+=ref[1]
        
        
    def features(
            self,frames,features=('length','width','area','perimeter','edges')
            ):
        '''
        Extract features for all ROIs in frames

        Parameters
        ----------
        frames : list of int
            List of frames to run.
        features : list of str, optional
            List of features to extract.
            The default is ('length','width','area').

        Returns
        -------
        None.

        '''
        
        self._msg('Starting feature extraction (%d frames)'%(len(frames),))
        
        # Check if fluo channels are requested in features (fluo1, fluo2, etc):
        fluo_channels = [int(x[4:]) for x in features if x[0:4]=='fluo']
        
        # Load fluo images if any:
        if len(fluo_channels):
            # Read fluorescence frames:
            fluo_frames = self.reader.getframes(
                positions=self.position_nb,
                channels=fluo_channels,
                frames=frames,
                squeeze_dimensions=False,
                rotate=self.rotate
                )[0]
            # Apply drift correction
            if self.drift_correction:
                for f in range(fluo_frames.shape[1]):
                    fluo_frames[:, f, :, :], _ = utils.driftcorr(
                        fluo_frames[:, f, :, :], drift=self.drift_values
                        )
        else:
            fluo_frames=None
        
        # Run through frames:
        for f in frames:
            self._msg('Feature extraction - frame %d/%d '%(f,len(frames,)))
            # Run through ROIs and extract features:
            for roi in self.rois:
                roi.extract_features(
                    frame=f,
                    fluo_frames=fluo_frames[f] if fluo_frames is not None else None,
                    features=features
                    )


    def save(self,filename=None,frames=None,save_format=('pickle','movie')):
        '''
        Save to disk

        Parameters
        ----------
        filename : str or None, optional
            File name for save file. If None, the file will be saved to
            PositionXXXXXX in the current directory.
            The default is None.
        frames : list of int or None, optional
            List of frames to save in movie. If None, all frames are run.
            The default is None.
        save_format : tuple of str, optional
            Formats to save the data to. Options are 'pickle', 'legacy' (ie
            Matlab format), and 'movie' for saving an mp4 movie.
            The default is ('pickle', 'movie').

        Returns
        -------
        None.

        '''
        
        if filename is None:
            filename = './Position%06d'%(self.position_nb,)
        else:
            os.path.splitext(filename)[0] # remove extension if any
        
        
        if 'legacy' in save_format:
            self._msg('Saving to legacy format\n%s'%filename+'.mat')
            utils.legacysave(self, filename+'.mat')
        
        if 'pickle' in save_format:
            self._msg('Saving to pickle format\n%s'%filename+'.pkl')
            import pickle
            pickle.dump(self, open(filename+'.pkl','wb'))
        
        if 'movie' in save_format:
            self._msg('Saving results movie\n%s'%filename+'.mp4')
            movie = utils.results_movie(self, frames=frames)
            utils.vidwrite(movie, filename+'.mp4', verbose=False)
    
    
    def load(self, filename):
        '''
        Load position from pickle file

        Parameters
        ----------
        filename : str or None, optional
            File name for save file.

        Returns
        -------
        None.

        '''
        
        p = utils.load_position(filename)
        
        for (k, v) in p.__dict__.items():
            if k not in self._pickle_skip:
                exec('self.%s = v'%(k,))
    
    
    def clear(self):
        '''
        Clear Position-specific variables from memory (can be loaded back with 
        load())

        Returns
        -------
        None.

        '''
        
        self._msg('Clearing variables from memory')
        for k in self.__dict__.keys():
                if k not in self._pickle_skip:
                    exec('self.%s = None'%(k,))
        
        
class ROI:
    '''
    ROI processor object
    '''
    
    def __init__(self, roi_nb, box, crop_windows=False):
        '''
        Initialize ROI

        Parameters
        ----------
        roi_nb : int
            ROI index.
        box : dict
            Crop box for ROI, formatted as in the utilities.py cropbox 
            function input dict.
        crop_windows : bool, optional
            Flag to crop and stitch back windows for segmentation and tracking.
            The default is False.

        Returns
        -------
        None.

        '''
        self.roi_nb = roi_nb
        self.box = box
        self.img_stack = []
        self.seg_stack = []
        self.lineage = utils.Lineage()
        self.label_stack = []
        self.crop_windows = crop_windows
        self.verbose = 1
        
        if crop_windows:
            self.scaling = None
        else:
            self.scaling = (
                (box['ybr']-box['ytl'])/cfg.target_size_seg[0],
                (box['xbr']-box['xtl'])/cfg.target_size_seg[1]
                )
        
    
    def get_segmentation_inputs(self, img):
        '''
        Compile segmentation inputs for ROI

        Parameters
        ----------
        img : 2D array
            Single frame to crop and send for segmentation.

        Returns
        -------
        x : 4D array
            Segmentation input array. Dimensions are 
            (windows, size_y, size_x, 1).
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        '''
        
        # Crop and scale:
        i = utils.rangescale(utils.cropbox(img, self.box),rescale=(0,1))
        # Append i as is to input images stack:
        self.img_stack.append(i)
        
        if self.crop_windows:
            # Crop out windows:
            x, windows_y, windows_x = utils.create_windows(
                i, target_size=cfg.target_size_seg
                )
            windows = (windows_y, windows_x)
            # Shape x to expected format:
            x = x[:,:,:,np.newaxis]
            
        else: 
            # Resize to unet input size
            x = cv2.resize(i,dsize=cfg.target_size_seg[::-1])
            windows = None
            # Shape x to expected format:
            x = x[np.newaxis,:,:,np.newaxis]
        
        return x, windows
    
    
    def process_segmentation_outputs(self, y, frame=None, windows=None):
        '''
        Process outputs after they have been segmented.

        Parameters
        ----------
        y : 4D array
            Segmentation output array. Dimensions are 
            (windows, size_y, size_x, 1).
        frame : int or None, optional
            Frame index. If None, this is considered the latest frame's output.
            The default is None.
        windows : tuple of 2 lists
            y and x coordinates of crop windows if any, or None.

        Returns
        -------
        None.

        '''
        
        # Stitch windows back together (if needed):
        if windows is None:
            y = y[0,:,:,0]
        else:
            y = utils.stitch_pic(y[...,0], windows[0], windows[1])
        
        # Binarize:
        y = (y>.5).astype(np.uint8)
        # Crop out segmentation if image was smaller than target_size
        y = y[:self.img_stack[0].shape[0],:self.img_stack[0].shape[1]]
        # Area filtering:
        y = utils.opencv_areafilt(y, min_area=cfg.min_cell_area)
        
        # Append to segmentation results stack:
        if frame is None:
            self.seg_stack.append(y)
        else:
            if len(self.seg_stack) <= frame:
                self.seg_stack+=[None for _ in range(frame+1-len(self.seg_stack))] # Extend list
            self.seg_stack[frame] = y
    
    
    def get_tracking_inputs(self, frame=None):
        '''
        Compile tracking inputs for ROI

        Parameters
        ----------
        frame : int, optional
            The frame to compile for. If None, the earliest frame not yet
            tracked is run.
            The default is None.

        Raises
        ------
        RuntimeError
            Segmentation has not been completed up to frame yet.

        Returns
        -------
        x : 4D array or None
            Tracking input array. Dimensions are (previous_cells, 
            cfg.target_size_track[1], cfg.target_size_track[0], 4). If no 
            previous cells to track from (e.g. first frame or glitch), None is
            returned.
        boxes : List of dict or None
            Crop and fill boxes to re-place outputs in the ROI

        '''
        
        # If no frame number passed, run latest:
        if frame is None:
            frame=len(self.lineage.cellnumbers)
        
        # Check if segmentation data is ready:
        if len(self.seg_stack)<=frame:
            raise RuntimeError("Segmentation incomplete - frame %d"%(frame,))
        
        # If no previous cells to track from, directly update lineage:
        if frame == 0 \
            or len(self.lineage.cellnumbers) < frame \
                or len(self.lineage.cellnumbers[frame-1])==0:
                    
            # Cell poles
            poles = utils.getpoles(self.seg_stack[frame],scaling=self.scaling)
            
            # Create new orphan cells
            for c in range(len(poles)):
                self.lineage.update(None, frame, attrib=[c], poles=[poles[c]])
            
            return None, None
        
        # Otherwise, get cell contours:
        cells, _ = cv2.findContours(
            self.seg_stack[frame-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        cells.sort(key=lambda elem: np.max(elem[:,0,1])) # Sorting along Y
        
        # Allocate empty tracking inputs array:
        x = np.empty(
            shape=(len(cells),)+cfg.target_size_track+(4,),
            dtype=np.float32
            )
        
        # Run through contours and compile inputs:
        boxes = []
        for c, cell in enumerate(cells):
            
            if self.crop_windows:
                curr_img = self.img_stack[frame]
                prev_img = self.img_stack[frame-1]
                # Cell-centered crop boxes:
                cb, fb = utils.gettrackingboxes(cell, curr_img.shape)
                draw_offset = (-cb['xtl']+fb['xtl'],-cb['ytl']+fb['ytl'])
            else:
                curr_img = cv2.resize(
                    self.img_stack[frame],dsize=cfg.target_size_seg[::-1]
                    )
                prev_img = cv2.resize(
                    self.img_stack[frame-1],dsize=cfg.target_size_seg[::-1]
                    )
                cb = fb = dict(ytl=None, xtl=None, ybr=None, xbr=None)
                draw_offset=None
            boxes += [(cb, fb)]
            
            # Current image
            x[c,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],0] = utils.cropbox(
                curr_img, cb
                )
            
            # Segmentation mask of one previous cell (seed)
            x[c,:,:,1] = cv2.drawContours(
                np.zeros(cfg.target_size_track, dtype=np.float32),
                [cell],
                0,
                offset=draw_offset,
                color=1.,
                thickness=-1
                )
            
            # Previous image
            x[c,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],2] = utils.cropbox(
                prev_img, cb
                )
            
            # Segmentation of all current cells
            x[c,fb['ytl']:fb['ybr'],fb['xtl']:fb['xbr'],3] = utils.cropbox(
                self.seg_stack[frame], cb
                )
        
        # Return tracking inputs and crop and fill boxes:
        return x, boxes
    
    
    def process_tracking_outputs(self, y, frame=None, boxes=None):
        '''
        Process output from tracking U-Net

        Parameters
        ----------
        y : 4D array
            Tracking output array. Dimensions are (previous_cells, 
            cfg.target_size_track[1], cfg.target_size_track[0], 1).
        frame : int, optional
            The frame to process for. If None, the earliest frame not yet
            tracked is run.
            The default is None.
        boxes : List of dict or None
            Crop and fill boxes to re-place outputs in the ROI

        Returns
        -------
        None.

        '''
        
        if frame is None:
            frame = len(self.lineage.cellnumbers)
        
        # Get scores and attributions:
        labels = utils.label_seg(self.seg_stack[frame])
        scores = utils.getTrackingScores(
            labels, y[:,:,:,0], boxes=boxes if self.crop_windows else None
            )
        if scores is None:
            self.lineage.update(None, frame)
            return
        attributions = utils.getAttributions(scores)
        
        # Get poles:
        poles = utils.getpoles(self.seg_stack[frame],labels,scaling=self.scaling)
        
        # Update lineage:
        # Go through old cells:
        for o in range(attributions.shape[0]):
            attrib = attributions[o,:].nonzero()[0]
            new_cells_poles = []
            for n in attrib:
                new_cells_poles+=[poles[n]]
            self.lineage.update(o, frame, attrib=attrib, poles=new_cells_poles)
        # Go through "orphan" cells:
        for n in range(attributions.shape[1]):
            attrib = attributions[:,n].nonzero()[0]
            new_cells_poles=[poles[n]]
            if len(attrib)==0:
                self.lineage.update(None, frame, attrib=[n], poles=new_cells_poles)
    

    def extract_features(
        self,
        frame=None,
        fluo_frames=None,
        features=('length','width','area','perimeter','edges')
        ):
        '''
        Extract single cell features

        Parameters
        ----------
        frame : int, optional
            The frame to extract for. If None, the earliest frame not yet
            extracted is run.
            The default is None.
        fluo_frames : 3D array, optional
            Fluorescent images to extract fluo from. Dimensions are
            (channels, size_y, size_x).
            The default is None.
        features : list of str, optional
            Features to extract. Options are ('length','width','area','fluo1',
            'fluo2','fluo3'...)
            The default is ('length','width','area').

        Returns
        -------
        None.

        '''
        
        # Default frame:
        if frame is None:
            frame = len(self.label_stack)
        
        # Add Nones to label stack list if not long enough:
        if len(self.label_stack)<=frame:
            self.label_stack+=[None for _ in range(frame+1-len(self.label_stack))] # Initialize
        
        # Compile labels frame:
        if self.label_stack[frame] is None:
            if len(self.lineage.cellnumbers)<=frame:
                cell_nbs = []
            else:
                cell_nbs = [c+1 for c in self.lineage.cellnumbers[frame]]
            labels = utils.label_seg(self.seg_stack[frame],cell_nbs)
            
            if self.crop_windows:
                self.label_stack[frame] = labels
            else:
                resize = (
                    self.box["xbr"]-self.box["xtl"],
                    self.box["ybr"]-self.box["ytl"]
                    )  
                self.label_stack[frame] = cv2.resize(
                    labels, resize, interpolation=cv2.INTER_NEAREST
                    )
        
        # Get cells in resized frame:
        cells, contours = utils.getcellsinframe(
            self.label_stack[frame], return_contours=True
            )

        # Run through cells in resized frame:
        for cell, cnt in zip(cells, contours):
            
            # Mark cell if it touches the edges of the ROI:
            if 'edges' in features:
                edge_str = ''
                if any(cnt[:,0,0]==0):
                    edge_str+='-x'
                if any(cnt[:,0,0]==self.label_stack[frame].shape[1]-1):
                    edge_str+='+x'
                if any(cnt[:,0,1]==0):
                    edge_str+='-y'
                if any(cnt[:,0,1]==self.label_stack[frame].shape[0]-1):
                    edge_str+='+y'
                self.lineage.setvalue(cell,frame,'edges',edge_str)
                
            
            # Morphological features:
            if 'length' in features or 'width' in features:
                rotrect = cv2.minAreaRect(cnt)
                if 'length' in features:
                    self.lineage.setvalue(cell,frame,'length',max(rotrect[1]))
                if 'width' in features:
                    self.lineage.setvalue(cell,frame,'width',min(rotrect[1]))
            if 'area' in features:
                self.lineage.setvalue(cell,frame,'area',cv2.contourArea(cnt))
            if 'perimeter' in features:
                self.lineage.setvalue(cell,frame,'perimeter',cnt.shape[0])
            
            # Fluorescence:
            fluo_features = [x for x in features if x[0:4]=='fluo']
            if len(fluo_features)>0:
                # Pixels where the cell is:
                pixels = np.where(self.label_stack[frame] == cell+1)
            for f, fluostr in enumerate(fluo_features):
                # Compute mean fluo value for the cell:
                value = np.mean(fluo_frames[
                    f,
                    pixels[0]+self.box['ytl'] if self.box is not None else pixels[0],
                    pixels[1]+self.box['xtl'] if self.box is not None else pixels[1]
                    ])
                # Update cell:
                self.lineage.setvalue(cell,frame,fluostr,value)


if __name__ == "__main__":
    
    # This is weird, but it's the only work-around I found to pickle
    # looking for classes in the __main__ namespace when re-loading:
    from pipeline import Pipeline

    # Init reader:
    xpreader = utils.xpreader()
    
    # Init pipeline:
    xp = Pipeline(xpreader)
    
    # Run it:
    xp.process()