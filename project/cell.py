import numpy as np

def euc_dist(arr1, arr2):
    return(np.linalg.norm(arr1 - arr2))

def reindex_cell_labels(img):
    idx = np.unique(img)
    for i, x in enumerate(idx):
        img[img == x] = i
    return(img)

class Cell:
    def __init__(self, label, x_m=None, y_m=None, size=None):
        self.label = label
        self.pxl_pos = None 
        self.x_m = x_m
        self.y_m = y_m
        self.size = size
        self.dist2board = None
        self.c_pos = None
        self.nn_t = [None, 1000]
        self.nn_tp1 = [None, 1000]
        self.self_tp1 = [None, 1000]
        self.nn_tp2 = [None, 1000]
        self.self_tp2 = [None, 1000]
        
    def dist_to_board(self, img):
        self.dist2board = min(min(abs(self.y_m - 0), abs(self.y_m - img.shape[1])),
                              min(abs(self.x_m - 0), abs(self.x_m - img.shape[0])) )
        
    def get_pos_from_img(self, img):
        self.pxl_pos = np.argwhere(img == self.label)
        self.x_m = int(np.mean(self.pxl_pos[:,0]))
        self.y_m = int(np.mean(self.pxl_pos[:,1]))
        self.c_pos = np.array([self.x_m, self.y_m])
        self.size = len(self.pxl_pos[:,0])
        
    def find_NN(self, cellc):
        n_cells = len(cellc.cells)
        for m, lab in enumerate(cellc.cells_labels):
            if lab != self.label:
                d = euc_dist(self.c_pos, cellc.cells[m].c_pos)
                if d < self.nn_t[1]:
                    self.nn_t[0] = lab
                    self.nn_t[1] = d
                    
    def find_NN_otherFrame(self, cellc, timepoint='tp1'):
        n_cells = len(cellc.cells)
        for m, lab in enumerate(cellc.cells_labels):
            d = euc_dist(self.c_pos, cellc.cells[m].c_pos)
            if timepoint == 'tp1':
                if d < self.self_tp1[1]:
                    self.self_tp1[0] = lab
                    self.self_tp1[1] = d
            elif timepoint == 'tp2':
                if d < self.self_tp2[1]:
                    self.self_tp2[0] = lab
                    self.self_tp2[1] = d
                    
        for m, lab in enumerate(cellc.cells_labels):
            d = euc_dist(self.c_pos, cellc.cells[m].c_pos)
            if timepoint == 'tp1':
                if lab != self.self_tp1[0]:
                    if d < self.nn_tp1[1]:
                        self.nn_tp1[0] = lab
                        self.nn_tp1[1] = d
            elif timepoint == 'tp2':
                if lab != self.self_tp2[0]:
                    if d < self.nn_tp2[1]:
                        self.nn_tp2[0] = lab
                        self.nn_tp2[1] = d
                
class CellContainer:
    def __init__(self, img):
        self.cells = []
        self.cells_x = []
        self.cells_y = [] 
        self.cells_labels = []  
        self.n_c = 0
        self.init_from_img(img)

    def init_from_img(self, img):
        cells_idx = np.unique(img)
        cells_idx = cells_idx[cells_idx!=0]
        for i, idx in enumerate(cells_idx):
            cell = Cell(idx) # create a cell with a label
            cell.get_pos_from_img(img) # get info about pixel, size and center of cell
            cell.dist_to_board(img) # calculate minimal distance to image boarders
            self.add_cell(cell)
    
    def add_cell(self, cell):
        self.cells.append(cell)
        self.cells_x.append(cell.x_m)
        self.cells_y.append(cell.y_m)
        self.cells_labels.append(cell.label)
        self.n_c+=1

    def find_NN(self):
        for i, idx in enumerate(self.cells_labels):
            self.cells[i].find_NN(self)
            
    def find_NN_other_frame(self, cc_next, timepoint = 'tp1'):
        for i, idx in enumerate(self.cells_labels):
            self.cells[i].find_NN_otherFrame(cc_next, timepoint)

    def score_surface(self, seg_true, seg_pred):
        mean_score=0
        nbr_zero_score=0
        hit, n_cells_considered = 0, 0
        for i, lab_train in enumerate(self.cells_labels):
            # ignore small cells 
            if self.cells[i].size < 20:
                next
            n_cells_considered+=1
            
            lab_test=self.cells[i].self_tp1[0] #label of the test image

            seg_pred_1c = seg_pred.copy()
            seg_pred_1c[seg_pred_1c!=lab_test]=0 #only the cell is different than 0
            seg_pred_1c[seg_pred_1c==lab_test]=1 #binary mask for the cell

            seg_true_1c=seg_true.copy()
            seg_true_1c[seg_true_1c!=lab_train]=0 
            seg_true_1c[seg_true_1c==lab_train]=1 #binary mask 

            sum_lab = seg_true_1c+seg_pred_1c 

            common_surface = sum_lab.copy()
            common_surface[sum_lab<=1] = 0
            nb_union = np.count_nonzero(sum_lab.copy())
            nb_common_surface = np.count_nonzero(common_surface)
            score = nb_common_surface/nb_union
            
            if nb_common_surface == 0: #if the commmon surface is null, then we don't consider this pair of cells.
                nbr_zero_score+=1
            mean_score+=score
            
            # Second metric of accuracy
            true_surface = np.count_nonzero(seg_true_1c)
            pred_surface = np.count_nonzero(seg_pred_1c)
            ratio_true = nb_common_surface / true_surface
            ratio_pred = nb_common_surface / pred_surface
            if ratio_true > 0.51 and ratio_pred > 0.51:
                hit+=1 

        mean_score=mean_score/(n_cells_considered+1-nbr_zero_score)
        ratio_hit = hit / (n_cells_considered+1-nbr_zero_score)
        return mean_score, ratio_hit

