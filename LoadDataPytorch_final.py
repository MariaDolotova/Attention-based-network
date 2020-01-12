'''
DataGenerators and additinal functions for patch generation 
    
'''

import numpy as np
from skimage import io



PATH='path_to_your_data'
PATCH_SIZE=224


from torch.utils import data


def generate_whole_image_patches(x, height1, width1, size=64):
    height, width, depth = x.shape
    patches = []

    for i in range(height // size):
        for j in range(width // size):
            patch = x[i*size:(i+1)*size, j*size:(j+1)*size, :]
            patch = np.array(patch)
            patches.append(patch)
    patches = np.array(patches)
    return patches


class Dataset(data.Dataset):
    def __init__(self, name, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.name=name


    def __len__(self):
        'Denotes the number of batches per epoch'
        
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]

        img_id=ID.split('_')
        img_id=str(img_id[0])+'_'+str(img_id[1])
        img=io.imread(str(PATH)+str(img_id)+'.png')
        height_new=img.shape[0]//224
        width_new=img.shape[1]//224
        img= img[0:height_new*224, 0:width_new*224,:]
        patches=generate_whole_image_patches(img, height_new, width_new, size=224)
        X = np.empty((1, len(patches), 3, 224, 224))
        y = np.empty(1, dtype=int)
        X = patches
        X=np.rollaxis(X, 3, 0)
        X1 = np.empty((512, height_new, width_new))
  
        y = self.labels[ID]
    

        return  {'img_id': img_id, 'img_input': X, 'dim_input': X1}, y