# Testing of SpmGlm class
from nitools.spm import SpmGlm
import numpy as np
import nibabel as nb
import os

if __name__=='__main__':
    spm = SpmGlm('/Users/jdiedrichsen/Data/fivedigitFreq2/GLM_firstlevel/s01/glm3')
    spm.get_info_from_spm_mat()
    pass 