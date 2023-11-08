"""Test script for different functions of CIFTI nitools
    using the Working memory example data
"""
import nibabel as nb
import numpy as np
import nitools as nt

stdmesh_dir = "/Users/jdiedrichsen/Python/surfAnalysisPy/standard_mesh"
atlas_dir = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates"

def split_cifti1():
    A = nb.load(stdmesh_dir + '/fs_L/fs_LR.32k.LR.sulc.dscalar.nii')
    G=nt.split_cifti_to_giftis(A,type='func')
    nb.save(G[0],stdmesh_dir + '/fs_L/fs_LR.32k.L.shape.gii')
    nb.save(G[1],stdmesh_dir + '/fs_R/fs_LR.32k.R.shape.gii')

def split_cifti2():
    A = nb.load(atlas_dir + '/fsaverage6/atl-MSHBM_Prior_15_fsaverage6/MSHBM_Prior_15_fsaverage6.dlabel.nii')
    G=nt.split_cifti_to_giftis(A)
    nb.save(G[0],atlas_dir + '/fsaverage6/atl-MSHBM_Prior_15_fsaverage6/MSHBM_Prior_15_fsaverage6.L.label.gii')
    nb.save(G[1],atlas_dir + '/fsaverage6/atl-MSHBM_Prior_15_fsaverage6/MSHBM_Prior_15_fsaverage6.R.label.gii')



if __name__ == '__main__':
    # make_func_cifti()
    split_cifti2()
    pass