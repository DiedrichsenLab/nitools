"""Test script for different functions of CIFTI nitools
    using the Working memory example data
"""
import nibabel as nb
import numpy as np
import nitools as nt

stdmesh_dir = "/Users/jdiedrichsen/Python/surfAnalysisPy/standard_mesh"
atlas_dir = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32"

def split_cifti():
    A = nb.load(stdmesh_dir + '/fs_L/fs_LR.32k.LR.sulc.dscalar.nii')
    G=nt.split_cifti_to_giftis(A,type='func')
    nb.save(G[0],stdmesh_dir + '/fs_L/fs_LR.32k.L.shape.gii') 
    nb.save(G[1],stdmesh_dir + '/fs_R/fs_LR.32k.R.shape.gii') 

if __name__ == '__main__':
    # make_func_cifti()
    split_cifti()
    pass