# Testing of SpmGlm class
from nitools.spm import SpmGlm
import nitools as nt
import nibabel as nb
import numpy as np
import os
import timeit

def test_sampler():
    basedir = '/Users/jdiedrichsen/Dropbox/projects/rsa_example_dataset'
    spm = SpmGlm(basedir + '/glm_firstlevel')
    spm.get_info_from_spm_mat()
    [beta,resms,info] = spm.get_betas(basedir + '/anat/subcortical_mask.nii')
    [residuals,beta,info] = spm.get_residuals(basedir + '/anat/subcortical_mask.nii')
    coords = nt.get_mask_coords(basedir + '/anat/subcortical_mask.nii')
    fnames = fnames=[f.split(',')[0] for f in spm.rawdata_files]
    data = nt.sample_images(np.unique(fnames),coords)

def test_sample_speed(): 
    basedir = '/Users/jdiedrichsen/Dropbox/projects/rsa_example_dataset'
    spm = SpmGlm(basedir + '/glm_firstlevel')
    spm.get_info_from_spm_mat()

    coords = nt.get_mask_coords(basedir + '/anat/subcortical_mask.nii')
    for i in range(10):
        t1 = timeit.default_timer()
        data = nt.sample_images(np.unique(spm.rawdata_files),coords,use_dataobj=True)
        t2 = timeit.default_timer() 
        data = nt.sample_images(np.unique(spm.rawdata_files),coords,use_dataobj=False)
        t3 = timeit.default_timer() 
        print(f"Time for first run: {t2-t1}")
        print(f"Time for second run: {t3-t2}")

def test_GLM():
    basedir = '/Users/jdiedrichsen/Dropbox/projects/rsa_example_dataset'
    spm = SpmGlm(basedir + '/glm_firstlevel')
    spm.get_info_from_spm_mat()
    [beta,resms,info] = spm.get_betas(basedir + '/anat/subcortical_mask.nii')
    [residuals,beta1,info] = spm.get_residuals(basedir + '/anat/subcortical_mask.nii')
    assert np.allclose(beta,beta1,atol=1e-4,equal_nan=True),'betas are not the same'
    resms1 = np.mean(residuals**2,axis=0)/spm.eff_df
    assert np.allclose(resms,resms1,atol=1e-4,equal_nan=True),'ResMS are not the same'

def test_relative_paths():
    # If the data has been moved, we should adapt the paths stored in SPM.mat
    basedir = os.path.expanduser('~/data/project')
    spm = SpmGlm(basedir + '/glm_firstlevel')
    spm.get_info_from_spm_mat()
    assert spm.rawdata_files[0] == basedir+'/func/uas01_run01.nii,1  '

if __name__=='__main__':
    test_GLM()
    pass
