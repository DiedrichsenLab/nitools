"""Test script for different functions of CIFTI nitools
    using the Working memory example data
"""
import nibabel as nb
import numpy as np
import nitools as nt

base_dir = "/Users/jdiedrichsen/data/vol_test"



if __name__ == '__main__':
    # make_func_cifti()
    nt.change_nifti_numformat(base_dir+"/sub-8_task-task_run-01_bold.nii",base_dir+"/new_bold.nii")
    pass