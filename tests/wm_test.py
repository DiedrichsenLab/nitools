"""Test script for different functions of CIFTI nitools
    using the Working memory example data
"""
import nibabel as nb
import numpy as np
import nitools as nt

base_dir = "/Users/jdiedrichsen/data/wm_cerebellum"
atlas_dir = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32"

def make_func_cifti():
    hem = ['L','R']
    hem_name = ['cortex_left','cortex_right']
    Data = []
    bm = []
    for i,h in enumerate(hem):
        X = []
        names = []
        for cond in ['encode','retriev']:
            filen = f"{base_dir}/wgroup.{cond}-rest.{h}.func.gii"
            G = nb.load(filen)
            a = nt.get_gifti_anatomical_struct(G)
            names = names + nt.get_gifti_column_names(G)
            X.append(nt.get_gifti_data_matrix(G))
        Data.append(np.concatenate(X,axis=1))
        N = X[0].shape[0]
        bm.append(nb.cifti2.BrainModelAxis.from_mask(np.ones(N,),
                                            name=hem_name[i]))
    row_axis = nb.cifti2.ScalarAxis(names)
    header = nb.Cifti2Header.from_axes((row_axis,bm[0]+bm[1]))
    Data=np.concatenate(Data,axis=0)
    cifti_img = nb.Cifti2Image(dataobj=Data.T,header=header)
    nb.save(cifti_img,base_dir + '/wgroup.dscalar.nii')
    pass

def make_parcel_gifti():
    pass

if __name__ == '__main__':
    make_func_cifti()