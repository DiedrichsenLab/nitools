import nitools as nt
import nibabel as nb
import trimesh.triangles as tri
import bezier as bz
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def test_border():
    """ Test the border functions"""
    wdir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32/'
    fname = wdir + 'fs_LR.32k.L.border'
    sname = wdir + 'fs_LR.32k.R.flat.surf.gii'
    borders = nt.read_borders(fname)
    surf = nb.load(sname)
    coords = borders[0].get_coords(surf)
    pass

def project_suit_borders():
    """ Project the suit xyz borders onto the flat surface
    """
    wdir = '/Users/jdiedrichsen/Python/SUITPy/SUITPy/surfaces/'

    surf = nb.load(wdir+'FLAT.surf.gii')
    borders = np.genfromtxt(wdir+'borders.txt', delimiter=',')
    b2,b_info= nt.read_borders(wdir + 'fissures_flat.border')
    v,w = nt.project_border(borders,surf)
    i = 0
    for b in b2:
        N = b.vertices.shape[0]
        b.vertices = v[i:i+N,:]
        b.weights = w[i:i+N,:]
        i = i+N
    nt.save_borders(b2,b_info,wdir+'fissures2_flat.border')

def test_resample_border():
    """ Test the resample border function"""
    wdir = '/Users/jdiedrichsen/Python/SurfAnalysisPy/standard_mesh/fs_L/'
    border,b_info = nt.read_borders(wdir+'fs_LR.32k.L.border')
    surf = nb.load(wdir+'fs_LR.32k.L.sphere.surf.gii')
    newborder = deepcopy(border)
    for i,b in enumerate(border):
        newborder[i].vertices,newborder[i].weights,coord = resample_border(b,surf)

    nt.save_borders(newborder,b_info,wdir+'fs_LR.32k.L_resampled.border')

if __name__=="__main__":
    project_suit_borders()
    # test_resample_border()