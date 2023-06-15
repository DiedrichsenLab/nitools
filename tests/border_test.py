import nitools as nt
import nibabel as nb

if __name__=="__main__":
    wdir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32/'
    fname = wdir + 'fs_LR.32k.L.border'
    sname = wdir + 'fs_LR.32k.R.flat.surf.gii'
    borders = nt.read_borders(fname)
    surf = nb.load(sname)
    coords = borders[0].get_coords(surf)
    pass
