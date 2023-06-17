import nitools as nt
import nibabel as nb
import trimesh.triangles as tri
import numpy as np
def test_border():
    wdir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fs_LR_32/'
    fname = wdir + 'fs_LR.32k.L.border'
    sname = wdir + 'fs_LR.32k.R.flat.surf.gii'
    borders = nt.read_borders(fname)
    surf = nb.load(sname)
    coords = borders[0].get_coords(surf)
    pass

def project_border(XYZ,surf):
    V,F=surf.agg_data()
    triangles = V[F,:]
    N = triangles.shape[0]
    P = XYZ.shape[0]
    vertices = np.zeros((P,3),dtype=int)
    weights = np.zeros((P,3))
    for n in range(P):
        bary=tri.points_to_barycentric(triangles,np.tile(XYZ[n,:],(N,1)))   
        idx=np.where(np.all(np.logical_and(bary>=0,bary<=1),axis=1))[0][0]
        vertices[n,:] = F[idx,:]
        weights[n,:] = bary[idx,:]
    return vertices,weights

if __name__=="__main__":
    wdir = '/Users/jdiedrichsen/Python/SUITPy/SUITPy/surfaces/'
    
    surf = nb.load(wdir+'FLAT.surf.gii')
    borders = np.genfromtxt(wdir+'borders.txt', delimiter=',')
    b2,b_info= nt.read_borders(wdir + 'fissures_flat.border')
    v,w = project_border(borders,surf)
    i = 0
    for b in b2:
        N = b.vertices.shape[0]
        b.vertices = v[i:i+N,:]
        b.weights = w[i:i+N,:]
        i = i+N
    nt.save_borders(b2,b_info,wdir+'fissures2_flat.border')
    pass 