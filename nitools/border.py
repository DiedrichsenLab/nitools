""" Other neuroimaging file formats
"""
import numpy as np
import nibabel as nb
import xml.etree.ElementTree as ET
import trimesh.triangles as tri
import bezier as bz
from copy import deepcopy

class Border:
    """Border class

    Attributes:
        name (str): Name of the border
        vertices (np.array): Vertices of the border
        weights (np.array): Weights of Barycentric coordinates
    """
    def __init__(self,name=None,vertices=None,weights=None):
        self.name= name
        self.vertices = vertices
        self.weights = weights

    def get_coords(self,surf):
        """Gets coords the border onto a surface
        Barycentric coordinates are transformed into Euclidean coordinates

        Args:
            surf (str or nibabel): Surface file
        Returns:
            coords (np.array): Coordinates of the border on the surface
        """
        if isinstance(surf,str):
            surf = nb.load(surf)
        coordV = surf.agg_data('NIFTI_INTENT_POINTSET')
        coords = coordV[self.vertices,:]
        weights = self.weights / np.sum(self.weights,axis=1)[:,np.newaxis]
        weights = weights.reshape(-1,3,1)
        coords = np.sum(coords*weights,axis=1)
        return coords

def read_borders(fname):
    """ Reads a Workbench border file
    from XML format into a list of Border objects

    Args:
        fname (str): Filename

    Returns:
        borders (list): List of border objects
        binfo (dict): Dictionary of border-file information
    """
    borders = []
    tree = ET.parse(fname)
    root = tree.getroot()
    for bclass in root:
        for bn,border in enumerate(bclass):
            for i,bp in enumerate(border):
                V = bp.findall('Vertices')[0]
                W = bp.findall('Weights')[0]
                vert = np.fromstring(V.text,sep=' ',dtype=int).reshape(-1,3)
                weights = np.fromstring(W.text,sep=' ').reshape(-1,3)
                bb=Border(border.get('Name'),vert,weights)
                bb.num = bn
                bb.partnum = i
                borders.append(bb)
    return borders,root.attrib

def save_borders(borders,binfo,fname):
    """Saves a list of borders to a connectome workbench border file

    Args:
        borders (list): List of border objects
        binfo (dict): Dictionary of border-file information
        fname (str): filename
    """
    root = ET.Element("BorderFile",attrib=binfo)
    bclass = ET.SubElement(root, "Class", attrib={'Name':"Class1",
                                                  "Red":"0",
                                                  "Green":"0",
                                                  "Blue":"0"})
    current_name = ''
    for b in borders:
        if b.name !=current_name:
            bo=ET.SubElement(bclass, "Border", attrib={'Name':b.name,
                                                  "Red":"0",
                                                  "Green":"0",
                                                  "Blue":"0"})
            current_name = b.name
        bp=ET.SubElement(bo, "BorderPart", attrib={'Closed':'False'})
        V=ET.SubElement(bp, "Vertices")
        V.text = np.array2string(b.vertices).replace('[','')
        V.text = V.text.replace(']','')
        W=ET.SubElement(bp, "Weights")
        W.text = np.array2string(b.weights).replace('[','')
        W.text = W.text.replace(']','')

    tree = ET.ElementTree(root)
    tree.write(fname)


def project_border(XYZ,surf):
    """ Project a set of points onto a surface

    Args:
        XYZ (ndarray): Nx3 array of points
        surf (GiftiImage): nibabel surface object
    Returns:
        vertices: Nx3 array of vertices
        weights: Nx3 array of weights
    """
    V,F=surf.agg_data()
    # Get the triangle data
    triangles = V[F,:]
    N = triangles.shape[0]
    P = XYZ.shape[0]
    vertices = np.zeros((P,3),dtype=int)
    weights = np.zeros((P,3))
    # Loop through all the points
    for n in range(P):
        # Find the closest vertex on the surface
        sq_dist=np.sum((V-XYZ[n,:])**2,axis=1)
        v_idx=np.argmin(sq_dist)
        # Find the triangle that contains this vertex
        tri_indices=np.where(np.any(F==v_idx,axis=1))[0]
        bary=tri.points_to_barycentric(triangles[tri_indices],np.tile(XYZ[n,:],(tri_indices.shape[0],1)))
        idx=np.where(np.all(np.logical_and(bary>=0,bary<=1),axis=1))[0]
        if len(idx)==0:
            idx=0 # Arbitrarily pick the first one
            bary[idx,:]=0
            i=np.where(F[tri_indices[idx],:]==v_idx)[0][0]
            bary[idx,i]=1
        else:
            idx=idx[0]
        vertices[n,:] = F[tri_indices[idx],:]
        weights[n,:] = bary[idx,:]
    return vertices,weights

def resample_border(border,surf,stepsize=5):
    """ Resample a border with a fixed step size

    Args:
        border(Border): Border object
        surf(GiftiImage): nibabel surface object
        step(float): Step size in mm
    Returns:
        border: Border object
    """
    # Get the coordinates of the existing border
    xyz = border.get_coords(surf).T
    n_points = xyz.shape[1]
    # Create a bezier curve
    bz_curve = bz.Curve(xyz,n_points-1)
    # Figure out how many points we need
    n_resample = int(np.floor(bz_curve.length/stepsize))
    x=np.linspace(0,1,n_resample)
    # Resample the curve
    coord = np.zeros((3,n_resample))
    for i in range(n_resample):
        coord[:,i:i+1]=bz_curve.evaluate(x[i])
    # Project the border back to the surface
    v,w = project_border(coord.T,surf)
    return v,w,coord.T
