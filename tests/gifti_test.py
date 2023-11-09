
import nibabel as nb
import nitools as nt


atlas_dir = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fsLR_32/'



atlas_dir2 = '/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/projects/Atlas_templates/fsLR_32average6/atl-MSHBM_Prior_15_fsaverage6/'

def make_label_gifti():
    hem = ['L','R']
    hem_name = ['CortexLeft','CortexRight']
    T = nb.load(atlas_dir2 + 'MSHBM_Prior_15_fsaverage6.R.label.gii')
    l = nt.gifti.get_gifti_labels(T)
    c = nt.get_gifti_colortable(T)
    for i,h in enumerate(hem):
        A  = nb.load(atlas_dir2 + f'MSHBM_Prior_15_fsLR32.{h}.func.gii')
        GII = nt.make_label_gifti(A.agg_data(),anatomical_struct=hem_name[i],label_names=l,label_RGBA=c[0])
        nb.save(GII,atlas_dir2 + f'MSHBM_Prior_15_fsLR32.{h}.label.gii')


def make_parcel_gifti():
    pass

def join_label_gifti():
    fname = [atlas_dir + '/Icosahedron-162.32k.L.label.gii',
             atlas_dir + '/Icosahedron-162.32k.R.label.gii']
    cifti_img = nt.join_giftis_to_cifti(fname,seperate_labels = True,join_zero=True)
    nb.save(cifti_img,base_dir+'/Icosahedron-162.dlabel.nii')

def join_label_gifti2():
    fname = [atlas_dir + 'MSHBM_Prior_15_fsLR32.L.label.gii',
             atlas_dir + 'MSHBM_Prior_15_fsLR32.R.label.gii']
    cifti_img = nt.join_giftis_to_cifti(fname,seperate_labels = False)
    nb.save(cifti_img,atlas_dir+'MSHBM_Prior_15_fsLR32.dlabel.nii')


def join_func_gifti():
    fname = [base_dir + '/wgroup.encode-rest.L.func.gii',
             base_dir + '/wgroup.encode-rest.R.func.gii']
    cifti_img = nt.join_giftis_to_cifti(fname)
    nb.save(cifti_img,base_dir+'/wgroup.encode-rest.dscalar.nii')


if __name__ == '__main__':
    # make_func_cifti()
    join_label_gifti2()
    pass