"""Handeling SPM (Statistical Parametric Mapping) fMRI data

Utility object that helps to extract time series data, beta coefficients, and residuals from a GLM stored in a SPM.mat file.

## Usage
```
spm = SpmGlm('path/to/spm')
spm.get_info_from_spm_mat()
[residuals, beta, info] = spm.get_residuals('my_ROI_Mask.nii')
```
"""
from __future__ import annotations
from os.path import normpath, dirname
import numpy as np
import nibabel as nb
import nitools as nt
from scipy.io import loadmat
import pandas as pd
from scipy.stats import gamma
from scipy.special import gammaln


class SpmGlm:
    """class for handling first-levels GLMs estimated in SPM

    Attributes:
        path (str):
            paths to directory containing SPM files
    """

    def __init__(self, path: str, nibabelMock=None, globMock=None):
        self.path = normpath(path)
        # self.nibabel = import_nibabel(nibabelMock)
        # self.glob = globMock or glob.glob

    def get_info_from_spm_mat(self):
        """Initializes information for SPM.mat file

        Args:
            spm_mat_path (str): _description_
        """
        try:
            # SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
            SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
            spm_file_loaded_with_scipyio = True
        except Exception as e:
            print(e)
            print(
                f"Error loading SPM.mat file. The file was saved as mat-file version 7.3 (see https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html). Try loading the mat-file with Matlab, and saving it as mat-file version 7.0 intead. Use this command:  ")
            print(f"cp {self.path}/SPM.mat {self.path}/SPM.mat.backup")
            print(
                f"matlab -nodesktop -nosplash -r \"load('{self.path}/SPM.mat'); save('{self.path}/SPM.mat', '-struct', 'SPM', '-v7'); exit\"")

        # Get basic information from SPM.mat
        self.nscans = SPM['nscan']
        self.nruns = len(self.nscans)
        # Get the name and information on all the beta files``
        self.beta_files = [v['fname'] for v in SPM['Vbeta']]
        self.beta_names = []
        self.run_number = []
        # Extract run number and condition name from SPM names
        for reg_name in SPM['xX']['name']:
            s = reg_name.split(' ')
            self.run_number.append(int(s[0][3:-1]))
            self.beta_names.append(s[1])
        self.run_number = np.array(self.run_number)
        self.beta_names = np.array(self.beta_names)
        # Get the raw data file name
        self.rawdata_files = [self.relocate_file(p) for p in SPM['xY']['P']]
        # Get the necesssary matrices to reestimate the GLM for getting the residuals
        self.filter_matrices = [k['X0'] for k in SPM['xX']['K']]
        self.reg_of_interest = SPM['xX']['iC']
        self.design_matrix = SPM['xX']['xKXs']['X']  # Filtered and whitened design matrix
        self.eff_df = SPM['xX']['erdf']  # Effective degrees of freedom
        self.weight = SPM['xX']['W']  # Weight matrix for whitening
        self.pinvX = SPM['xX']['pKX']  # Pseudo-inverse of (filtered and weighted) design matrix
        self.X = SPM["xX"]["X"]
        self.bf = SPM['xBF']['bf']
        self.Volterra = SPM['xBF']['Volterra']
        self.Sess = SPM['Sess']
        self.T = SPM["xBF"]["T"]
        self.T0 = SPM["xBF"]["T0"]


    def relocate_file(self, fpath: str) -> str:
        """SPM file entries to current project directory and OS.

        These are file paths suffixed with two spaces and an index number.
        They are stored as absolute paths on the computer where it is
        generated, meaning the project directory, as well as OS-dependent
        path separator may have changed. This method fixes these.

        Args:
            fpath (str): SPM-style file path and number from unknown OS

        Returns:
            str: SPM-style file path
        """
        norm_fpath = fpath.replace('\\', '/')
        base_path = dirname(self.path).replace('\\', '/')
        c = norm_fpath.find('imaging_data')
        g = base_path.find('glm')
        return base_path[:g] + norm_fpath[c:]
        # return norm_fpath

    def get_betas(self, mask):
        """
        Samples the beta images of an estimated SPM GLM at the mask locations
        also returns the ResMS values, and the obseration descriptors (run and condition) name

        Args:
            mask (ndarray or nibabel nifti1image):
                Indicates which voxels to extract
                Could be a binary 3d-array or a nifti image of the same size as the data
                Or a 3xP array of coordinates to extract (in mm space)
        Returns:
            data (ndarray): N x P array of beta coefficients
            resms (ndarray): 1d array of ResMS values
            obs_descriptors (dict): with lists reg_name and run_number (N long)
        """

        if isinstance(mask, str):
            mask = nb.load(mask)
        if isinstance(mask, nb.Nifti1Image):
            coords = nt.get_mask_coords(mask)
        elif isinstance(mask, np.ndarray) and (mask.shape[0] == 3):
            coords = mask
        else:
            raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')

        # Generate the list of relevant beta images:
        indx = self.reg_of_interest - 1
        beta_files = [f'{self.path}/{self.beta_files[i]}' for i in indx]
        # Get the data from beta and ResMS files
        rms_file = [f'{self.path}/ResMS.nii']
        data = nt.sample_images(beta_files + rms_file, coords, use_dataobj=False)
        # Return the data and the observation descriptors
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return data[:-1, :], data[-1, :], info

    def get_residuals(self, mask):
        """
        Collects 3d images of a range of GLM residuals
        (typical SPM GLM results) and corresponding metadata
        (scan number + run info) into respective lists

        Args:
            res_range (range): range of to be saved residual images per run
        """
        # Sample the relevant time series data
        if isinstance(mask, str):
            mask = nb.load(mask)
        if isinstance(mask, nb.Nifti1Image):
            coords = nt.get_mask_coords(mask)
        elif isinstance(mask, np.ndarray) and (mask.shape[0] == 3):
            coords = mask
        else:
            raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')

        data = nt.sample_images(self.rawdata_files, coords, use_dataobj=True)

        # Filter and temporal pre-whiten the data
        fdata = self.spm_filter(self.weight @ data)  # spm_filter

        # Estimate the beta coefficients and residuals
        beta = self.pinvX @ fdata
        residuals = fdata - self.design_matrix @ beta

        # Return the regressors of interest
        indx = self.reg_of_interest - 1
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return residuals, beta[indx, :], info

    def spm_filter(self, data):
        """
        Does high pass-filtering and temporal weighting of the data (indentical to spm_filter)

        Args:
            data (ndarray): 2d array of time series data (TxP)
        Returns:
            data (ndarray): 2d array of time series data (TxP)
        """
        scan_bounds = self.nscans.cumsum()
        scan_bounds = np.insert(scan_bounds, 0, 0)

        fdata = data.copy()
        for i in range(self.nruns):
            Y = fdata[scan_bounds[i]:scan_bounds[i + 1], :]
            if self.filter_matrices[i].size > 0:
                # Only apply with filter matrices are not empty
                Y = Y - self.filter_matrices[i] @ (self.filter_matrices[i].T @ Y)
        return fdata

    def rerun_glm(self, data):
        """
        Re-estimate the GLM (without hyperparameter estimation) on new data

        Args:
            data (ndarray): 2d array of time series data (TxP)
        Returns:
            beta (ndarray): 2d array of beta coefficients (PxQ)
            info (dict): with lists reg_name and run_number (Q long)
            data_filt (ndarray): 2d array of filtered time series data (TxP)
            data_hat (ndarray): 2d array of predicted time series data (TxP) - 
                This is predicted only using regressors of interest (without the constant or other nuisance regressors)
            data_adj (ndarray): 2d array of adjusted time series data (TxP)
                This is filtered timeseries with constants and other nuisance regressors substrated out
            residuals (ndarray): 2d array of residuals (TxP)
        """

        # Filter and temporal pre-whiten the data
        data_filt = self.spm_filter(self.weight @ data)  # spm_filter

        # Estimate the beta coefficients and residuals
        beta = self.pinvX @ data_filt
        residuals = data_filt - self.design_matrix @ beta
        # Get estimated (predicted) timeseries (regressors of interest without the constant)
        data_hat = self.design_matrix[:, self.reg_of_interest[:-1]] @ beta[self.reg_of_interest[:-1], :]

        # Get adjusted timeseries
        data_adj = data_hat + residuals

        # Return the regressors of interest (apart from the constant)
        indx = self.reg_of_interest - 1
        info = pd.DataFrame({'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]})

        return beta[indx, :], info, data_filt, data_hat, data_adj, residuals

    def convolve_glm(self, bf):
        """
        Re-convolves the SPM structure with a new basis function.

        Parameters:
        SPM : dict
            SPM design structure.

        Returns:
        SPM : dict
            Modified SPM structure.
        """

        self.bf = bf
        Xx = np.array([])
        Xb = np.array([])
        # iCs, iCc, iCb, iN = [], [], [], []

        # if self.bf.ndim == 2:
        #     num_basis = self.bf.shape[1]
        # else:
        #     num_basis = 1
        num_scan = self.nscans

        for s in range(len(self.Sess)):
            # Number of scans for this session
            k = num_scan[s]

            # Get stimulus functions U
            U = self.Sess[s]["U"]
            num_cond = len(U)

            # Convolve stimulus functions with basis functions
            X, Xn, Fc = spm_Volterra(U, self.bf, self.Volterra)

            # Resample regressors at acquisition times (32 bin offset)
            X = X[np.arange(k) * self.T + self.T0 + 32, :]

            # Orthogonalize within trial type
            for i in range(len(Fc)):
                X[:, Fc[i]["i"][0] - 1] = spm_orth(X[:, Fc[i]["i"][0] - 1])

            # Get user-specified regressors
            C = self.Sess[s]["C"]["C"]
            if C.size > 0:
                num_reg = C.shape[1]
                X = np.hstack([X, spm_detrend(C)])
            else:
                num_reg = 0

            # Store session info
            if Xx.size == 0:
                Xx = X
                Xb = np.ones((k, 1))
            else:
                Xx = blkdiag([Xx, X])
                Xb = blkdiag([Xb, np.ones((k, 1))])

            # iCs.extend([s + 1] * (X.shape[1] + num_reg))
            # iCc.extend(np.kron(np.arange(1, num_cond + 1), np.ones(num_basis, dtype=int)).tolist() + [0] * num_reg)
            # iCb.extend(np.kron(np.ones(num_cond, dtype=int), np.arange(1, num_basis + 1)).tolist() + [0] * num_reg)
            # iN.extend([0] * (num_cond * num_basis) + [1] * num_reg)

        # Finalize design matrix
        self.X = np.hstack([Xx, Xb])

        # Compute weighted and filtered design matrix
        if hasattr(self, "weight"):
            self.design_matrix = self.spm_filter(self.weight @ self.X)
        else:
            self.design_matrix = self.spm_filter(self.X)

        self.design_matrix = self.design_matrix.astype(float)

        # Compute pseudoinverse of weighted and filtered design matrix
        self.pinvX = np.linalg.inv(self.design_matrix.T @ self.design_matrix) @ self.design_matrix.T

        # # Indices for regressors
        # SPM["xX"]["iC"] = list(range(Xx.shape[1]))
        # SPM["xX"]["iB"] = list(range(Xb.shape[1])) + Xx.shape[1]
        # SPM["xX"]["iCs"] = iCs + list(range(1, num_scan + 1))
        # SPM["xX"]["iCc"] = iCc + [0] * num_scan
        # SPM["xX"]["iCb"] = iCb + [0] * num_scan
        # SPM["xX"]["iN"] = iN + [2] * num_scan


def cut(X, pre, at, post, padding='last'):
    """
    Cut segment from signal X.

    Parameters:
    X : np.ndarray
        Input data (rows: time samples, cols: cols)
    pre : int
        N samples before `at`
    at : int or None
        Sample index at which to cut. If None or NaN, returns NaNs.
    post : int
        N samples after `at`
    padding : str, optional
        Padding strategy when time is not available ('nan', 'zero', 'last'). Default is 'last'.

    Returns:
    np.ndarray
        The extracted segment of the trajectory.
    """
    if at is None or np.isnan(at):
        return np.full((pre + post + 1, X.shape[1]), np.nan)

    rows, cols = X.shape
    start, end = max(0, at - pre), min(at + post + 1, rows)
    y0 = X[start:end, :]

    pad_before, pad_after = max(0, pre - (at - start)), max(0, post - (rows - at - 1))

    if padding == 'nan':
        y = np.pad(y0, ((pad_before, pad_after),(0, 0), ), mode='constant', constant_values=np.nan)
    elif padding == 'zero':
        y = np.pad(y0, ((pad_before, pad_after), (0, 0), ), mode='constant', constant_values=0)
    elif padding == 'last':
        y = np.pad(y0, ( (pad_before, pad_after), (0, 0), ), mode='edge')
    else:
        raise ValueError("Unknown padding option. Use: 'nan', 'last', 'zero'")

    return y


def avg_cut(X, pre, at, post, padding='last'):
    """
    Takes a vector of sample locations (at) and returns the signal (X) aligned and averaged around those locations
    Args:
        X (np.array):
            Signal to cut.
        pre (int):
            N samples before cut
        at (np.array or list):
            Cut locations in samples
        post (int):
            N samples after cut
        padding (str, optional):
            Padding strategy when time is not available ('nan', 'zero', 'last'). Default is 'last'.
        stats (str):
            Sufficient statistic used for the area
               mean:       Mean of the voxels
               whitemean:  Mean of the voxel, spatially weighted by noise
        axis (int):
            Axis along which the sufficient statistic is applied. If X.shape = (n_voxels, n_samples) then axis=0
        ResMS (np.array):
            Residual mean squares of shape (n_voxels,) only used if stats='prewhiten' for spatial prewhitening

    Returns:

        y (np.array):
            Signal cut and averaged around locations at.

    """

    y_tmp = []
    for a in at:
        y_tmp.append(cut(X, pre, a, post, padding))

    return np.array(y_tmp)


def blkdiag(matrices):
    """
    Constructs a block diagonal matrix from multiple input matrices.

    Parameters:
    *matrices : list of ndarray
        Matrices to be placed on the block diagonal.

    Returns:
    y : ndarray
        Block diagonal concatenated matrix.
    """
    if len(matrices) == 0:
        return np.array([])

    # Determine final shape
    rows = np.array([m.shape[0] for m in matrices]).sum()
    cols = np.array([m.shape[1] for m in matrices]).sum()

    # Preallocate result matrix with zeros
    y = np.zeros((rows, cols), dtype=matrices[0].dtype)

    # Fill the block diagonal
    r_offset, c_offset = 0, 0
    for m in matrices:
        r, c = m.shape
        y[r_offset:r_offset + r, c_offset:c_offset + c] = m
        r_offset += r
        c_offset += c

    return y


def spm_detrend(x, p=0):
    """
    Polynomial detrending over columns.

    Parameters:
    x : ndarray or list of ndarrays
        Data matrix (or list of matrices).
    p : int, optional
        Order of polynomial (default: 0, i.e., mean subtraction).

    Returns:
    y : ndarray or list of ndarrays
        Detrended data matrix.
    """

    # Handle case where x is a list (equivalent to cell arrays in MATLAB)
    if isinstance(x, list):
        return [spm_detrend(xi, p) for xi in x]

    # Check dimensions
    m, n = x.shape
    if m == 0 or n == 0:
        return np.array([])

    # Mean subtraction (order 0)
    if p == 0:
        return x - np.mean(x, axis=0, keepdims=True)

    # Polynomial adjustment
    G = np.zeros((m, p + 1))
    for i in range(p + 1):
        G[:, i] = (np.arange(1, m + 1) ** i)

    y = x - G @ np.linalg.pinv(G) @ x
    return y


def spm_orth(X, OPT='pad'):
    """
    Recursive Gram-Schmidt orthogonalisation of basis functions

    Parameters:
    X : ndarray
        Input matrix
    OPT : str, optional
        'norm' for Euclidean normalization
        'pad' for zero padding of null space (default)

    Returns:
    X : ndarray
        Orthogonalized matrix
    """
    # Turn off warnings (equivalent to MATLAB warning('off','all'))
    np.seterr(all='ignore')

    if X.ndim==1:
        X = X[:, np.newaxis]
    n, m = X.shape
    X = X[:, np.any(X, axis=0)]  # Remove zero columns
    rankX = np.linalg.matrix_rank(X)

    try:
        x = X[:, [0]]
        j = [0]

        for i in range(1, X.shape[1]):
            D = X[:, [i]]
            D = D - x @ np.linalg.pinv(x) @ D

            if np.linalg.norm(D, 1) > np.exp(-32):
                x = np.hstack((x, D))
                j.append(i)

            if len(j) == rankX:
                break
    except:
        x = np.zeros((n, 0))
        j = []

    # Restore warnings
    np.seterr(all='warn')

    # Normalization if requested
    if OPT == 'pad':
        X_out = np.zeros((n, m))
        X_out[:, j] = x
    elif OPT == 'norm':
        X_out = x / np.linalg.norm(x, axis=0, keepdims=True)
    else:
        X_out = x

    # drop extra dimensions if input was a vector
    X_out = np.squeeze(X_out)

    return X_out


def spm_Volterra(U, bf, V=1):
    """
    Generalized convolution of inputs (U) with basis set (bf)

    Parameters:
    U : list of dict
        Input structure array containing 'u' (time series) and 'name' (labels)
    bf : ndarray
        Basis functions
    V : int, optional
        Order of Volterra expansion (default: 1)

    Returns:
    X : ndarray
        Design Matrix
    Xname : list
        Names of regressors (columns) in X
    Fc : list of dict
        Contains indices and names for each input
    """
    X = []
    Xname = []
    Fc = []

    if bf.ndim == 2:
        num_basis = bf.shape[1]
    else:
        num_basis = 1

    # First-order terms
    for i, u_dict in enumerate(U):
        ind = []
        ip = []
        for k in range(u_dict['u'].shape[1]):
            for p in range(num_basis):
                if num_basis > 1:
                    x = u_dict['u'].todense()[:, k]
                    d = np.arange(x.shape[0])
                    x = np.convolve(x, bf[:, p], mode='full')[:len(d)]
                else:
                    x = u_dict['u'].todense()[:, k]
                    d = np.arange(x.shape[0])
                    x = np.asarray(x).flatten()
                    x = np.convolve(x, bf, mode='full')[:len(d)]
                    x = x[:, None]
                X.append(x)

                Xname.append(f"{u_dict['name'][k]}*bf({p + 1})")
                ind.append(len(X))
                ip.append(k + 1)

        Fc.append({'i': ind, 'name': u_dict['name'][0], 'p': ip})

    X = np.column_stack(X) if X else np.empty((len(U[0]['u']), 0))

    # Return if first order
    if V == 1:
        return X, Xname, Fc

    # Second-order terms
    for i in range(len(U)):
        for j in range(i, len(U)):
            ind = []
            ip = []
            for p in range(bf.shape[1]):
                for q in range(bf.shape[1]):
                    x = U[i]['u'][:, 0]
                    y = U[j]['u'][:, 0]
                    x = np.convolve(x, bf[:, p], mode='full')[:len(d)]
                    y = np.convolve(y, bf[:, q], mode='full')[:len(d)]
                    X = np.column_stack((X, x * y))

                    Xname.append(f"{U[i]['name'][0]}*bf({p + 1})x{U[j]['name'][0]}*bf({q + 1})")
                    ind.append(X.shape[1])
                    ip.append(1)

            Fc.append({'i': ind, 'name': f"{U[i]['name'][0]}x{U[j]['name'][0]}", 'p': ip})

    return X, Xname, Fc


def spm_hrf(RT, P=None, T=16):
    """
    Haemodynamic response function (HRF)

    Parameters:
    RT : float
        Scan repeat time in seconds (TR)
    P : list or ndarray, optional
        Parameters of the response function (two Gamma functions), defaults to [6, 16, 1, 1, 6, 0, 32]
    T : int, optional
        Microtime resolution (default: 16)

    Returns:
    hrf : ndarray
        Hemodynamic response function
    p : ndarray
        Parameters of the response function
    """
    # Default parameters if not provided
    if P is None:
        P = [6, 16, 1, 1, 6, 0, 32]

    # Ensure P has the correct length
    p = np.array([6, 16, 1, 1, 6, 0, 32])
    p[:len(P)] = P

    # Microtime resolution
    RT = RT / T
    dt = RT / T
    t = np.linspace(0, p[6], np.ceil(1 + p[6] / dt).astype(int)) - p[5]

    peak = (t ** p[0]) * np.exp(-t / p[2])
    peak /= np.max(peak)  # Normalize

    undershoot = (t ** p[1]) * np.exp(-t / p[3])
    undershoot /= np.max(undershoot)  # Normalize

    hrf = peak - (undershoot / p[4])

    # Downsample to TR resolution
    hrf = hrf[np.floor(np.arange(0, p[6] / RT) * T).astype(int)]

    # Normalize HRF
    hrf = hrf / np.sum(hrf)

    return hrf, p
