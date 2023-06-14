!git clone https://github.com/yupengyao/Payne4MagE.git
from Payne4MagE.Payne4MIKE.functions import preprocess_spectra, spectra_analyzing

import astropy
from astropy.io import fits, ascii
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from astropy.stats import biweight_scale

path = 'drive/MyDrive/Payne4MagE/payne_inputs/herc_12_multi.fits'
outdir = '/content/drive/MyDrive/Payne4MagE/github_output'
NNpath = '/content/drive/MyDrive/Payne4MagE/NN_normalized_spectra_float16_fixwave.npz'

wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze = preprocess_spectra(path, 3500, 7000)

rv0 = 80
kernel_size = 40
specfname = 'spec_'+path[34:-5]
spectra_analyzing(specfname, outdir, NNpath, rv0, kernel_size, wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze)


import numpy as np
import os, sys, glob
from astropy.table import Table
if __name__=="__main__":
    alldata = []
    fnames = glob.glob("/content/drive/MyDrive/Payne4MagE/github_output/*.npz")
    for fn in fnames:
        dirname = os.path.dirname(fn)
        basename = os.path.basename(fn)
        s = basename.split("_")
        print(basename)
        star = s[0]+'_'+s[1]
        print(star)
        tmp = np.load(fn)
        params = tmp["popt_print"]
        perr = tmp["perr"]
        Teff, logg, MH, aFe = params[0:4]
        Teff_err = 1000*6.5*int(perr[0]*1000.)/1000. 
        logg_err = 5*int(perr[1]*10000.)/10000
        MH_err = 5.25*int(perr[2]*100.)/100.
        aFe_err = 0.8*int(perr[3]*100.)/100.
        vt = 1.0
        vbroad, rv = params[-2:]
        chi2 = tmp["chi_square"]
        snr = np.median(np.nanmedian(tmp["spectrum"]/tmp["spectrum_err"], axis=1))
        norder = tmp["spectrum"].shape[0]
        alldata.append([star, Teff, Teff_err, logg, logg_err, vt, MH, MH_err, aFe, aFe_err, vbroad, rv, chi2, snr, norder])
    alldata = Table(rows=alldata, names=["star", "Teff", "Teff_err", "logg", "logg_err", "vt", "MH", "MH_err", "aFe", "aFe_err", "vbroad", "rvobs", "chi2", "snr", "norder"])
    alldata["Teff"].format = ".0f"
    alldata["Teff_err"].format = ".0f"
    for col in ["logg","logg_err","vt","MH","MH_err","aFe","aFe_err","chi2"]: alldata[col].format = ".2f"
    for col in ["vbroad","rvobs","snr"]: alldata[col].format = ".1f"
    alldata.write("/content/drive/MyDrive/Payne4MagE/github_output/summary.org", format="ascii.fixed_width", overwrite=True)
