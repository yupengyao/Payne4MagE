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

summary(outdir)
