# Clone this repository 
!git clone https://github.com/yupengyao/Payne4MagE.git

# import packages
from Payne4MagE.Payne4MIKE.functions import preprocess_spectra, spectra_analyzing
from Payne4MagE.Payne4MIKE.summary import summary
import astropy
from astropy.io import fits, ascii
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from astropy.stats import biweight_scale

# Input and preprecess the spectrum.
indir = '/content/drive/MyDrive/Payne4MagE/payne_inputs/herc_12_multi.fits' 
wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze = preprocess_spectra(indir, 3500, 7000)

# Spectra analyzing.
rv0 = 80 # Radius velocity of the target.
kernel_size = 40 # kernel size: 40 for MagE spectra and 10 for MIKE spectra.
specfname = 'spec_' + indir[34:-5]
outdir = '/content/drive/MyDrive/Payne4MagE/github_output'
NNpath = '/content/drive/MyDrive/Payne4MagE/NN_normalized_spectra_float16_fixwave.npz' # Path of the emulator model.
spectra_analyzing(specfname, outdir, NNpath, rv0, kernel_size, wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze) # Analyzing the spectra.

# Go through the result folder and make a summary table.
summary(outdir)
