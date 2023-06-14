# Read and preprocess the spectra

#from smhr_session import Session
import pickle
import numpy as np
from astropy.stats import biweight_scale
from scipy import interpolate
from scipy import signal
from scipy.stats import norm
import time, os, glob
import sys, os
from .spectrum import Spectrum1D
from .utils_alexmodes import fast_find_continuum
#from astropy.stats import biweight_scale
from .spectral_model import DefaultPayneModel
from . import plotting
from . import utils
from . import fitting
### Read/prepare spectrum

def preprocess_spectra(specfname, wmin, wmax):
  """
  specfname: The path to keep the spectra
  wmin: The minimum wavelength of the spectra
  wmax: The maximum wavelength of the spectra
  """

  assert os.path.exists(specfname)

  def read_spectrum(fname):
      specs = Spectrum1D.read(fname)
      waves = np.array([np.median(spec.dispersion) for spec in specs])
      iisort = np.argsort(waves)
      specs = [specs[ix] for ix in iisort]

      Npix = len(specs[0].dispersion)
      Nord = len(specs)

      wavelength = np.zeros((Nord, Npix))
      spectrum = np.zeros((Nord, Npix))
      spectrum_err = np.zeros((Nord, Npix))
      for i,spec in enumerate(specs):
          assert len(spec.dispersion) == Npix
          wavelength[i] = spec.dispersion
          spectrum[i] = spec.flux
          spectrum_err[i] = spec.ivar**-0.5
      return wavelength, spectrum, spectrum_err
  def get_quick_continuum(wavelength, spectrum):
      cont = np.zeros_like(wavelength)
      for i in range(cont.shape[0]):
          cont[i] = fast_find_continuum(spectrum[i])
      return cont

  wavelength, spectrum, spectrum_err = read_spectrum(specfname)
  wavelength_blaze = wavelength.copy() # blaze and spec have same
  spectrum_blaze = get_quick_continuum(wavelength, spectrum)
  wavelength, spectrum, spectrum_err = utils.cut_wavelength(wavelength, spectrum, spectrum_err, wmin, wmax)
  wavelength_blaze, spectrum_blaze, spectrum_blaze_err = utils.cut_wavelength(
      wavelength_blaze, spectrum_blaze, spectrum_blaze.copy(), wmin, wmax)
  num_order, num_pixel = wavelength.shape
  spectrum = np.abs(spectrum)
  spectrum_err[(spectrum_err==0) | np.isnan(spectrum_err)] = 999.
  spectrum_blaze = np.abs(spectrum_blaze)    
  spectrum_blaze[spectrum_blaze == 0] = 1.

  # rescale the spectra by its median so it has a more reasonable y-range
  spectrum, spectrum_err = utils.scale_spectrum_by_median(spectrum, spectrum_err.copy())
  spectrum_blaze = spectrum_blaze/np.nanmedian(spectrum_blaze, axis=1)[:,np.newaxis]
  # some orders are all zeros, remove these
  bad_orders = np.all(np.isnan(spectrum), axis=1)

  if bad_orders.sum() > 0:
      print("Removing {} bad orders".format(bad_orders.sum()))
  wavelength, spectrum, spectrum_err, spectrum_blaze = \
      wavelength[~bad_orders], spectrum[~bad_orders], spectrum_err[~bad_orders], spectrum_blaze[~bad_orders]

  # eliminate zero values in the blaze function to avoid dividing with zeros
  # the truncation is quite aggresive, can be improved if needed
  ind_valid = np.min(np.abs(spectrum_blaze), axis=0) != 0
  spectrum_blaze = spectrum_blaze[:,ind_valid]
  wavelength_blaze = wavelength_blaze[:,ind_valid]

  # match the wavelength (blaze -> spectrum)
  spectrum_blaze, wavelength_blaze = utils.match_blaze_to_spectrum(wavelength, spectrum, wavelength_blaze, spectrum_blaze)
  
  return wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze
  
def spectra_analyzing(specfname, outdir, NNpath, rv0, kernel_size, wavelength, spectrum, spectrum_err, wavelength_blaze, spectrum_blaze):
  norder, npix = wavelength.shape
  num_order = norder
  mask_list = [(4850,4880),(6550,6575), (6276,6320),(6866,6881),(6883,6962),(6985,7070)]

  spectrum_err = utils.mask_wavelength_regions(wavelength, spectrum_err, mask_list)
  ## Sigma clip away outliers in the noise, these are bad pixels. We will increase their spectrum_err to very large
  mask = (~np.isfinite(spectrum)) | (~np.isfinite(spectrum_err)) | (~np.isfinite(spectrum_blaze))
  for j in range(spectrum_err.shape[0]):
      err_cont = fast_find_continuum(spectrum_err[j])
      err_norm = spectrum_err[j]/err_cont - 1
      err_errs = biweight_scale(err_norm)
      mask[j, np.abs(err_norm/err_errs) > 10] = True

  # Normalize the errors to cap out at 999
  spectrum_err[spectrum_err > 999] = 999

  ### Radial Velocity


  RV_array = np.array([rv0/100.])
  target_wavelength = 5183
  for iorder in range(num_order):
      if (wavelength[iorder,0] < target_wavelength) and (wavelength[iorder,-1] > target_wavelength):
          break
  else:
      target_wavelength = 4861
      for iorder in range(num_order):
          if (wavelength[iorder,0] < target_wavelength) and (wavelength[iorder,-1] > target_wavelength):
              break
  print("Median RV order wavelength:",np.median(wavelength[iorder]))

  name = os.path.basename(specfname)[:-5]
  outfname_smh = os.path.join(outdir, os.path.basename(specfname)[:-5]+".smh")
  outfname_bestfit = os.path.join(outdir, os.path.basename(specfname)[:-5]+"_paynefit.npz")
  outfname_mask = os.path.join(outdir, os.path.basename(specfname)[:-5]+"_masks.pkl")
  print("Saving to output directory:",outdir)

  #initial_stellar_labels = [4500, 1.5, 2.0, -2.0, 0.4, 0.0]
  initial_stellar_labels = [4500, 1.5, -2.0, 0.4]

  start = time.time()
  print("Running with Payne4MIKE (Conroy Kurucz grid)")

  model = DefaultPayneModel.load(NNpath, num_order=norder)
  errors_payne = utils.read_default_model_mask(wavelength_payne=model.wavelength_payne)
  model = DefaultPayneModel.load(NNpath, num_order=norder, errors_payne=errors_payne)
  print("starting fit")
  print(RV_array)

  num_all_labels = 4 + 3*norder + 2 # norder代表光谱
  num_stellar_labels = 4
  vbroadmin, vbroadmax, rvmin, rvmax = 0.1, 50000, -500, 500

  bounds = np.zeros((2, num_all_labels))
  # polynomial coefficients
  bounds[0,num_stellar_labels:] = -1000 
  bounds[1,num_stellar_labels:] = 1000

  bounds[0,:num_stellar_labels] = -0.5
  bounds[1,:num_stellar_labels] = 0.5

  bounds[0,-2] = vbroadmin
  bounds[1,-2] = vbroadmax
  bounds[0,-1] = rvmin/100.
  bounds[1,-1] = rvmax/100.


  out = fitting.fit_global(kernel_size, spectrum, spectrum_err, spectrum_blaze, wavelength,
                           model, initial_stellar_parameters=initial_stellar_labels,
                           RV_array = RV_array, order_choice=[iorder],
                           bounds_set=bounds)
  popt_best, model_spec_best, chi_square, perr = out
  print("Took",time.time()-start)
  popt_print = model.transform_coefficients(popt_best)

  print("[Teff [K], logg, Fe/H, Alpha/Fe] = ",\
        int(popt_print[0]*1.)/1.,\
        int(popt_print[1]*100.)/100.,\
        int(popt_print[2]*100.)/100.,\
        int(popt_print[3]*100.)/100.,\
  )
  
  print("[Teff_err [K], logg_err, Fe/H_err, Alpha/Fe_err] = ",\
      1000*6.5*int(perr[0]*1.)/1.,\
      5*int(perr[1]*100.)/100.,\
      5.25*int(perr[2]*100.)/100.,\
      0.8*int(perr[3]*100.)/100.,\
  )

  print("vbroad [km/s] = ", int(popt_print[-2]*10.)/10.)
  print("RV [km/s] = ", int(popt_print[-1]*10.)/10.)
  print("Chi square = ", chi_square)

  np.savez(outfname_bestfit,
           popt_best=popt_best,
           popt_print=popt_print,
           model_spec_best=model_spec_best,
           chi_square=chi_square,
           errors_payne=errors_payne,
           wavelength=wavelength,
           spectrum=spectrum,
           spectrum_err=spectrum_err,
           initial_stellar_labels=initial_stellar_labels)
  
  plotting.save_figures(kernel_size, name, wavelength, spectrum, spectrum_err, model_spec_best,
                        errors_payne=errors_payne, popt_best=popt_best, model=model,
                        outdir=outdir, outfname_format="wave")

  
  
  
  
  
