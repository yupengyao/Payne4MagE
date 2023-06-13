# Read and preprocess the spectra

#from smhr_session import Session
import pickle
import numpy as np

from scipy import interpolate
from scipy import signal
from scipy.stats import norm
import time, os, glob


### Read/prepare spectrum

specfname = sys.argv[1]
wmin = sys.argv[2]
wmax = sys.argv[3]
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
  
  
