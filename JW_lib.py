# -*- coding: utf-8 -*-
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as pyfits
import astropy.io.ascii as ascii
import scipy.constants
import astropy.constants
import pickle
from scipy import signal
from astropy import units as u
import numpy.fft as fft
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import time
import scipy.interpolate
import scipy.fftpack as fp
from datetime import datetime
import scipy.sparse as sp
from scipy.optimize import minimize

class Target():
    def __init__(self, distance=10.0, spec_path=None, inclination_deg=90.0, rotation_vel=5e3, radial_vel=1e4, spec_reso=1e5):
        self.distance = distance
        self.spec_path =  spec_path
        self.spec_reso = spec_reso
        if self.spec_path != None:
            self.spec_data = pyfits.open(self.spec_path)[1].data
            self.wavelength = self.spec_data["Wavelength"]
            self.flux = self.spec_data["Flux"]
            self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
            self.spec_header = pyfits.open(self.spec_path)[1].header
            self.PHXREFF = self.spec_header["PHXREFF"]
        else:
            wav_min = 2.000
            wav_max = 2.002
            wav_int = 5e-7
            self.wavelength = np.arange(wav_min, wav_max, wav_int)
            self.flux = 1.0 - np.exp(-1.0*(self.wavelength - (wav_max + wav_min)/2.0)**2 / 2.0 / (20.0*wav_int)**2) # 3 km/s microturbulence
            self.spectrum = Spectrum(self.wavelength, self.flux, spec_reso=self.spec_reso)
        self.inclination_deg = inclination_deg
        self.rotation_vel = rotation_vel
        self.radial_vel = radial_vel

class Instrument():
    def __init__(self, wav_med, telescope_size=10.0, pl_st_contrast=1e-10, spec_reso=1e5, read_noise=2.0, dark_current=1e-3, fiber_size=1.0, pixel_sampling=3.0, throughput=0.1, wfc_residual=200.0):   
        self.telescope_size = telescope_size
        self.pl_st_contrast = pl_st_contrast
        self.spec_reso = spec_reso
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.fiber_size = fiber_size
        self.pixel_sampling = pixel_sampling
        self.wfc_residual = wfc_residual # in nm
        self.wav_med = wav_med # in micron
        self.strehl = self.calStrehl()
        self.relative_strehl = self.calRelativeStrehl()
        self.throughput = throughput * self.relative_strehl

    def calStrehl(self):
        strehl = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / self.wav_med))**2) 
        return(strehl)

    def calRelativeStrehl(self):
        strehl_K = np.exp(-(2.0 * np.pi * (self.wfc_residual / 1e3 / 2.0))**2)
        strehl = self.calStrehl()
        return(strehl / strehl_K)

class Spectrum():
    def __init__(self, wavelength, flux, spec_reso=1e5, norm_flag=False):
        self.wavelength = wavelength
        self.flux = flux
        self.spec_reso = spec_reso
        self.norm_flag = norm_flag
        self.noise = None

    def converVelocitytoWavelength(self, vel_arr, lambda_0=2.0):
        # redshift velocity is positive
        delta_wav = vel_arr / scipy.constants.c * lambda_0
        return(delta_wav + lambda_0)

    def converWavelengthtoVelocity(self, wav_arr):
        lambda_0 = np.median(wav_arr)
        delta_wav = wav_arr - lambda_0
        vel_arr = delta_wav / lambda_0 * scipy.constants.c
        return(vel_arr)

    def calcRotationalBroadeningKernel(self, u1=1.0, u2=0.0, vsini=20e3, lambda_0=2.0):
        # from Hirano et al. 2010
        lambda_L = vsini / scipy.constants.c * lambda_0
        u3 = (1.0 - u1 / 3.0 - u2 / 6.0)
        c1 = 2.0 * (1.0 - u1 - u2) / np.pi / lambda_L / u3
        c2 = (u1 + 2.0 * u2) / 2.0 / lambda_L / u3
        c3 = 4.0 * u2 / 3.0 / np.pi / lambda_L / u3
        x = np.arange(lambda_0-5.0*lambda_L, lambda_0+5.0*lambda_L, lambda_L/1e3)
        ratio2 = ((x - lambda_0) / lambda_L)**2
        ratio2[np.where(ratio2 < 0.0)] = 0.0
        ratio2[np.where(ratio2 > 1.0)] = 1.0
        kernel = c1 * np.sqrt(1.0 - ratio2) + c2 * (1.0 - ratio2) + c3 * (1.0 - ratio2)**(1.5)
        if np.sum(kernel) < 0.0:
            kernel = kernel * -1.0
        kernel = kernel / np.sum(kernel)
        return(Spectrum(x, kernel))

    def sortSpec(self):
        ind = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[ind]
        self.flux = self.flux[ind]
        if np.size(self.noise) != 1:
            self.noise = self.noise[ind]

    def addNoise(self, noise):
        if np.size(self.noise) == 1:
            self.noise = noise
        else:
            print("Warning: spectrum noise already added")

    def writeSpec(self, file_name="tmp.dat",flag_python2=False,flag_append=False):
        if not flag_append:
            write_code = "wb"
        else:
            write_code = "ab"
        with open(file_name, write_code) as f:
            for i in np.arange(len(self.wavelength)):
                if flag_python2 == True:
                    if np.size(self.noise) == 1:
                        f.write("{0:20.8f}{1:20.8e}\n".format(self.wavelength[i], self.flux[i]))
                    else:
                        f.write("{0:20.8f}{1:20.8e}{2:20.8e}\n".format(self.wavelength[i], self.flux[i], self.noise[i]))
                else:
                    if np.size(self.noise) == 1:
                        f.write(bytes("{0:20.8f}{1:20.8e}\n".format(self.wavelength[i], self.flux[i]), 'UTF-8'))
                    else:
                        f.write(bytes("{0:20.8f}{1:20.8e}{2:20.8e}\n".format(self.wavelength[i], self.flux[i], self.noise[i]), 'UTF-8'))

    def getSpecNorm(self, num_chunks=10, poly_order=2, emission=False):
        wav = self.wavelength
        flx = self.flux
        num_pixels = len(wav)
        pix_chunk = int(np.floor(num_pixels / num_chunks))
        wav_chunk = np.zeros((num_chunks,))
        flx_chunk = np.zeros((num_chunks,))
        for i in np.arange(num_chunks):
            wav_chunk[i] = np.nanmedian(wav[i*pix_chunk:(i+1)*pix_chunk])
            if not emission:
                flx_chunk[i] = np.nanmax(flx[i*pix_chunk:(i+1)*pix_chunk])
            else:
                flx_chunk[i] = np.nanmin(flx[i*pix_chunk:(i+1)*pix_chunk]) 
        coeff = np.polyfit(wav_chunk, flx_chunk, poly_order)
        p = np.poly1d(coeff)
        flx_norm = p(wav)
        return(flx_norm)

    def combineSpec(self, spec):
        spec_new = self.copy()
        idx = np.argsort(np.hstack((self.wavelength, spec.wavelength)))
        spec_new.wavelength = np.hstack((self.wavelength, spec.wavelength))[idx]
        spec_new.flux = np.hstack((self.flux, spec.flux))[idx]
        if np.size(spec_new.noise) > 1:
            spec_new.noise = np.hstack((self.noise, spec.noise))[idx] # added 20170727
            print("Combining spectrum may cause trouble for attribute Noise")
        return(spec_new)

    def getChunk(self, wav_min, wav_max):
        spec_new = self.copy()
        idx = np.where((self.wavelength <= wav_max) & (self.wavelength > wav_min))
        if np.size(idx[0]) > 1: 
            spec_new.wavelength = self.wavelength[idx]
            spec_new.flux = self.flux[idx]
        else:
            spec_new.wavelength = None
            spec_new.flux = None
        if np.size(spec_new.noise) > 1:
            spec_new.noise = self.noise[idx]
        return(spec_new)

    def removeNaN(self):
        flx = self.flux
        wav = self.wavelength
        idx1 = np.isnan(flx)
        idx2 = np.isfinite(flx)
        flx[idx1] = np.interp(wav[idx1], wav[idx2], flx[idx2])
        self.flux = flx
        return(self)

    def removeZeros(self):
        flx = self.flux
        wav = self.wavelength
        idx1 = np.where(flx == 0.0)
        idx2 = np.where(flx != 0.0)
        flx[idx1] = np.interp(wav[idx1], wav[idx2], flx[idx2])
        self.flux = flx
        return(self)

    def saveSpec(self, file_name="tmp.pkl"):
        with open(file_name, "wb") as handle:
            pickle.dump(self, handle)

    def removeOutliers(self, sigma_reject=7.0, sigma_clip=None, kernel_size=None, flag_plot=False):
        if np.size(self.noise) > 1.0:
            wav = self.wavelength
            flx = self.flux
            unc = self.noise
            if sigma_clip is None:
                sigma_clip = np.nanmedian(flx) / np.nanmedian(unc)
            else: 
                sigma_clip = sigma_clip
            if kernel_size is None:
                if np.round((len(wav) / 20.0)) % 2:
                    kernel_size = int(np.round(len(wav) / 20.0))
                else:
                    kernel_size = int(np.round(len(wav) / 20.0)) + 1
            else:
                kernel_size = 3
            flx_std = np.std(flx - scipy.signal.medfilt(flx, kernel_size=kernel_size))
            # remove sigma outliers
            if flag_plot:
                plt.plot(wav, flx, alpha=0.5, label="orig")
            sigma_arr = (np.abs(flx - scipy.signal.medfilt(flx, kernel_size=kernel_size))) / flx_std
            ind_outliers = np.where(sigma_arr > sigma_reject)
            ind = np.where(sigma_arr < sigma_reject)
            flx[ind_outliers] = np.interp(wav[ind_outliers], wav[ind], flx[ind])
            if flag_plot:
                print("sigma_reject = ", sigma_reject)
                print("min and max of flux:", np.min(flx), np.max(flx))
                print("flx_std = ", flx_std)
                plt.errorbar(wav, flx, yerr=unc, label="spec")
                plt.plot(wav, scipy.signal.medfilt(flx, kernel_size=kernel_size), label="med")
                plt.plot(wav[ind_outliers], flx[ind_outliers], "o", label="outliers")
                plt.legend()
                plt.show()
            # remove uncertainty outliers
            if flag_plot:
                plt.plot(wav, flx, alpha=0.5, label="orig")
            spec = self.getCrossValidateSpectrum()
            sigma_arr = (np.abs(flx - spec.flux)) / unc
            ind_outliers = np.where(sigma_arr > sigma_clip)
            ind = np.where(sigma_arr < sigma_clip)
            flx[ind_outliers] = np.interp(wav[ind_outliers], wav[ind], flx[ind])
            if flag_plot:
                print("sigma_clip = ", sigma_clip)
                print("min and max of flux:", np.min(flx), np.max(flx))
                plt.errorbar(wav, flx, yerr=unc, label="spec")
                plt.plot(wav, spec.flux, label="remove one")
                plt.plot(wav[ind_outliers], flx[ind_outliers], "o", label="outliers")
                plt.legend()
                plt.show()
            self.flux = flx
            return(self) 
        else:
            print("Need to have noise attribute to remove outliers")
            return(self)

    def getCrossValidateSpectrum(self):
        flx = self.flux
        wav = self.wavelength
        spec = self.copy()
        for i in np.arange(len(wav)):
            ind = np.where(wav != wav[i])
            spec.flux[i] = np.interp(spec.wavelength[i], wav[ind], flx[ind])
        return(spec)

    def simSpeckleNoise(self, wav_min, wav_max, wav_int, wav_new):
        wav = np.arange(wav_min, wav_max + wav_int / 10.0, wav_int)
        wav_arr = np.array([])
        flx_arr = np.array([])
        for i, wav_tmp in enumerate(wav[:-1]):
            wav_mid = np.random.normal(wav_tmp + wav_int * 0.5, wav_int / 10.0, size=(1,))
            flx_mid = np.random.random(size=(1,)) 
            flx_tmp = np.random.random(size=(1,)) + 1.0
            wav_arr = np.hstack((wav_arr, [wav_tmp, wav_mid[0]]))
            flx_arr = np.hstack((flx_arr, [flx_tmp, flx_mid[0]]))
        wav_arr = np.hstack((wav_arr, [wav[-1]]))
        flx_arr = np.hstack((flx_arr, [np.random.random(1)[0] + 1.0]))
        f = scipy.interpolate.interp1d(wav_arr, flx_arr, kind="cubic")
        flx_new = f(wav_new)
        idx = np.where(flx_new < 0.1)
        flx_new[idx] = 0.1
        return(flx_new)

    def generateNoisySpec(self, speckle_noise=False, star_flux=1e0):
        spec = self.copy()
        flx = self.flux
        flx_new = np.zeros(np.shape(flx))
        num = len(flx)
        i = 0
        if hasattr(flx[0], 'value'):
            while i < num:
                #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
                flx_new[i] = np.random.normal(flx[i].value, self.noise[i].value, 1)
                i = i + 1
        else:
            while i < num:
                #flx_new[i] = np.max([np.random.poisson(np.round(flx[i]), 1)+0.0, np.random.normal(flx[i], self.noise[i], 1)])
                flx_new[i] = np.random.normal(flx[i], self.noise[i], 1)
                i = i + 1
        spec.flux = flx_new

        if speckle_noise:
            flx_speckle = self.simSpeckleNoise(np.min(spec.wavelength), np.max(spec.wavelength), 0.1, spec.wavelength)
            #spec.flux = spec.flux * flx_speckle
            spec.flux = spec.flux + flx_speckle * star_flux

        return(spec)

    def evenSampling(self):
        wav = self.wavelength
        flx = self.flux
        wav_int = np.median(np.abs(np.diff(wav)))
        wav_min = np.min(wav)
        wav_max = np.max(wav)
        wav_new = np.arange(wav_min, wav_max, wav_int)
        flx_new = np.interp(wav_new, wav, flx)
        self.wavelength = wav_new
        self.flux = flx_new

        return(self)

    def applyHighPassFilter(self, order = 5, cutoff = 100.0, pass_type="high", fourier_flag=False, plot_flag=False):
        if not fourier_flag:
            # cutoff is number of sampling per 1 micron, so 100 means 0.01 micron resolution, about R = 100 at 1 micron
            x = self.wavelength
            y = self.flux
            n = self.noise
            fs = 1.0 / np.median(x[1:-1] - x[0:-2])
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            #print("normal_cutoff = ", normal_cutoff)
            b, a = scipy.signal.butter(order, normal_cutoff, btype=pass_type, analog=False)
            yy = scipy.signal.filtfilt(b, a, y)
            spec = Spectrum(x, yy, spec_reso=self.spec_reso)
            if n is not None:
                spec.addNoise(n)
            return(spec)
        else:
            x = self.wavelength
            y = self.flux
            n = self.noise
            fy = fp.fft(y)
            fy = fp.fftshift(fy)
            delta_x = np.median(np.abs(np.diff(x)))
            N = len(x)
            fx = np.linspace(-0.5 / delta_x, 0.5 / delta_x - 0.5 / delta_x / N, num=N)
            if plot_flag:
                plt.plot(fx, np.abs(fy))
                plt.yscale("log")
            delta_x = np.median(np.abs(np.diff(x)))
            if pass_type == "high":
                filter_envelope = 1.0 - 1.0 * np.exp(-1.0 * fx**2 / (2.0 * cutoff**2))
            else:
                filter_envelope = 1.0 * np.exp(-1.0 * fx**2 / (2.0 * cutoff**2))
            if plot_flag:
                plt.plot(fx, filter_envelope * np.max(fy))
                plt.show()
            ffy = fp.ifft(fp.ifftshift(fy* filter_envelope))
            if plot_flag:
                plt.plot(x,y)
                plt.plot(x, ffy)
                plt.show()
            spec = Spectrum(x, ffy, spec_reso=self.spec_reso)
            if n is not None:
                spec.addNoise(n)
            return(spec)

    def copy(self):
        # make a copy of a spectrum object
        spectrum_new = Spectrum(self.wavelength.copy(), self.flux.copy(), spec_reso=self.spec_reso)
        if np.size(self.noise) > 1:
            spectrum_new.noise = self.noise.copy()
        return(spectrum_new)

    def pltSpec(self, noise_flag=False, **kwargs):
        # plot wav vs. flx 
        # **kwargs accepted by plot
        # image is stored as tmp.png
        fig, ax = plt.subplots()
        if not noise_flag:
            ax.plot(self.wavelength, self.flux, **kwargs)
            plt.show()
        else:
            ax.plot(self.wavelength, self.flux, **kwargs)
            ax.errorbar(self.wavelength, self.flux, yerr=self.noise)
            plt.show()
        #fig.savefig("./tmp.png")

    def scaleSpec(self, total_flux=1e4):
        # scale spectrum so that summed flux from each pixel is equal to total_flux
        num_pixels = len(self.wavelength)
        spec_total_flux = np.sum(self.flux)
        flx = self.flux / spec_total_flux * total_flux
        self.flux = flx

    def resampleSpec(self, wav_new, **kwargs):
        # resample a spectrum to a new wavelength grid
        flx_new = np.interp(wav_new, self.wavelength, self.flux, **kwargs)
        if np.size(self.noise) != 1:
            noise_new = np.interp(wav_new, self.wavelength, self.noise, left=1e9, right=1e9)
            self.noise = noise_new
        self.wavelength = wav_new
        self.flux = flx_new
        return self

    def resampleSpecByOrders(self, wav_seg, wav_new, method="cubic"):
        # wav_seg is from detectOrders
        # method options are: ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic, ‘cubic’
        flx_new = np.zeros(np.shape(wav_new))
        flx_new[:] = np.NaN
        flx_err_new = np.zeros(np.shape(wav_new))
        flx_err_new[:] = np.NaN
        num = len(wav_seg)
        count_num = 0.0
        idx_small_segs = np.array([])
        for i in np.arange(num):
            if np.size(wav_seg) != 2:
                wav_min, wav_max = wav_seg[i]
            else:
                wav_min, wav_max = wav_seg
            wav_range = wav_max - wav_min
            idx_new = np.where((wav_new > wav_min + wav_range / 40.0) & (wav_new < wav_max - wav_range / 40.0))
            idx_old = np.where((self.wavelength >= wav_min) & (self.wavelength <= wav_max))
            num_old = np.size(self.wavelength[idx_old])
            #if np.size(idx_old) < 5:
            if np.size(idx_old) < 15:
                if count_num == 0.0:
                    idx_small_segs = idx_new[0]
                else:
                    idx_small_segs = np.append(idx_small_segs, idx_new[0])
                count_num = count_num + 1.0
                continue
            #plt.plot(self.wavelength[idx_old], self.flux[idx_old], "bo")
            #plt.show()
            num_new = np.size(wav_new[idx_new])
            f = scipy.interpolate.interp1d(self.wavelength[idx_old], self.flux[idx_old], kind=method)
            flx_new[idx_new] = f(wav_new[idx_new])
            if np.size(self.noise) != 1:
                f = scipy.interpolate.interp1d(self.wavelength[idx_old], self.noise[idx_old], kind=method)
                flx_err_new[idx_new] = f(wav_new[idx_new]) * ((num_new + 0.0) / (num_old + 0.0))
        idx = np.isnan(flx_new)
        flx_new[idx] = np.nanmedian(flx_new)
        flx_err_new[idx] = 1e9
        if np.size(idx_small_segs) != 0:
            flx_new[idx_small_segs] = np.nanmedian(flx_new)
            flx_err_new[idx_small_segs] = 1e9
        spec_new = Spectrum(wav_new, flx_new)
        if np.size(self.noise) != 1:
            spec_new.addNoise(flx_err_new)
        return(spec_new)
 
    def detectOrders(self):
        wav = self.wavelength
        wav_dif = np.diff(wav)
        wav_int = np.nanmedian(wav_dif)
        ind = np.where(wav_dif > 10.0 * wav_int)
        if np.size(ind) == 0:
            return([np.min(wav), np.max(wav)])
        else:
            for i in np.arange(len(ind[0])+1):
                if i == 0:
                    tmp = ([[np.min(wav), wav[ind[0][i]]]])
                else:
                    if i < len(ind[0]):
                        tmp.append([wav[ind[0][i-1]+1], wav[ind[0][i]]])
                    else:
                        tmp.append([wav[ind[0][i-1]+1], np.max(wav)])
            return(tmp)

    def convolveKernel(self, kernel):
        spec = self.copy()
        spec.flux = np.convolve(self.flux, kernel, mode="same")
        return(spec)

    def createTriDiagonalMatrix(self, n, method="Second"):
    # Creating a regularization matrix (http://arxiv.org/pdf/1008.5115v2.pdf)
        if method == "Second":
            R_size = n
            R = np.zeros((R_size,R_size))
            np.fill_diagonal(R, 2.)
            R[0,0] = 1.
            R[R_size-1, R_size-1] = 1.
            for i in np.arange(R_size):
                for ii in np.arange(R_size):
                    if np.abs(i - ii) == 1:
                        R[i, ii] = -1.
        elif method == "First":
            R_size = n
            R = np.zeros((R_size,R_size))
            np.fill_diagonal(R, 1.)
            for i in np.arange(R_size-1):
                R[i, i+1] = -1.
        else:
            print("Wrong option for method!")
        return(R)

    def leastSquareDeconvolution(self, F, S, Lambda, n):
        Y_noise = self.flux
        print(len(F), np.shape(F))
        # Creating a regularization matrix (http://arxiv.org/pdf/1008.5115v2.pdf)
        R = self.createTriDiagonalMatrix(n)
        # Creating M matrix
        M = np.roll(scipy.linalg.toeplitz(F)[:, 0:n], int(-n/2), axis = 0)
        # LSD    
        if np.max(S) > 1e-3:
            ACF = np.transpose(M).dot(S).dot(S).dot(M) + Lambda * R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(S).dot(S).dot(Y_noise))
        else:
            ACF = np.transpose(M).dot(M) + Lambda * R
                                                    
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(Y_noise))        
        return(Z_noise)

    def leastSquareDeconvolution_multi(self, F, S, Lambda, n, method="Second"):
        # n is the number of points in Z, for multi template case n = num_F * n, where num_F is the number of templates
        F = np.array(F)
        num_F = len(F)
        Y_noise = self.flux
        # Creating a regularization matrix (http://arxiv.org/pdf/1008.5115v2.pdf)
        R = self.createTriDiagonalMatrix(n * num_F, method=method)
        # Creating M matrix
        M = np.zeros((len(Y_noise), n * num_F))
        for i in np.arange(num_F):
            M[:,i*n:(i+1)*n] = np.roll(scipy.linalg.toeplitz(F[i,:])[:, 0:n], int(-n/2), axis = 0)
        print(np.shape(M), np.shape(F), np.shape(S), np.shape(Y_noise), np.shape(R))
        # LSD    
        #print(np.max(S))
        if np.max(S) > 1e-6:
            ACF = np.transpose(M).dot(S).dot(S).dot(M) + Lambda * R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(S).dot(S).dot(Y_noise))
        else:
            ACF = np.transpose(M).dot(M) + Lambda * R

            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(Y_noise))
        return(Z_noise)

    def leastSquareDeconvolutionMultiProfileMultiLambda(self, F, S, Lambda, n, method="Second"):
        # n is the number of points in Z, for multi template case n = num_F * n, where num_F is the number of templates
        F = np.array(F)
        num_F = len(F)
        Y_noise = self.flux
        # Creating a regularization matrix (http://arxiv.org/pdf/1008.5115v2.pdf)
        R0 = self.createTriDiagonalMatrix(n, method=method)
        if num_F != 1:
            for i in np.arange(num_F):
                #print(Lambda[i])
                if i == 0:
                    R = R0 * Lambda[i]
                else:
                    R = scipy.linalg.block_diag(R, R0 * Lambda[i])
        else:
            R = R0 * Lambda
        # Creating M matrix
        M = np.zeros((len(Y_noise), n * num_F))
        for i in np.arange(num_F):
            M[:,i*n:(i+1)*n] = np.roll(scipy.linalg.toeplitz(F[i,:])[:, 0:n], int(-n/2), axis = 0)
        #print(np.shape(M), np.shape(F), np.shape(S), np.shape(Y_noise), np.shape(R))
        # LSD    
        #print(np.max(S))
        if np.max(S) > 1e-6:
            ACF = np.transpose(M).dot(S).dot(S).dot(M) + R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(S).dot(S).dot(Y_noise))
        else:
            ACF = np.transpose(M).dot(M) + R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(Y_noise))
        return(Z_noise)

    def leastSquareDeconvolutionMultiProfileMultiLambda_Bayesian(self, F, S, n, plot_flag=True, fix_flag=False, pars_best_fixed=None):
        # see Romas et al. (2015)
        # Github: https://github.com/aasensio/pyGPLSD/blob/master/gpLSD.py
        # n is the number of points in Z, for multi template case n = num_F * n, where num_F is the number of templates
        F = np.array(F)
        num_F = len(F)
        Y_noise = self.flux
        # Creating M matrix
        M = np.zeros((len(Y_noise), n * num_F))
        for i in np.arange(num_F):
            M[:,i*n:(i+1)*n] = np.roll(scipy.linalg.toeplitz(F[i,:])[:, 0:n], int(-n/2), axis = 0)

        # v is velocity grid for output line profile
        x = self.wavelength
        nLambda = n * num_F
        v = np.median(np.abs(np.diff(x))) / np.median(x) * 3e8 * (np.arange(nLambda) - nLambda / 2.0)

        # y and yerr
        y = self.flux
        if self.noise is None:
            yerr = np.zeros(np.shape(y)) + 0.1
        else:
            yerr = self.noise

        # set initial parameters for GP
        pars = np.log(np.array([1.0 / (10*710)**2, 1.0])) # lambdaGP, sigmaGP
        
        # calculate matrices required in marginal function
        W = M
        variance = yerr**2
        beta = 1.0 / variance
        AInvDiag = np.ones(len(beta)) * beta
        AInv = sp.diags(AInvDiag, 0) 
        logA = -np.sum(np.log(AInvDiag)) 
        P = W.T.dot(AInv.dot(y))
        Q = np.dot(y.T, (AInv.dot(y)))
        WTAInvW = W.T.dot(AInv.dot(W))

        # get optimized parameters for GP
        epsilon = 1e-9
        if fix_flag:
            pars_best = pars_best_fixed
        else:
            pars_best = self.optimizeDirectGP(v, epsilon, nLambda, WTAInvW, P, Q, logA, variance=variance, pars=pars)
            print(pars_best)

        # get covariance matrix for GP
        K = self.squaredExponential(pars_best, v)
        
        KInv, logK = self.cholInvert(K + epsilon * np.identity(nLambda))
        t = KInv + WTAInvW 
        tInv, logt = self.cholInvert(t)
        
        # get mean and covariance for line profile(s)
        mu = np.dot(tInv.dot(W.T), np.dot(AInv.toarray(), y))

        if plot_flag:
            y_sample = np.random.multivariate_normal(mu, tInv, size=20)
            plt.plot(y_sample.T, color="k", alpha=0.3)
            plt.plot(mu, color="k", lw=2)
            plt.errorbar(np.arange(len(mu)), mu, yerr=np.std(y_sample, axis=0), color="k", alpha=0.3)
            plt.show()

        return([mu, tInv])

    def marginal(self, pars, v, epsilon, nLambda, WTAInvW, P, Q, logA):
        """
        Compute the marginal posterior for the hyperparameters

        Parameters
        ----------
        pars : float array
            Value of the hyperparameters

        Returns
        -------
        logP : float
            Value of the marginal posterior
        """
        K = self.squaredExponential(pars, v)

    # We apply the Woodbury matrix identity to write the inverse in terms of the inverse of smaller matrices
        KInv, logK = self.cholInvert(K + epsilon * np.identity(nLambda))

        t = KInv + WTAInvW 
        tInv, logt = self.cholInvert(t)

    # And use the matrix determinant lemma
        logD = logt + logK + logA

        # First term in equation 22 is assumed to be constant
        # Second term in equation 22 has two parts:
        # -0.5 * Q is first term in equation 23
        # + 0.5 * np.dot(P.T, np.dot(tInv, P)) is the second term in equation 23
        # Third term in equation 23 is - 0.5 * logD
        logMarginal = -0.5 * Q + 0.5 * np.dot(P.T, np.dot(tInv, P)) - 0.5 * logD
        return -logMarginal

    #def _objDirect(x, user_data):
    #    return marginal(x)

    def optimizeDirectGP(self, *args, variance=None, pars=None):
        """
        Optimize the hyperparameters of the GP using the DIRECT optimization method

        Returns
        -------
        x: array of float
            Value of the optimal hyperpameters
        """
        l = [-30, np.log(np.min(variance) / len(variance))]
        u = [0.0, np.log(10.0*np.max(variance) / len(variance))]
        bounds = [(l[0], u[0]), (l[1], u[1])]
        res = minimize(self.marginal, pars, args=(args), bounds=bounds, method="SLSQP")
        print("Optimal lambdaGP={0:}, sigmaGP={1:}, loglambdaGP={2:}, logsigmaGP={3:} "
              .format(np.exp(res.x[0]), np.exp(res.x[1]), res.x[0], res.x[1]))
        return res.x

    def cholInvert(self, A):
        """
        Invert matrix A using a Cholesky decomposition. It works only for symmetric matrices.
        Input
        -----
        A: matrix to invert
        Output
        ------
        AInv: matrix inverse
        logDeterminant: returns the logarithm of the determinant of the matrix
        """
        L = np.linalg.cholesky(A)
        LInv = np.linalg.inv(L)
        AInv = np.dot(LInv.T, LInv)
        logDeterminant = 2.0 * np.sum(np.log(np.diag(L)))
        return AInv, logDeterminant

    def squaredExponential(self, pars, v):
        """
        Squared exponential covariance function

        Parameters
        ----------
        pars : float
            array of size 2 with the hyperparameters of the kernel

        Returns
        -------
        x : kernel matrix of size nl x nl
        """
        lambdaGP, sigmaGP = np.exp(pars)
        return sigmaGP * np.exp(-0.5 * lambdaGP * (v[:,None]-v[None,:])**2)        



    def leastSquareDeconvolutionMultiProfileMultiLambdaTaperingProfile(self, F, S, Lambda, n, TP=None, method="Second"):
        # n is the number of points in Z, for multi template case n = num_F * n, where num_F is the number of templates
        # TP is a n by N matrix for tapering profiles. The purpose is to separate profiles in velocity space
        F = np.array(F)
        num_F = len(F)
        Y_noise = self.flux
        # Creating a regularization matrix (http://arxiv.org/pdf/1008.5115v2.pdf)
        R0 = self.createTriDiagonalMatrix(n, method=method)
        if num_F != 1:
            for i in np.arange(num_F):
                #print(Lambda[i])
                if i == 0:
                    R = R0 * Lambda[i]
                else:
                    R = scipy.linalg.block_diag(R, R0 * Lambda[i])
        else:
            R = R0 * Lambda
        # Creating M matrix
        M = np.zeros((len(Y_noise), n * num_F))
        for i in np.arange(num_F):
            TP_rev = np.zeros((n,n))
            for j in np.arange(n):
                TP_rev[j,j] = 1.0 / TP[j,i]
            M[:,i*n:(i+1)*n] = np.dot(np.roll(scipy.linalg.toeplitz(F[i,:])[:, 0:n], int(-n/2), axis = 0), TP_rev)
        #print(np.shape(M), np.shape(F), np.shape(S), np.shape(Y_noise), np.shape(R))
        # LSD    
        #print(np.max(S))
        if np.max(S) > 1e-6:
            ACF = np.transpose(M).dot(S).dot(S).dot(M) + R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(S).dot(S).dot(Y_noise))
        else:
            ACF = np.transpose(M).dot(M) + R
            Z_noise = np.linalg.inv(ACF).dot(np.transpose(M).dot(Y_noise))
        return(Z_noise)

    def createErrorMatrix(self, err_arr):
        m = len(self.flux)
        S = np.zeros((m,m))
        if len(err_arr) != 1:
            err_arr_med = np.median(err_arr)
            for i in np.arange(m):
                #S[i, i] = np.max([1.0 / err_arr[i], 1e-3]) # for NIRSPEC_J0746_redux
                S[i, i] = np.max([1e-2 * err_arr_med / err_arr[i], 1e-6])
        else:
            np.fill_diagonal(S, 1/err_arr)
        return(S)

    def createErrorMatrix_old(self, SNR):
        m = len(self.flux)
        S = np.zeros((m,m))
        np.fill_diagonal(S, 1/SNR)
        return(S)

    def divideSpec(self, spec):
        spec.resampleSpec(self.wavelength)
        spec_tmp = self.copy()
        sepc_tmp.flux = self.flux / spec.flux
        return(spec_tmp)

    def improveTemplateSingleLine(self, spec_temp, Z_noise, plot_flag=False, minimum_flag=True, line_width_1=2e-6, line_width_2=1e-4):
        # remove nan
        self.removeNaN()
        spec_temp.removeNaN()
        # convolve spectrum with kernel Z_noise
        spec_convolv = spec_temp.convolveKernel(Z_noise)
        f_convolv = spec_convolv.flux
        #
        flx_obs = self.flux
        # resample spectrum to template
        self.resampleSpec(spec_temp.wavelength)
        # wavelength offset between template and observation
        wav = self.wavelength
        wav_delta = np.median(wav[1:-1] - wav[0:-2]) * (np.where(Z_noise == np.max(Z_noise))[0] - len(Z_noise) / 2.0)
        idx_delta = (np.where(Z_noise == np.max(Z_noise))[0] - len(Z_noise) / 2.0)[0]
        n = len(Z_noise)
        m = len(wav)
        # calculate ratio between flx_obs and f_convolv_new, a spike means a convolved line is shallower
        f_ratio = f_convolv / flx_obs
        if minimum_flag == False:
            idx = np.where(f_ratio == np.nanmax(f_ratio[int(n/2):int(m-n/2)]))[0][0]
            #idx_flx = np.where(((wav + wav_delta) > wav[idx] - 2e-5) & ((wav + wav_delta) < wav[idx] + 2e-5))
            idx_flx = np.where(((wav + wav_delta) > wav[idx] - line_width_1) & ((wav + wav_delta) < wav[idx] + line_width_1))
        else:
            idx = np.where(f_ratio == np.nanmin(f_ratio[int(n/2):int(m-n/2)]))[0][0]
            idx_flx = np.where(((wav + wav_delta) > wav[idx] - line_width_2) & ((wav + wav_delta) < wav[idx] + line_width_2))
            #idx_flx = np.where(((wav + wav_delta) > wav[idx] - 2e-5) & ((wav + wav_delta) < wav[idx] + 2e-5))
        spec_temp_new = spec_temp.copy()
        spec_temp_new.flux[idx_flx] = -0.2
        flx_temp_new = spec_temp_new.flux
        flx_temp = spec_temp.flux
        flx_local_max_dif = f_ratio[idx]
        # find the optimal depth for the maximum difference line
        f_convolv_new = spec_temp_new.convolveKernel(Z_noise).flux
        f_ratio_new = f_convolv_new / flx_obs
        flx_local_max_dif_new = f_ratio_new[idx]
        num_count = 0
        while (np.abs(flx_local_max_dif_new - 1.0) > 5e-3) & (num_count < 10):
            flx_temp_new[idx_flx] = flx_temp[int(idx - idx_delta)] - (flx_temp[int(idx - idx_delta)] - flx_temp_new[int(idx - idx_delta)]) / (flx_local_max_dif - flx_local_max_dif_new) * (flx_local_max_dif - 1.0)
            spec_temp_new.flux = flx_temp_new
            f_convolv_new = spec_temp_new.convolveKernel(Z_noise).flux
            f_ratio_new = f_convolv_new / flx_obs
            flx_local_max_dif_new = f_ratio_new[idx]
            num_count = num_count + 1
        print(np.std(f_ratio_new[int(n/2):int(m-n/2)]))
        if plot_flag == True:
            plt.plot(wav[int(n/2):int(m-n/2)], wav[int(n/2):int(m-n/2)] / wav[int(n/2):int(m-n/2)] * 1.5, "y")
            plt.plot(wav, f_ratio, "r")
            plt.plot(wav[idx], f_ratio[idx], "r.")
            plt.plot(wav, f_ratio_new, "b")
            plt.plot(wav + wav_delta, flx_temp_new, "b")
            plt.plot(wav + wav_delta, flx_temp, "r")
            plt.show()
        return(spec_temp_new)

    def improveTemplateMultiLine(self, spec_temp, S, Lambda, n, num_iteration, num_lines, flag_remove_emission=True, **kwargs):
        F = spec_temp.flux
        Z_noise = self.leastSquareDeconvolution(F, S, Lambda, n)
        n = len(Z_noise)
        m = len(spec_temp.flux)
        for i in np.arange(num_iteration):
            # correct for global trend difference
            f_convolv = spec_temp.convolveKernel(Z_noise).flux
            flx_correction = f_convolv[int(n/2):int(m-n/2)] / self.flux[int(n/2):int(m-n/2)]
            spec_temp.flux[int(n/2):int(m-n/2)] = spec_temp.flux[int(n/2):int(m-n/2)] / flx_correction
            F = spec_temp.flux
            Z_noise = self.leastSquareDeconvolution(F, S, Lambda, n)
            # correct for num_lines most dominant differences 
            for i in np.arange(num_lines):
                print(i)
                if ((not i % 5) & (flag_remove_emission == True)):
                    spec_temp = self.improveTemplateSingleLine(spec_temp, Z_noise, minimum_flag=True, **kwargs)
                spec_temp = self.improveTemplateSingleLine(spec_temp, Z_noise, minimum_flag=False, **kwargs)
        return(spec_temp)

    def resampleSpectoSpectrograph(self, pixel_sampling=3.0):
        # resample a spectrum to a wavelength grid that is determed by spectral resolution and pixel sampling rate
        # num_pixel_new = wavelength coverage range / wavelength per resolution element * pixel sampling rate
        num_pixel = len(self.wavelength)
        num_pixel_new = (np.nanmax(self.wavelength) - np.nanmin(self.wavelength)) / (np.nanmedian(self.wavelength) / self.spec_reso) * pixel_sampling
        wav_new = np.linspace(np.nanmin(self.wavelength), np.nanmax(self.wavelength), num = num_pixel_new)
        flx_new = np.interp(wav_new, self.wavelength, self.flux)
        return(Spectrum(wav_new, flx_new, spec_reso=self.spec_reso))

    def dopplerShift(self, rv_shift=0e0):
        #positive number means blue shift and vice versa
        beta = rv_shift / scipy.constants.c
        wav_shifted = self.wavelength * np.sqrt((1 - beta)/(1 + beta))
        #flx = np.interp(self.wavelength, wav_shifted, self.flux, left=np.nanmedian(self.flux), right=np.nanmedian(self.flux))
        flx = np.interp(self.wavelength, wav_shifted, self.flux, left=self.flux[0], right=self.flux[-1])
        flx[np.isnan(flx)] = np.nanmedian(flx)
        self.flux = flx
        return self

    def crossCorrelation(self, template, spec_mask=None, long_array=False, speed_flag=False):
        # positive peak means spectrum is blue shifted with respect to template
        # do not recommend long_array option. It does not produce the same SNR as the non-long_array option. 
        if not long_array:
            wav = self.wavelength
            flx = self.flux
            wav_temp = template.wavelength
            flx_temp = template.flux
            flx_temp = np.interp(wav, wav_temp, flx_temp)
            flx = flx - np.nanmedian(flx)
            flx_temp = flx_temp - np.nanmedian(flx_temp)
            if np.size(spec_mask) > 1:
                flx[spec_mask] = np.nanmedian(flx)
                flx_temp[spec_mask] = np.nanmedian(flx_temp)

            if speed_flag:
                num_pixels = len(wav)
                power_2 = np.ceil(np.log10(num_pixels + 0.0) / np.log10(2.0))
                num_pixels_new = 2.0**power_2
                wav_new = np.linspace(np.min(wav), np.max(wav), num_pixels_new)
                flx_new = np.interp(wav_new, wav, flx)
                flx_temp_new = np.interp(wav_new, wav, flx_temp)
                flx_temp = flx_temp_new
                flx = flx_new
                wav = wav_new

            cc = fp.ifft(fp.fft(flx_temp)*np.conj(fp.fft(flx)))
            ccf = fp.fftshift(cc)
            ccf = ccf - np.median(ccf)
            ccf = ccf.real 
    
            vel_int = np.nanmedian(np.abs(wav[1:-1] - wav[0:-2])) / np.nanmedian(wav) * scipy.constants.c
            nx = len(ccf)
            ccf = ccf / (nx + 0.0)
            vel = (np.arange(nx)-(nx-1)/2.0) * vel_int
        else:
            num_chunks = 4
            num_pixels = len(self.wavelength) 
            pix_chunk = int(np.floor(num_pixels / (num_chunks + 0.0)))
            for i in np.arange(num_chunks):
                spec_tmp = Spectrum(self.wavelength[i*pix_chunk:(i+1)*pix_chunk], self.flux[i*pix_chunk:(i+1)*pix_chunk])
                template_tmp = Spectrum(template.wavelength[i*pix_chunk:(i+1)*pix_chunk], template.flux[i*pix_chunk:(i+1)*pix_chunk])
                ccf_tmp = spec_tmp.crossCorrelation(template_tmp)
                if i == 0:
                    ccf_total = ccf_tmp
                else:
                    ccf_total = CrossCorrelationFunction(ccf_tmp.vel, ccf_tmp.ccf + ccf_total.ccf)
                vel = ccf_total.vel
                ccf = ccf_total.ccf             

        return(CrossCorrelationFunction(vel, ccf))

    def spectral_convolve(self, kernel):
        wav_inc_old = np.median(np.abs(kernel.wavelength[1:]-kernel.wavelength[0:-1]))
        wav_inc = np.median(np.abs(self.wavelength[1:]-self.wavelength[0:-1]))
        wav_new = np.arange(np.min(kernel.wavelength), np.max(kernel.wavelength), wav_inc)
        flx_new = np.interp(wav_new, kernel.wavelength, kernel.flux)
        flx_new = flx_new * wav_inc / wav_inc_old
        kernel = Spectrum(wav_new, flx_new)
        spec = self.copy()
        spec = spec.convolveKernel(kernel.flux)
        return(spec)

    def spectral_blur(self, rpower=1e5, quick_blur=False):
        # broaden a spectrum given its spectral resolving power
        if not quick_blur:
            wave = self.wavelength
            tran = self.flux
    
            wmin = wave.min()
            wmax = wave.max()
    
            nx = wave.size
            x  = np.arange(nx)
    
            A = wmin
            B = np.log(wmax/wmin)/nx
            wave_constfwhm = A*np.exp(B*x)
            tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
            dwdx_constfwhm = np.diff(wave_constfwhm)
            fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm
    
            fwhm_pix  = fwhm_pix[0]
            sigma_pix = fwhm_pix/2.3548
            kx = np.arange(nx)-(nx-1)/2.
            kernel = 1./(sigma_pix*np.sqrt(2.*np.pi))*np.exp(-kx**2/(2.*sigma_pix**2))
    
            tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
            tran_conv = fft.fftshift(tran_conv).real
            tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)
    
            self.wavelength = wave
            self.flux = tran_oldsampling
        else:
            pixel_to_sum = int(102400.0 / rpower)
            if pixel_to_sum >= 1.5:
                num_pixels = len(self.wavelength)
                num_pixels_new = int(np.floor((num_pixels + 0.0) / pixel_to_sum))
                wav = np.zeros((num_pixels_new,))
                flx = np.zeros((num_pixels_new,))
                for i in np.arange(num_pixels_new):
                    wav[i] = np.mean(self.wavelength[i*pixel_to_sum:(i+1)*pixel_to_sum])       
                    flx[i] = np.mean(self.flux[i*pixel_to_sum:(i+1)*pixel_to_sum])
                self.wavelength = wav
                self.flux = flx
        self.spec_reso = rpower
        return self

    def rotational_blur(self, rot_vel=3e4):
        # broaden a spectrum given the rotation of a target
        # kernel is a cosine function with only [-pi/2, pi/2] phase
        # -pi/2 phase corresponds to fwhm_pix for rpower of c / rot_vel
        wave = self.wavelength
        tran = self.flux

        wmin = wave.min()
        wmax = wave.max()

        nx = wave.size
        x  = np.arange(nx)

        A = wmin
        B = np.log(wmax/wmin)/nx
        wave_constfwhm = A*np.exp(B*x)
        tran_constfwhm = np.interp(wave_constfwhm, wave, tran)
        dwdx_constfwhm = np.diff(wave_constfwhm)
        rpower = scipy.constants.c / rot_vel
        fwhm_pix = wave_constfwhm[1:]/rpower/dwdx_constfwhm

        fwhm_pix  = fwhm_pix[0]
        sigma_pix = fwhm_pix/2.3548
        kx = np.arange(nx)-(nx-1)/2.
        kernel = np.cos(2.0 * np.pi * kx / (4.0 * fwhm_pix))
        idx = ((kx < -fwhm_pix) | (kx > fwhm_pix))
        kernel[idx] = 0.0
        kernel = kernel / np.sum(kernel)

        tran_conv = fft.ifft(fft.fft(tran_constfwhm)*np.conj(fft.fft(kernel)))
        tran_conv = fft.fftshift(tran_conv).real
        tran_oldsampling = np.interp(wave,wave_constfwhm,tran_conv)

        self.wavelength = wave
        self.flux = tran_oldsampling

        return self

    def getContinuum(self, poly_order=3, kernel_size=51, method="poly"):
        # method: poly and median
        if method == "poly":
            wav = self.wavelength
            flx = self.flux
            num = len(wav)
            x = np.arange(num)
            flx_med = np.median(flx[int(np.round(num/10)):-int(np.round(num/10))])
            flx_std = np.std(flx[int(np.round(num/10)):-int(np.round(num/10))])
            ind = np.where(np.abs(flx - flx_med) < (1.0 * flx_std))
            x_ind = x[ind]
            y_ind = flx[ind]
            coeffs = np.polyfit(x_ind, y_ind, poly_order)
            p = np.poly1d(coeffs)
            cont = p(x)
        elif method == "median":
            flx = self.flux
            cont = scipy.signal.medfilt(flx, kernel_size=kernel_size)
        return(cont)    

    def getSlope(self, poly_order=1):
        wav = self.wavelength
        flx = self.flux
        num = len(wav)
        x = np.arange(num)
        flx_med = np.median(flx[int(np.round(num/10)):-int(np.round(num/10))])
        flx_std = np.std(flx[int(np.round(num/10)):-int(np.round(num/10))])
        ind = np.where(np.abs(flx - flx_med) < (2.0 * flx_std))
        x_ind = np.hstack([x[ind][0:int(np.round(num/5))], x[ind][-int(np.round(num/5)):]])
        y_ind = np.hstack([flx[ind][0:int(np.round(num/5))], flx[ind][-int(np.round(num/5)):]])
        coeffs = np.polyfit(x_ind, y_ind, poly_order)
        p = np.poly1d(coeffs)
        cont = p(x)
        return(cont)

    def calcQFactor(self, sigma_readout = 10.0, plot_flag=False):
        wav = self.wavelength
        flx = self.flux
        # based on Bouchy et al. 2001: http://www.aanda.org/articles/aa/pdf/2001/29/aa1316.pdf
        #sigma_readout = 10.0 # in electrons 
        idx = np.argsort(wav)
        wav = wav[idx]
        flx = flx[idx]
        wav_delta = wav[1:] - wav[0:-1]
        flx_delta = flx[1:] - flx[0:-1]
        d_flx_d_wav = flx_delta / wav_delta
        W_arr = (wav[1:]**2 * d_flx_d_wav**2) / (flx[1:] + sigma_readout**2)
        A_arr = flx[1:]
        
        num = len(W_arr)
        idx = np.argsort(W_arr)
        W_arr = W_arr[idx][0:int(num*0.997)]
        A_arr = A_arr[idx][0:int(num*0.997)]
        
        W = np.sum(W_arr)
        A = np.sum(A_arr)
        if plot_flag:
            plt.hist((wav[1:]**2 * d_flx_d_wav**2) / (flx[1:] + sigma_readout**2), bins=2000)
        Q = np.sqrt(W / A)
        return(Q)

class CrossCorrelationFunction():
    def __init__(self, vel, ccf, un = None):
        self.vel = vel
        self.ccf = ccf
        self.un = un

    def getCCFchunk(self, vmin=-1e9, vmax=1e9):
        cc = self.ccf
        vel = self.vel
        idx = np.where((vel < vmax) & (vel > vmin))
        return(CrossCorrelationFunction(vel[idx], cc[idx]))

    def addNoise(self, un):
        if self.un is None:
            self.un = un
        else:
            print("CCF has already had noise added!")
        return self

    def pltCCF(self, save_fig=False, filename="tmp.png", **kwargs):
        plt.plot(self.vel, self.ccf, **kwargs)
        if save_fig:
            plt.savefig(filename)
        plt.show()

    def resampleCCF(self, vel_new):
        # resample a ccf to a new vel grid
        ccf_new = np.interp(vel_new, self.vel, self.ccf)
        self.vel = vel_new
        self.ccf = ccf_new
        if not(self.un is None):
            self.un = np.interp(vel_new, self.vel, self.un)
        return self

    def calcCentroid(self,cwidth=5, method="CenterMass", plot_flag=False):
        cc = self.ccf
        vel = self.vel
        
        #
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        cc = cc[idx]
        vel = vel[idx]
        #

        if method == "CenterMass":
            maxind = np.argmax(cc)
            mini = max([0,maxind-cwidth])
            maxi = min([maxind+cwidth+1,cc.shape[0]])
            weight = cc[mini:maxi] - np.min(cc[mini:maxi])
            centroid = (vel[mini:maxi]*weight).sum()/weight.sum()
            if plot_flag:
                plt.plot(vel, cc, "bo")
                plt.plot([centroid, centroid], [np.min(cc), np.max(cc)], "r--")
                plt.show()
        elif method == "PolyFit":
            maxind = np.argmax(cc)
            x = vel[maxind-cwidth:maxind+cwidth+1]
            y = cc[maxind-cwidth:maxind+cwidth+1]
            z = np.polyfit(x, y, 2)
            centroid = -z[1] / 2.0 / z[0]
            if plot_flag:
                plt.plot(x, y, "bo")
                plt.plot(x, np.poly1d(z)(x), "b")
                plt.show()
        elif method == "Bisector":
            print("CCF is smoothed first")
            ccf_new = self.smoothCCF(method="median_filter", kernel_size=31)
            [bis, y_val] = ccf_new.calculateLineBisector(plot_flag=plot_flag)
            centroid = np.median(bis)
        else:
            print("Please choose an approriate method: CenterMass or PolyFit.")
        return centroid

    def calcSNRrms(self, peak=None):
        cc = self.ccf

        # 
        vel = self.vel
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        cc_tmp = cc[idx]
        ind_max = np.argmax(cc_tmp) + idx[0][0]
        #

        #ind_max = np.argmax(cc)
        num = len(cc)
        if ind_max > (num / 2.0):
            ind_rms = [0, int(num / 4.0)]
        else:
            ind_rms = [-int(num / 4.0), -1]
        snr = cc[ind_max] / np.std(cc[ind_rms[0]:ind_rms[1]])
        if not (peak is None):
            snr = peak / np.std(cc[ind_rms[0]:ind_rms[1]])
        return(snr)

    def calcSNRrmsNoiseless(self, ccf_noise_less, peak=None):
        cc = self.ccf
        cc_subtracted = cc - ccf_noise_less.ccf
             
        ind_max = np.argmax(cc)
        num = len(cc)                
        snr = np.max([cc[ind_max] / np.std(cc_subtracted[0:int(num / 4.0)]), cc[ind_max] / np.std(cc_subtracted[-int(num / 4.0):-1])])                                 
        if not (peak is None):                           
            snr = np.max([peak / np.std(cc_subtracted[0:int(num / 4.0)]), peak / np.std(cc_subtracted[-int(num / 4.0):-1])])                                                               
        return(snr)

    def calcSNRnoiseLess(self, ccf_noise_less):
        cc = self.ccf
        cc_subtracted = cc - ccf_noise_less.ccf
        nx = len(cc) + 0.0

        # 
        vel = self.vel
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        cc_tmp = cc[idx]
        ind_max = np.argmax(cc_tmp) + idx[0][0]
        #

        #ind_max = np.argmax(cc)
        num = len(cc)
        if ind_max > (num / 2.0):
            ind_rms = [0, int(num / 4.0)]

            #
            ind_rms = [np.max(int(ind_max-nx/100), 0), np.min([int(ind_max+nx/100), int(nx-1)])]
            # 
        else:
            ind_rms = [-int(num / 4.0), -1]

            #
            ind_rms = [np.max(int(ind_max-nx/100), 0), np.min([int(ind_max+nx/100), int(nx-1)])]
            # 
        snr = cc[ind_max] / np.std(cc_subtracted[ind_rms[0]:ind_rms[1]])
        #snr = np.max([cc[ind_max] / np.std(cc_subtracted[0:int(num / 4.0)]), cc[ind_max] / np.std(cc_subtracted[-int(num / 4.0):-1])])
        return(snr)

    def smoothCCF(self, method="median_filter", kernel_size=5):
        cc = self.ccf
        vel = self.vel
        if method == "median_filter":
            cc_new = scipy.signal.medfilt(cc, kernel_size=kernel_size)
            ccf_new = CrossCorrelationFunction(vel, cc_new)
        if method == "gaussian_kernel":
            num = len(cc)
            kernel_x = np.arange(int(np.round(num / 10))) - int(np.round(num / 10)) / 2.0
            kernel_y = np.exp(-1.0 * (kernel_x)**2 / (2.0 * kernel_size)**2)
            kernel_y = kernel_y / np.sum(kernel_y)
            cc_new = np.convolve(cc, kernel_y, mode="same")
            cc_new[0:len(kernel_x)-1] = cc_new[len(kernel_x)]
            cc_new[-len(kernel_x):] = cc_new[-len(kernel_x)-1]
            ccf_new = CrossCorrelationFunction(vel, cc_new)
        if method == "none":
            ccf_new = CrossCorrelationFunction(vel, cc)
        return(ccf_new)

    def levelCCF(self):
        cc = self.ccf
        vel = self.vel
        cen = self.calcCentroid()
        ind_max = np.where(cc == np.max(cc))[0][0]
        num = len(cc)
        x = np.hstack([vel[0:int(np.round(ind_max-num / 10))], vel[int(np.round(ind_max+num / 10)):]])
        y = np.hstack([cc[0:int(np.round(ind_max-num / 10))], cc[int(np.round(ind_max+num / 10)):]])
        coeffs = np.polyfit(x, y, 1)
        p = np.poly1d(coeffs)
        y_fit = p(vel)
        cc_new = cc - y_fit + np.median(y_fit)
        ccf_new = CrossCorrelationFunction(vel, cc_new)
        return(ccf_new)

    def shiftCCF(self, vel_shift):
        cc = self.ccf
        vel = self.vel
        vel_new = vel - vel_shift
        cc_new = np.interp(vel_new, vel, cc)
        if self.un is None:
            ccf_new = CrossCorrelationFunction(vel, cc_new)
        else: 
            un = self.un
            un_new = np.interp(vel_new, vel, un)
            ccf_new = CrossCorrelationFunction(vel, cc_new, un=un_new)
        return(ccf_new)

    def yShiftCCF(self, y_shift):
        cc = self.ccf
        vel = self.vel
        cc_new = cc + y_shift
        ccf_new = CrossCorrelationFunction(vel, cc_new)
        return(ccf_new)

    def calcYShift(self, ccf_temp):
        cc = self.ccf
        vel = self.vel
        cc_dif = cc - ccf_temp.ccf
        return(np.median(cc_dif))

    def calcPeak(self):
        cc = self.ccf
        nx = len(cc) + 0.0

        vel = self.vel
        idx = np.where((vel < (scipy.constants.c / 20.0)) & (vel > (-scipy.constants.c / 20.0)))
        cc_tmp = cc[idx]
        ind_max = np.argmax(cc_tmp) + idx[0][0]

        return(cc[ind_max])

    def writeCCF(self, file_name="tmp.dat", python2=False):
        with open(file_name, "wb") as f:
            if not python2:
                for i in np.arange(len(self.vel)):
                    if self.un is None:
                        f.write(bytes("{0:20.8e}{1:20.8e}\n".format(self.vel[i], self.ccf[i]), 'UTF-8'))
                    else:
                        f.write(bytes("{0:20.8e}{1:20.8e}{2:20.8e}\n".format(self.vel[i], self.ccf[i], self.un[i]), 'UTF-8'))
            else:
                for i in np.arange(len(self.vel)):
                    if self.un is None:
                        f.write("{0:20.8e}{1:20.8e}\n".format(self.vel[i], self.ccf[i]))
                    else:
                        f.write("{0:20.8e}{1:20.8e}{2:20.8e}\n".format(self.vel[i], self.ccf[i], self.un[i]))

    def calculateLineBisector(self, plot_flag=False):
        vel = self.vel
        cc = self.ccf
        cc_max = self.calcPeak()
        idx_max = np.where(cc == np.max(cc))[0][0]
        num = 20.0

        cc1 = cc[0:idx_max+1][::-1]
        vel1 = vel[0:idx_max+1][::-1]
        cc1_diff = np.diff(cc1)
        if np.array(np.where(cc1_diff > 0)).any():
            idx_sign_change=np.where(cc1_diff > 0)[0][0]
        else:
            idx_sign_change=len(cc1)
        y = cc1[0:idx_sign_change+1]
        x = vel1[0:idx_sign_change+1]
        idx_y = np.argsort(y)
        x = x[idx_y]
        y = y[idx_y]
        y_interp_1 = (np.arange(num)+0.0)/num*np.max(y)
        x_interp_1 = np.interp(y_interp_1, y, x) # y has to be in ascending order in interpolation 

        cc2 = cc[idx_max:]
        vel2 = vel[idx_max:]
        cc2_diff = np.diff(cc2)
        if np.array(np.where(cc2_diff > 0)).any():
            idx_sign_change=np.where(cc2_diff > 0)[0][0]
        else:
            idx_sign_change=len(cc2)
        y = cc2[0:idx_sign_change+1]
        x = vel2[0:idx_sign_change+1]
        idx_y = np.argsort(y)
        x = x[idx_y]
        y = y[idx_y]
        y_interp_2 = (np.arange(num)+0.0)/num*np.max(y)
        x_interp_2 = np.interp(y_interp_2, y, x)

        if plot_flag == True:
            plt.plot(vel, cc, "bo-")
            plt.plot((x_interp_1 + x_interp_2) / 2.0, y_interp_2, "rx")
            plt.show()
        bis = (x_interp_1 + x_interp_2) / 2.0
        y_val = y_interp_2
        return([bis, y_val])

    def crossCorrelation(self, template, spec_mask=None, long_array=False, speed_flag=False):
        # positive peak means spectrum is blue shifted with respect to template
        # do not recommend long_array option. It does not produce the same SNR as the non-long_array option. 
        if not long_array:
            wav = self.vel
            flx = self.ccf
            wav_temp = template.vel
            flx_temp = template.ccf
            flx_temp = np.interp(wav, wav_temp, flx_temp)
            flx = flx - np.nanmedian(flx)
            flx_temp = flx_temp - np.nanmedian(flx_temp)
            if spec_mask != None:
                flx[spec_mask] = np.nanmedian(flx)
                flx_temp[spec_mask] = np.nanmedian(flx_temp)

            if speed_flag:
                num_pixels = len(wav)
                power_2 = np.ceil(np.log10(num_pixels + 0.0) / np.log10(2.0))
                num_pixels_new = 2.0**power_2
                wav_new = np.linspace(np.min(wav), np.max(wav), num_pixels_new)
                flx_new = np.interp(wav_new, wav, flx)
                flx_temp_new = np.interp(wav_new, wav, flx_temp)
                flx_temp = flx_temp_new
                flx = flx_new
                wav = wav_new

            cc = fp.ifft(fp.fft(flx_temp)*np.conj(fp.fft(flx)))
            ccf = fp.fftshift(cc)
            ccf = ccf - np.median(ccf)
            ccf = ccf.real 
    
            vel_int = np.nanmedian(np.abs(wav[1:-1] - wav[0:-2]))
            nx = len(ccf)
            ccf = ccf / (nx + 0.0)
            vel = (np.arange(nx)-(nx-1)/2.0) * vel_int

        return(CrossCorrelationFunction(vel, ccf))


class Atmosphere():
    def __init__(self, spec_tran_path=None, spec_radi_path=None, radial_vel=1e1):
        self.spec_tran_path = spec_tran_path
        self.spec_radi_path = spec_radi_path
        self.radial_vel = radial_vel
        if self.spec_tran_path != None:
            with open(spec_tran_path, "rb") as handle:
                [self.spec_tran_wav, self.spec_tran_flx] = pickle.load(handle) 
        else:
            self.spec_tran_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_tran_flx = np.zeros(np.shape(self.spec_tran_wav)) + 1.0
        if self.spec_radi_path != None:
            self.spec_radi_data = ascii.read(spec_radi_path)
            self.spec_radi_wav = self.spec_radi_data["col1"][:] # in nm
            self.spec_radi_wav = self.spec_radi_wav / 1e3 # now in micron
            self.spec_radi_flx = self.spec_radi_data["col2"][:] # in ph/s/arcsec**2/nm/m**2
            self.spec_radi_flx = self.spec_radi_flx * 1e3 # now in ph/s/arcsec**2/micron/m**2
            self.spec_radi_wav = np.hstack([np.arange(0.1, 0.9, 1e-5), self.spec_radi_wav]) # to avoid missing information in optical below 0.9 micron
            self.spec_radi_flx = np.hstack([np.zeros(np.shape(np.arange(0.1, 0.9, 1e-5))) + 1e-99, self.spec_radi_flx])
        else:
            self.spec_radi_wav = np.arange(0.1, 5.0, 1e-5)
            self.spec_radi_flx = np.zeros(np.shape(self.spec_radi_wav)) + 1e-99

    def getTotalSkyFlux(self, wav_min, wav_max, tel_size=10.0, multiple_lambda_D=1.0, t_exp=1e3, eta_ins=0.1):
        # get total flux of sky emission
        idx = ((self.spec_radi_wav < wav_max) & (self.spec_radi_wav > wav_min))
        wav = self.spec_radi_wav[idx]
        flx = self.spec_radi_flx[idx]
        wav_int = np.abs(wav[1:-1] - wav[0:-2])
        fiber_size = np.nanmedian(wav) * 1e-6 / tel_size / np.pi * 180.0 * 3600.0
        fiber_size = fiber_size * multiple_lambda_D # multiple times lambda / D
        flx_skybg_total = np.sum(flx[0:-2] * t_exp * fiber_size **2 * wav_int * np.pi * (tel_size / 2.0)**2) * eta_ins
        
        return(flx_skybg_total)

class PhoenixSpec():
    def __init__(self, file_name=None):
        self.file_name=file_name

    def readPHOENIXSpec(self):
        if self.file_name == None:
            print("No file name")
            spec_tmp = None
        else:
            spec = pyfits.open(self.file_name)
            wav = spec[1].data["Wavelength"]
            flx = spec[1].data["Flux"]
            spec_tmp = Spectrum(wav, flx)
        return(spec_tmp)

class RotatingTarget(Target):
    def __init__(self, ld_a=1.68, ld_b=-0.83, **kwargs):
        ##**kwargs: distance=10.0, spec_path=None, inclination_deg=90.0, rotation_vel=5e3, radial_vel=1e4, spec_reso=1e5
        Target.__init__(self ,**kwargs)
        self.limb_darkening_para = [ld_a, ld_b]

    def generateIntegratedSpectrum(self, u_obs, plot_star_surface = False, wav_min = 2.3, wav_max = 2.31, longitude_number = 20j, latitude_number = 10j, lsf=None):
        # render the sphere mesh
        u, v = np.mgrid[0:2*np.pi:longitude_number, 0:np.pi:latitude_number]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)

        v_obs = self.inclination_deg / 180.0 * np.pi
        x_obs=np.cos(u_obs)*np.sin(v_obs)
        y_obs=np.sin(u_obs)*np.sin(v_obs)
        z_obs=np.cos(v_obs)

        los_angle = np.arccos(np.transpose(np.inner(np.transpose([x,y,z]),[x_obs,y_obs,z_obs])))
        ind = np.where(los_angle < np.pi/2)

        rv_angle = np.arccos(np.inner(np.transpose([((-1)*np.sin(u)),np.cos(u),0]),[x_obs,y_obs,z_obs]))
        rv_grid = self.rotation_vel * np.sin(v) * np.cos(rv_angle)
        if 1 == 0:
            plt.imshow(np.transpose(rv_grid), interpolation="none", origin="upper", aspect="auto", extent=[0, 360, -90, 90])
            plt.colorbar()
            plt.show()

        #Limb Darkening (Claret et al. 2012, Equation 2 for quadratic darkening, VizieR data for Teff=1500 and log(g)=5.5)
        ld_a = self.limb_darkening_para[0]
        ld_b = self.limb_darkening_para[1]
        ld_grid = 1 - ld_a * (1 - np.cos(los_angle)) - ld_b * (1 - np.cos(los_angle))**2
        if 1 == 0:
            plt.imshow(np.transpose(ld_grid), interpolation="none", origin="upper", aspect="auto", extent=[0, 360, -90, 90])
            plt.colorbar()
            plt.show()

        #Read in unshifted spectrum and initialize integrated spectrum
        spec_chunk = self.spectrum.getChunk(wav_min, wav_max) 
        wav_unshifted = spec_chunk.wavelength
        flx_unshifted = spec_chunk.flux / np.max(spec_chunk.flux) # maximum normalized to unity
        flx_integral = np.zeros((len(flx_unshifted)))
        flx_integral_ld = np.zeros((len(flx_unshifted)))

        #Integrate the visible hemisphere to generate spectrum
        flx_integral_ld_arr = np.zeros((int(np.abs(longitude_number)),int(np.abs(latitude_number)),len(flx_unshifted)))

        surface_area = 0.0
        ld_weight = 0.0
        d_u = 2*np.pi / np.abs(longitude_number)
        d_v = np.pi / np.abs(latitude_number)
        for i in np.arange(len(ind[0])):
            surface_area_patch = np.sin(v[ind[0][i],ind[1][i]]) * d_u * d_v * np.cos(los_angle[ind[0][i],ind[1][i]])
            surface_area = surface_area + surface_area_patch
            ld_weight = ld_weight + ld_grid[ind[0][i],ind[1][i]] * surface_area_patch
            rv_shift = rv_grid[ind[0][i],ind[1][i]]
            spec_tmp = spec_chunk.copy()
            spec_tmp.dopplerShift(rv_shift=rv_shift+self.radial_vel)
            if lsf != None:
                spec_tmp = spec_tmp.spectral_convolve(lsf)
            flx_integral = flx_integral + spec_tmp.flux * surface_area_patch 
            flx_integral_ld = flx_integral_ld + spec_tmp.flux * surface_area_patch * ld_grid[ind[0][i],ind[1][i]]
            flx_integral_ld_arr[ind[0][i],ind[1][i],:] = spec_tmp.flux * surface_area_patch * ld_grid[ind[0][i],ind[1][i]]
            #plt.plot(flx_integral_ld_arr[ind[0][i],ind[1][i],:])
        #plt.show()
        flx_integral = flx_integral / surface_area 
        flx_integral_ld = flx_integral_ld / ld_weight
        if 1 == 0:
            plt.plot(flx_integral)
            plt.show()
        if (plot_star_surface == True):
            plt.rcParams['figure.figsize'] = (15.0, 15.0)
            plt.rcParams.update({'font.size': 20})
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_aspect("equal")
            #ax.view_init(45,165)
            ax.view_init(v_obs / np.pi * 180, u_obs / np.pi * 180)
            ax.plot_wireframe(x, y, z, color="black", linewidth=0.1)
            plt.axis('on')
            cmhot = plt.get_cmap("RdBu")
            cax = ax.scatter(x[ind],y[ind],z[ind],rv_grid[ind],s=50,c=rv_grid[ind],cmap=cmhot,vmin=np.min(rv_grid),vmax=np.max(rv_grid))
            plt.colorbar(cax, shrink=0.5, aspect=5)
            plt.show()
        return {'Wavelength':wav_unshifted, 'Flux_GridPoint':flx_unshifted, \
                'Flux_LD':flx_integral_ld, \
                "Flux_LD_grid":flx_integral_ld_arr, \
                "Normalization": ld_weight, "Visible_Index": ind} 

    def generateSpotedSpectrum(self, wav_unshifted, flx_integral_ld_arr, ld_weight, spot_map, idx_visible):
        ind = idx_visible
        flx_integral_ld = np.zeros((len(flx_integral_ld_arr[0,0,:]),))
        spot_map_norm = 0.0
        for i in np.arange(len(ind[0])):
            spot_map_norm = spot_map_norm + spot_map[ind[0][i],ind[1][i]]
            flx_integral_ld = flx_integral_ld + flx_integral_ld_arr[ind[0][i],ind[1][i],:] * spot_map[ind[0][i],ind[1][i]]
        flx_integral_ld = flx_integral_ld / ld_weight
        return(Spectrum(wav_unshifted, flx_integral_ld))

class FitsImage2d():
    def __init__(self, file_name=None, image_type=None, **kwargs):
        self.file_name=file_name
        self.image_type=image_type
    def readImage(self):
        im = pyfits.open(self.file_name)
        im_data = im[0].data
        return(im_data)

    def removeBadPixels(self, im_data=None, sigma_clip=10.0): # need to check
        sigma = np.sqrt(np.abs(im_data)+1e-9)
        img_medfilt = scipy.signal.medfilt(im_data,3)
        sigma = np.nanstd(np.sort(im_data.flatten())[int(0.1*np.size(im_data)):int(0.9*np.size(im_data))])
        print("std=", sigma)
        badimg = np.abs((im_data - img_medfilt))/sigma > sigma_clip
        img_rectc = im_data.copy()
        img_rectc[badimg] = img_medfilt[badimg]
        return(np.abs((im_data - img_medfilt))/sigma)

class PharoImage2d(FitsImage2d):
    def __init__(self, file_name=None, image_type=None, **kwargs):
        self.file_name=file_name
        self.image_type=image_type
    def readImage(self):
        im = pyfits.open(self.file_name)
        im_data = im[0].data[3,:,:]
        return(im_data)

class FitsList():
    def __init__(self, file_list=None, image_type=None, instrument_name=None, **kwargs):
        self.file_list = file_list
        self.instrument_name = instrument_name 
        self.image_type = image_type 
        self.image_arr = self.getImageArr()

    def getImageArr(self, slice_num=3):
        if self.file_list is not None:
            with open(self.file_list, 'r') as f:
                file_list = f.read().split("\n")
            file_list_num = len(file_list)
            if self.instrument_name == "pharo":
                count_num = 0
                for idx, file_name in enumerate(file_list):
                    if file_name != "":
                        im = PharoImage2d(file_name=file_name, image_type=self.image_type, slice_num=slice_num).readImage()
                        if idx == 0:
                            im_arr = np.zeros((np.shape(im)[0], np.shape(im)[1], file_list_num))
                        im_arr[:,:,idx] = im
                    count_num = count_num + 1
        return(im_arr[:,:,0:count_num-1])


class AB_Pattern(FitsImage2d):
    def __init__(self, image_orientation="horizontal", **kwargs):
        FitsImage2d.__init__(self, **kwargs)
        self.image_orientation=image_orientation
        self.im_data = self.readImage()

    def transposeImage(self):
        if self.image_orientation == "vertical":
            self.im_data = np.transpose(self.im_data)
        elif self.image_orientation == "horizontal":
            self.im_data = self.im_data
        else:
            print("Please check image_orientation keyword: need to be either horizontal or vertical.")

    def getYdistribution(self):
        y_cut = np.median(self.im_data, axis=1)
        return(y_cut)

    def getPosSpec(self, flag_sum=False):
        y_cut = self.getYdistribution()
        y_med = np.median(y_cut)
        y_cut = y_cut - y_med
        y_max = np.max(y_cut)
        ind = np.where(y_cut < y_max / 10.0)
        if not flag_sum:
            y_cut[:] = 1.0
            y_cut[ind] = 0.0
            flx = np.dot(np.transpose(self.im_data), y_cut) 
            #flx = flx / np.median(flx) * np.sum(y_cut)
        else:
            y_cut[:] = 1.0
            y_cut[ind] = 0.0 
            flx = np.dot(np.transpose(self.im_data), y_cut)
        return(flx)

    def getSpecInd(self, ind, flag_apodize=False, flag_med=False):
        y_cut = self.getYdistribution()
        if not flag_apodize:
            y_cut[:] = 0.0
            y_cut[ind] = 1.0
        else:
            cen = np.median(ind)
            wid = len(ind) + 0.0
            y_cut[:] = 0.0
            y_cut[ind] = np.exp(-(ind - cen)**2 / 2.0 / (wid / 2.3 / 3.0)**2)
        if not flag_med:
            flx = np.dot(np.transpose(self.im_data), y_cut)
        else:
            flx = np.median(self.im_data[ind, :], axis=0)    
        return(flx)
        
    def getNegSpec(self, flag_sum=False):
        y_cut = self.getYdistribution()
        y_med = np.median(y_cut)
        y_cut = y_cut - y_med
        y_min = np.min(y_cut)
        ind = np.where(y_cut > y_min / 10.0)
        if not flag_sum:
            y_cut[:] = 1.0
            y_cut[ind] = 0.0
            flx = np.dot(np.transpose(self.im_data), y_cut) 
            #flx = flx / np.median(flx) * np.sum(y_cut)
        else:
            y_cut[:] = 1.0
            y_cut[ind] = 0.0 
            flx = np.dot(np.transpose(self.im_data), y_cut)
        return(flx)

    def getCombinedSpec(self):
        flx_pos = self.getPosSpec()
        flx_neg = self.getNegSpec()
        #plt.plot(flx_pos)
        #plt.plot(flx_neg)
        #plt.show()
        flx = flx_pos - flx_neg
        return(flx)

    def DSFunc(self, pixels, Coefs):
        if np.size(Coefs) >= 2:
            WaveOut = np.zeros(len(pixels))
            for i in range(0,len(Coefs)):
                    WaveOut = WaveOut + Coefs[i]*(pixels/1000.0)**i
        else:
            WaveOut = np.nan
        return(WaveOut)

class SphereMap():
    def __init__(self, longitude_number = 20j, latitude_number = 10j, flux_map=None):
        self.longitude_number = longitude_number
        self.latitude_number = latitude_number
        self.flux_map = flux_map
        # render the sphere mesh
        self.u, self.v = np.mgrid[0:2*np.pi:self.longitude_number, 0:np.pi:self.latitude_number]
        self.x=np.cos(self.u)*np.sin(self.v)
        self.y=np.sin(self.u)*np.sin(self.v)
        self.z=np.cos(self.v)
        
    def pltVisiblePart(self, ind):
        f = np.zeros(np.shape(self.u))
        f[ind] = 1.0
        plt.imshow(np.transpose(f), interpolation="none", origin="upper", aspect="auto", extent=[0, 360, -90, 90])
        plt.show()

    def pltMap(self):
        if np.size(self.flux_map) != 1:
            plt.imshow(np.transpose(self.flux_map), interpolation="none", origin="upper", aspect="auto", extent=[0, 360, -90, 90], vmin=0, vmax=1)
            plt.colorbar()
            plt.show()
        else:
            print("Warning: flux_map is None")

    def calcVisibleIndex(self, u_obs, v_obs):
        # u_obs and v_obs are in the unit of radian
        x_obs=np.cos(u_obs)*np.sin(v_obs)
        y_obs=np.sin(u_obs)*np.sin(v_obs)
        z_obs=np.cos(v_obs)
        los_angle = np.arccos(np.transpose(np.inner(np.transpose([self.x,self.y,self.z]),[x_obs,y_obs,z_obs])))
        ind = np.where(los_angle < np.pi/2)
        return(ind)

    def calcSingleSpotMap(self, surface_map, u_spot, v_spot, spot_size, spot_contrast):
        # u_obs and v_obs are in the unit of radian
        x_spot=np.cos(u_spot)*np.sin(v_spot)
        y_spot=np.sin(u_spot)*np.sin(v_spot)
        z_spot=np.cos(v_spot)
        spot_angle = np.arccos(np.transpose(np.inner(np.transpose([self.x,self.y,self.z]),[x_spot,y_spot,z_spot])))
        spot_grid = np.where(spot_angle < spot_size, spot_angle, 1)
        spot_grid = np.where(spot_grid == 1, spot_grid, 0) # see the broadcasting trick from: http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
        ind_spot_map = np.where(spot_grid == 0)
        surface_map[ind_spot_map] = spot_contrast
        return(surface_map)

class RadialVelocity():
# code from http://radvel.readthedocs.io/en/master/
    def __init__(self, orbital_elements):
        self.orbital_elements = orbital_elements
        # per, tp, e, om, K 

    def rv_drive(self, t, use_C_kepler_solver=False):
        """RV Drive
        
        Args:
            t (array of floats): times of observations
            orbel (array of floats): [per, tp, e, om, K].\
                Omega is expected to be\
                in degrees
            use_C_kepler_solver (bool): (default: True) If \
                True use the Kepler solver written in C, else \
                use the Python/NumPy version.

        Returns:
            rv: (array of floats): radial velocity model
        
        """
        orbel = self.orbital_elements    

        # unpack array
        per, tp, e, om, k = orbel
        om = om / 180 * np.pi
        
        # Error checking
        if e == 0.0:
            M = 2 * np.pi * ( ((t - tp) / per) - np.floor( (t - tp) / per ) )
            return k * np.cos( M + om )
        
        if per < 0: per = 1e-4
        if e < 0: e = 0
        if e > 0.99: e = 0.99


        # Calculate the approximate eccentric anomaly, E1, via the mean anomaly  M.
        if use_C_kepler_solver:
            rv = _kepler.rv_drive_array(t, per, tp, e, om, k)
        else:
            M = 2 * np.pi * ( ((t - tp) / per) - np.floor( (t - tp) / per ) )
            eccarr = np.zeros(t.size) + e
            E1 = self.kepler(M, eccarr)
            # Calculate nu
            nu = 2 * np.arctan( ( (1+e) / (1-e) )**0.5 * np.tan( E1 / 2 ) )
            # Calculate the radial velocity
            rv = k * ( np.cos( nu + om ) + e * np.cos( om ) ) 
        
        return rv
        
    def kepler(self, inbigM, inecc):
        """Solve Kepler's Equation

        Args:
            inbigM (array): input Mean annomaly
            inecc (array): eccentricity

        Returns:
            eccentric annomaly: array
        
        """
        
        Marr = inbigM  # protect inputs; necessary?
        eccarr = inecc
        conv = 1.0e-12  # convergence criterion
        k = 0.85

        Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr  # first guess at E
        # fiarr should go to zero when converges
        fiarr = ( Earr - eccarr * np.sin(Earr) - Marr)  
        convd = np.abs(fiarr) > conv  # which indices have not converged
        nd = np.sum(convd == True) # number of converged elements
        count = 0

        while nd > 0:  # while unconverged elements exist
            count += 1
            
            M = Marr[convd]  # just the unconverged elements ...
            ecc = eccarr[convd]
            E = Earr[convd]

            fi = fiarr[convd]  # fi = E - e*np.sin(E)-M    ; should go to 0
            fip = 1 - ecc * np.cos(E) # d/dE(fi) ;i.e.,  fi^(prime)
            fipp = ecc * np.sin(E) # d/dE(d/dE(fi)) ;i.e.,  fi^(\prime\prime)
            fippp = 1 - fip  # d/dE(d/dE(d/dE(fi))) ;i.e.,  fi^(\prime\prime\prime)

            # first, second, and third order corrections to E
            d1 = -fi / fip 
            d2 = -fi / (fip + d1 * fipp / 2.0)
            d3 = -fi / (fip + d2 * fipp/ 2.0 + d2 * d2 * fippp / 6.0) 
            E = E + d3
            Earr[convd] = E
            fiarr = ( Earr - eccarr * np.sin( Earr ) - Marr) # how well did we do?
            convd = np.abs(fiarr) > conv  #test for convergence
            nd = np.sum(convd == True)
            
        if Earr.size > 1: 
            return Earr
        else: 
            return Earr[0]

def semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter'):
    """
    Compute Doppler semi-amplitude

    :param Msini: mass of planet [Mjup]
    :type Msini: float
    
    :param P: Orbital period [days]
    :type P: float
    
    :param Mtotal: Mass of star + mass of planet [Msun]
    :type Mtotal: float
    
    :param e: eccentricity
    :type e: float

    :param Msini_units: Units of returned Msini. Must be 'earth', or 'jupiter' (default 'jupiter').
    :type Msini_units: string

    :return: Doppler semi-amplitude [m/s]
    """

    K_0 = 28.4329

    if Msini_units.lower() == 'jupiter':
        K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**-0.333 * \
            Mtotal**-0.667
    elif Msini_units.lower() == 'earth':
        K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**-0.333 * \
            Mtotal**-0.667*(astropy.constants.M_earth/astropy.constants.M_jup).value
    elif Msini_units.lower() == 'sun':
        K = K_0 * ( 1 - e**2 )**-0.5 * Msini * ( P / 365.0 )**-0.333 * \
            Mtotal**-0.667*(astropy.constants.M_sun/astropy.constants.M_jup).value
    else: 
        raise Exception("Msini_units must be 'earth', or 'jupiter', or 'sun'")
        
    return K

def readSpec(file_name="tmp.dat"):
    tmp = ascii.read(file_name)
    if len(tmp.keys()) == 2:
        spec = Spectrum(tmp["col1"], tmp["col2"])
    elif len(tmp.keys()) == 3:
        spec = Spectrum(tmp["col1"], tmp["col2"])
        spec.addNoise(tmp["col3"])
    else:
        spec = None
        print("Warning: file needs to have 2 or 3 columns.")
    return(spec)

class MEM():
    def __init__(self, R_origin=None, R=None, D=None, D_un=None,  m=None, x_dim=None, y_dim=None):
        self.R = R
        self.R_origin = R_origin
        self.D = D
        self.D_un = D_un
        self.m = m
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.response_map = self.calResponseMap()
        self.m = self.m * self.response_map + 1e-9
        
    def array_2d_1d(self, array_2d):
        #array_2d = np.transpose(array_2d)
        array_1d = np.zeros((np.shape(array_2d)[0]*np.shape(array_2d)[1]))
        for ii in np.arange(np.shape(array_2d)[0]):
            for iii in np.arange(np.shape(array_2d)[1]):
                iiii = iii + np.shape(array_2d)[1] * ii
                array_1d[iiii] = array_2d[ii,iii]
        return(array_1d)

    def array_1d_2d(self, array_1d,x_dim,y_dim):
        array_2d = np.zeros((x_dim,y_dim))
        for iiii in np.arange(len(array_1d)):
            iii = iiii % y_dim
            ii = int((iiii - iii) / y_dim)
            array_2d[ii,iii] = array_1d[iiii]
        return(array_2d)

    def calResponseMap(self):
        response_map = np.zeros(np.shape(self.m))
        for i in np.arange(len(self.m)):
            im_init_tmp = np.zeros(np.shape(self.m)) + self.m
            im_init_tmp[i] = im_init_tmp[i] - 0.1
            if np.sum(np.abs(self.R.dot(im_init_tmp) - self.R.dot(self.m))) != 0:
                response_map[i] = 1
        return(response_map)

    def pltSpecData(self):
        for i in np.arange(np.shape(self.R_origin)[3]):
            plt.plot(self.D[i*np.shape(self.R_origin)[2]:(i+1)*np.shape(self.R_origin)[2]][:])
        plt.show()
 
    def pltComparisonImage(self, im1, im2):
        # im1 - output im2 - input
        x = np.arange(self.x_dim)
        x_lables = ["{0:3.0f}".format(x[i] / (np.max(x)+1.0) * 360.0) for i in np.arange(len(x))]
        y = np.arange(self.y_dim)
        y_lables = ["{0:3.0f}".format(- ((y[i] + 0.5) / (np.max(y)+1.0) * 180.0 - 90.0)) for i in np.arange(len(y))]

        ax1 = plt.subplot2grid((3,7), (0,0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid((3,7), (0,3), colspan=3, rowspan=2)
        ax3 = plt.subplot2grid((3,7), (2,0), colspan=7)

        #im = ax1.imshow(np.transpose(array_1d_2d(h_init,np.shape(im_array_T)[0],np.shape(im_array_T)[1])), \
        im = ax1.imshow(np.transpose(self.array_1d_2d(im1, self.x_dim, self.y_dim)), aspect="auto", cmap='winter', vmin=0.0,vmax=1.0, interpolation='none')
        #plt.colorbar(im)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_lables, rotation="vertical")
        ax1.set_yticks(y)
        ax1.set_yticklabels(y_lables)
        ax1.set_xlabel("Longitude [degree]")
        ax1.set_ylabel("Latitude [degree]")

        #im = ax2.imshow(np.transpose(array_1d_2d(h * response_map,np.shape(im_array_T)[0],np.shape(im_array_T)[1])),aspect="auto", cmap='winter', vmin=0.0,vmax=np.max(im_array), interpolation='none')
        im = ax2.imshow(np.transpose(self.array_1d_2d(im2, self.x_dim, self.y_dim)), aspect="auto", cmap='winter', vmin=0.0,vmax=1.0, interpolation='none')
        #im = ax2.imshow(im_array,aspect="auto", cmap='winter', vmin=0.0,vmax=np.max(im_array), interpolation='none')
        #plt.colorbar(im, cax=ax2)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_lables, rotation="vertical")
        ax2.set_yticks(y)
        ax2.set_yticklabels(y_lables)
        ax2.set_xlabel("Longitude [degree]")
        ax2.set_ylabel("Latitude [degree]")

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.85, 0.45, 0.045, 0.5])
        plt.colorbar(im, cax=cax)

        idx = np.where(self.response_map == 1.0)
        ax3.plot(im1[idx] - im2[idx],".", markersize=10)
        ax3.annotate("{0:6s} {1:5.2f}".format("RMS = ", np.std(im1[idx] - im2[idx])), xy=(15, 0.05))
        ax3.set_xlabel("Grid Number")
        ax3.set_ylabel("Output - Input")
        ax3.set_ylim(-3.0 * np.std(im1[idx] - im2[idx]), 3.0 * np.std(im1[idx] - im2[idx]))
        ax3.yaxis.set_ticks(np.arange(-0.4, 0.5, 0.2))
        plt.tight_layout()
        plt.savefig("tmp.eps", format="eps", dpi = 100)
        plt.show()

        D_init = self.R.dot(im1) 
        for i in np.arange(np.shape(self.R_origin)[3]):
            plt.plot(i*2 + ((D_init-self.D)/self.D_un)[i*np.shape(self.R_origin)[2]:(i+1)*np.shape(self.R_origin)[2]][:],".")    
        plt.show()
    
        D_init = self.R.dot(self.m)
        for i in np.arange(np.shape(self.R_origin)[3]):
            plt.plot(((D_init-self.D)/self.D_un)[i*np.shape(self.R_origin)[2]:(i+1)*np.shape(self.R_origin)[2]][:]) 
        plt.show()
 
    def pltImageData(self, im_1d, vmin=0, vmax=1):
        plt.imshow(np.transpose(self.array_1d_2d(im_1d, self.x_dim, self.y_dim)), aspect="auto", cmap="hot", vmin=vmin, vmax=vmax, interpolation="None")
        plt.colorbar()
        plt.show()

    def Create_DS(self, theta, method="classic"):
        #print("Create_DS", time.clock())
        DS = np.zeros(len(self.m))
        if method == "classic" or method == "classic_auto_scaling":
            for i in np.arange(len(self.m)):
                #DS[i] = 1 - 1/np.log(10) - np.log10(theta[i] / self.m[i])
                DS[i] = -np.log(theta[i] / self.m[i])
        elif method == "bound_0_1":
            for i in np.arange(len(self.m)):
                #DS[i] = - np.log10(theta[i] / self.m[i]) + np.log10((1.0-theta[i]) / (1.0-self.m[i]))
                DS[i] = - np.log(theta[i] / self.m[i]) + np.log((1.0-theta[i]) / (1.0-self.m[i]))
        return(DS)

    def Create_DL(self, theta, sigma_inv, C=1.0):
        #print("Create_DL", time.clock())
        DL = np.zeros(len(theta))
        sigma_inv_diag = np.diag(sigma_inv)
        RC = self.R.dot(C)
        for i in np.arange(len(theta)):
            #DL[i] = np.sum(2.0 * (self.D - self.R.dot(theta)).dot(sigma_inv**2).dot(-self.R[:,i]))
            DL[i] = np.sum(1.0 * (self.D - RC.dot(theta)) * -RC[:,i] * sigma_inv_diag**2)
        return(DL)

    def Create_Sigma_Inv(self):
        sigma = np.eye(len(self.D))
        for i in np.arange(len(self.D)):
            sigma[i,i] = self.D_un[i]
        sigma_inv = np.linalg.inv(sigma)
        sigma_det = np.linalg.det(sigma)
        return([sigma, sigma_inv, sigma_det])

    def Create_DDS(self, theta, method="classic"):
        #print("Create_DDS", time.clock())
        DDS = np.zeros((len(self.m), len(self.m)))
        if method == "classic" or method == "classic_auto_scaling":
            for i in np.arange(len(self.m)):
                #DDS[i,i] = -1/np.log(10) / theta[i]
                DDS[i,i] = -1 / theta[i]
        elif method == "bound_0_1":
            for i in np.arange(len(self.m)):
                #DDS[i,i] = -1/np.log(10) * (1 / theta[i] + 1 / (1.0 - theta[i]))
                DDS[i,i] = -1 * (1 / theta[i] + 1 / (1.0 - theta[i]))
        return(DDS)

    def Create_DDL(self, sigma_inv, C=1.0):
        #print("Create_DDL", time.clock())
        DDL = np.zeros((len(self.m), len(self.m)))
        sigma_inv_diag = np.diag(sigma_inv)
        RC = self.R.dot(C)
        i = 0
        while i < len(self.m):
            j = 0
            while j <= i:
                #DDL[i,j] = DDL[j,i] = 2.0 * self.R[:,i].dot(sigma_inv**2).dot(self.R[:,j])
                DDL[i,j] = DDL[j,i] = np.sum(2.0 * RC[:,i] * RC[:,j] * sigma_inv_diag**2) # reduce from 1xN times NxN times Nx1 to N times N times N
                j = j + 1
            i = i + 1
        return(DDL)  

    def Entropy_func(self, theta, method="classic"):
        #print("Entropy_func", time.clock())
        if method == "classic" or method == "classic_auto_scaling":
            #return(np.sum(theta - self.m - theta * np.log10(theta / self.m)))
            return(np.sum(theta - self.m - theta * np.log(theta / self.m)))
        elif method == "bound_0_1":
            #return(np.sum(-theta * np.log10(theta / self.m) - (1.0-theta) * np.log10((1.0-theta) / (1.0-self.m))))
            return(np.sum(-theta * np.log(theta / self.m) - (1.0-theta) * np.log((1.0-theta) / (1.0-self.m))))

    def calOmega(self, alpha_init, inflation_factor = 10.0, termination_threshold = 0.01, verbal_flag=True, time_flag=False):
        h_init = self.m
        H = 1.0
        G = 0.0
        count_num = 0
        count_max = 50
        while H > termination_threshold * G and count_num < count_max:
            if time_flag:
                print(count_num, time.clock())
            DS = self.Create_DS(h_init)
            sigma, sigma_inv, sigma_det = self.Create_Sigma_Inv()
            DL = self.Create_DL(h_init, sigma_inv)
            DQ = alpha_init * DS - DL # DQ is g on Page 28
            DDS = self.Create_DDS(h_init)
            DDL = self.Create_DDL(sigma_inv)
            mu = np.linalg.inv(-DDS)
            if time_flag:
                print("A1", time.clock())
            A = np.sqrt(mu).dot(DDL).dot(np.sqrt(mu))
            if time_flag:
                print("A2", time.clock())
            B = np.eye(len(self.m)) + A / alpha_init
            if time_flag:
                print("A3", time.clock())
            beta = alpha_init * inflation_factor
            dh = np.sqrt(mu).dot(np.linalg.inv(beta * np.eye(len(self.m)) + A)).dot(np.sqrt(mu)).dot(DQ)
            if time_flag:
                print("A4", time.clock())
            h_init = h_init + dh
            idx = np.where(h_init <= 0.0)
            h_init[idx] = 1e-9
            H = 0.5 / alpha_init * DQ.dot(np.sqrt(mu)).dot(np.linalg.inv(B)).dot(np.sqrt(mu)).dot(DQ)
            if time_flag:
                print("A5", time.clock())
            G = np.matrix.trace(np.linalg.inv(alpha_init * B).dot(A))
            if time_flag:
                print("A6", time.clock())
            Omega_init = G / (-2 * alpha_init * self.Entropy_func(h_init)) # see Page 27
            if time_flag:
                print("A7", time.clock())
            count_num = count_num + 1
        if verbal_flag:
            print(alpha_init, H, G / 2.0, Omega_init, np.shape(h_init[idx]), count_num, time.clock())   
        return([Omega_init, h_init])

    def createConvolutionMatrix(self, correlation_length=1, plot_flag=False):
        padding_length = len(self.m)
        a = np.arange(padding_length)
        a = a - np.median(a)
        a = np.exp(-(a)**2/(2.0*(correlation_length)**2))
        a = a / np.sum(a)
        if plot_flag:
            plt.plot(a)
            plt.show()
        b = scipy.linalg.toeplitz(a)
        b = np.roll(b, int(-padding_length/2), axis=0)
        return(b)

    def calOmega_general(self, alpha_init, inflation_factor = 10.0, termination_threshold = 0.01, method="classic", correlation_length=1.0,  verbal_flag=True, time_flag=False):
        h_init = self.m
        H = 1.0
        G = 0.0
        count_num = 0
        count_max = 50
        while H > termination_threshold * G and count_num < count_max:
            if time_flag:
                print(count_num, time.clock())
            DS = self.Create_DS(h_init, method=method)
            sigma, sigma_inv, sigma_det = self.Create_Sigma_Inv()
            C = self.createConvolutionMatrix(correlation_length=correlation_length)
            DL = self.Create_DL(h_init, sigma_inv, C=C)
            DQ = alpha_init * DS - DL # DQ is g on Page 28
            DDS = self.Create_DDS(h_init, method=method)
            DDL = self.Create_DDL(sigma_inv, C=C)
            mu = np.linalg.inv(-DDS)
            if time_flag:
                print("A1", time.clock())
            A = np.sqrt(mu).dot(DDL).dot(np.sqrt(mu))
            if time_flag:
                print("A2", time.clock())
            B = np.eye(len(self.m)) + A / alpha_init
            if time_flag:
                print("A3", time.clock())
            beta = alpha_init * inflation_factor
            dh = np.sqrt(mu).dot(np.linalg.inv(beta * np.eye(len(self.m)) + A)).dot(np.sqrt(mu)).dot(DQ)
            if time_flag:
                print("A4", time.clock())
            h_init = h_init + dh
            if method == "bound_0_1":
                idx0 = np.where((h_init <= 0.0) & (h_init >= 1.0))
                idx = np.where(h_init <= 0.0)
                h_init[idx] = 1e-9
                idx = np.where(h_init >= 1.0)
                h_init[idx] = 1.0 - 1e-9
                idx = idx0
            elif method == "classic" or method == "classic_auto_scaling":
                idx = np.where(h_init <= 0.0)
                h_init[idx] = 1e-9
            H = 0.5 / alpha_init * DQ.dot(np.sqrt(mu)).dot(np.linalg.inv(B)).dot(np.sqrt(mu)).dot(DQ)
            if time_flag:
                print("A5", time.clock())
            G = np.matrix.trace(np.linalg.inv(alpha_init * B).dot(A))
            if time_flag:
                print("A6", time.clock())
            Omega_init = G / (-2 * alpha_init * self.Entropy_func(h_init, method=method)) # see Page 27
            if time_flag:
                print("A7", time.clock())
            count_num = count_num + 1
        if verbal_flag:
            print(alpha_init, H, G / 2.0, Omega_init, np.shape(h_init[idx]), count_num, time.clock())   
        return([Omega_init, h_init])

    def calOmegaVsAlpha(self, alpha_arr, plot_flag=True,  **kwarg):
        Omega_arr = np.zeros(np.shape(alpha_arr))
        for idx, alpha in enumerate(alpha_arr):
            #Omega_init, h_init = self.calOmega(alpha, **kwarg)
            Omega_init, h_init = self.calOmega_general(alpha, **kwarg)
            Omega_arr[idx] = Omega_init
        if plot_flag:
            plt.plot(alpha_arr, Omega_arr, "bo")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\Omega$")
            plt.show()
        return([alpha_arr, Omega_arr])

    def calOptimalAlpha(self, alpha_arr, Omega_arr, termination_threshold_Omega = 0.05, **kwarg):
        idx = np.argsort(Omega_arr)
        Omega_init_arr = Omega_arr[idx]
        alpha_init_arr = alpha_arr[idx]
        f_interp = scipy.interpolate.interp1d(Omega_init_arr, alpha_init_arr)
        alpha_init = f_interp(1.0)
        Omega_init = 0.0
        while np.abs(1.0 - Omega_init) > termination_threshold_Omega:
            Omega_init, h_init = self.calOmega_general(alpha_init, **kwarg)
            #include new alpha and Omega value:
            alpha_init_arr = np.append(alpha_init_arr, alpha_init)
            Omega_init_arr = np.append(Omega_init_arr, Omega_init)
            idx = np.argsort(alpha_init_arr)[::-1] # Omega(alpha) is not monotonic
            alpha_init_arr = alpha_init_arr[idx]
            Omega_init_arr = Omega_init_arr[idx]
            tmp, idx = np.unique(alpha_init_arr, return_index=True)
            alpha_init_arr = alpha_init_arr[idx]
            Omega_init_arr = Omega_init_arr[idx]    
            Omega_init_arr = Omega_init_arr[np.isfinite(Omega_init_arr)]
            alpha_init_arr = alpha_init_arr[np.isfinite(Omega_init_arr)]
            # Update alpha_init value
            f_interp = scipy.interpolate.interp1d(Omega_init_arr, alpha_init_arr)
            alpha_init = f_interp(1.0)
        return(alpha_init)                

class Pynirspec_spec1D():
    def __init__(self, file_name=None, dither_pattern="ABBA", wav_file=None, output_spec_name=None, flag_append=True):
        self.file_name=file_name
        self.flag_append = flag_append
        self.spec = pyfits.open(self.file_name)
        self.header = self.spec[0].header
        self.mjd = self.header["MJD-OBS"]
        self.airmass = self.header["AIRMASS"]
        self.order = int(self.header["ORDER"])
        if dither_pattern == "ABBA":
            self.elptime = self.header["ELAPTIME"] * 4.0
        elif dither_pattern == "AB":
            self.elptime = self.header["ELAPTIME"] * 2.0
        else:
            self.elptime = self.header["ELAPTIME"] * 1.0
        self.wav_file = wav_file
        self.num_pixels = len(self.spec[1].data.field(0))
        self.wav_solution = self.getWavelengthSolution()
        self.spec_all = self.getSpec()
        self.output_spec_name = output_spec_name
        self.spec_all.writeSpec(file_name=self.output_spec_name,flag_python2=False,flag_append=self.flag_append)

    def DSFunc(self, pixels, Coefs):
        if np.size(Coefs) >= 2:
            WaveOut = np.zeros(len(pixels))
            for i in range(0,len(Coefs)):
                    WaveOut = WaveOut + Coefs[i]*(pixels/1000.0)**i
        else:
            WaveOut = np.nan
        return(WaveOut)

    def getWavelengthSolution(self):
        wav_sol = ascii.read(self.wav_file)
        if self.order in wav_sol.field(0):
            idx = np.where(wav_sol.field(0) == self.order)
            Coefs = np.array([])
            num_columns = len(wav_sol.columns)
            for i in np.arange(num_columns - 1) + 1:
                Coefs = np.append(Coefs, wav_sol[idx].field(i))
        else:
            Coefs = np.nan
        WaveOut = self.DSFunc(np.arange(self.num_pixels), Coefs)
        return(WaveOut)

    def getSpec(self):
        spec_pos_st = Spectrum(self.wav_solution, self.spec[1].data.field(1))
        spec_pos_st.removeNaN()
        noise_tmp = self.spec[1].data.field(2)
        noise_tmp[np.where(noise_tmp == 0.0)] = np.max(noise_tmp)
        spec_pos_st.addNoise(noise_tmp)
        #spec_pos_sk = self.spec[1].data.field(3)
        spec_neg_st = Spectrum(self.wav_solution, self.spec[1].data.field(8))
        spec_neg_st.removeNaN()
        noise_tmp = self.spec[1].data.field(9)
        noise_tmp[np.where(noise_tmp == 0.0)] = np.max(noise_tmp)
        spec_neg_st.addNoise(noise_tmp)
        #spec_pos_sk = self.spec[1].data.field(10)
        ccf = spec_neg_st.crossCorrelation(spec_pos_st)
        vel_shift = ccf.calcCentroid()
        spec_neg_st.dopplerShift(rv_shift=vel_shift)
        spec_neg_st.resampleSpec(spec_pos_st.wavelength)
        # combine spectra
        pos_err = spec_pos_st.noise
        neg_err = spec_neg_st.noise
        pos_flx = spec_pos_st.flux
        neg_flx = spec_neg_st.flux
        all_flx = (pos_flx * pos_err**(-2) + neg_flx * neg_err**(-2)) / (pos_err**(-2) + neg_err**(-2))
        all_err = (pos_err**(-2) + neg_err**(-2))**(-0.5)
        spec_all_st = Spectrum(self.wav_solution, all_flx)
        spec_all_st.addNoise(all_err)
        #plt.plot(spec_all_st.wavelength, spec_all_st.flux)
        # remove outliers
        flx_ratio = spec_pos_st.flux / spec_neg_st.flux
        flx_ratio_std = np.std(np.sort(flx_ratio)[int(0.05*self.num_pixels):int(0.95*self.num_pixels)])
        flx_ratio_med = np.median(flx_ratio)
        #plt.plot(spec_neg_st.wavelength, flx_ratio)
        #print(flx_ratio_med, flx_ratio_std)
        dif = np.abs(flx_ratio - flx_ratio_med)
        sigma_clip = 10.0
        idx_in = np.where(dif < sigma_clip * flx_ratio_std)
        idx_out = np.where(dif >= sigma_clip * flx_ratio_std)
        spec_all_st.flux[idx_out] = np.interp(spec_all_st.wavelength[idx_out], spec_all_st.wavelength[idx_in], spec_all_st.flux[idx_in])
        #plt.plot(spec_all_st.wavelength, spec_all_st.flux)
        #plt.plot(spec_pos_st.wavelength, spec_pos_st.flux)
        #plt.plot(spec_neg_st.wavelength, spec_neg_st.flux)
        #plt.show()
        spec_norm = spec_all_st.getContinuum(poly_order=3)
        #plt.plot(spec_all_st.wavelength, spec_all_st.flux)
        #plt.plot(spec_all_st.wavelength, spec_norm)
        #plt.show()
        spec_all_st.flux = spec_all_st.flux / spec_norm
        spec_all_st.noise = spec_all_st.noise / spec_norm
        #plt.errorbar(spec_all_st.wavelength, spec_all_st.flux, yerr=spec_all_st.noise)
        #plt.show()
        return(spec_all_st)

class RossiterMcLaughlinEffect():
    def __init__(self, self_luminous=False, b=0.0, t0=4.0, vrot=10e3, oblqt=45.0, R=1.0, r=0.5, f12=0.8, vrot2=20e3, P=1.0, Mtotal=1.0, e=0.3, om=60, tp=0.2, mass_ratio=0.01, u1=1.0, u2=0.0, lambda0=2.0, lsf=None):
        self.b = b # impact parameter
        if self.b == 0.0:
            self.b = 1e-6
        self.t0 = t0 # transit duration in hours
        self.vrot = vrot # primary rotational velocity in m/s
        self.oblqt = oblqt / 180.0 * np.pi # the angle (in deg) between orbital angular momentum axis and spin axis of the primary
        self.oblqt = self.oblqt % (2.0 * np.pi)
        self.R = R # primary radius in unity
        self.r = r # secondary radius in fraction of the primary
        self.self_luminous = self_luminous # whether the secondary is self luminous, if True, then SB, if False, then planet. 
        self.f12 = f12 # flux ratio between primary and secondary out of eclipse,i.e. < 1
        self.vrot2 = vrot2 # rotational velocity of the secondary
        self.msini = Mtotal / (1.0 + (1.0 / mass_ratio)) / 0.0009543 # companion mass in Jupiter mass, 0.0009543 is jupiter mass in solar mass
        self.orbitalPeriod = P # orbital period in days
        self.mtotal = Mtotal # total mass in solar mass
        self.eccentricity = e # eccentricity
        self.argumentOfPeriastron = om # argument of periastron in degree
        self.timeAtPeriastron = tp # time at periastron in days 
        self.massRatio = mass_ratio
        self.u1 = u1 # limb darkening 1
        self.u2 = u2 # limb darkening 2
        self.lambda0 = lambda0 # central wavelength in um, make sure lambda0 corresponds to the center of lsf
        self.lsf = lsf # line spread function of the instrument, should be Specrum class
    
    def intersection_area(self, t):
        # from: https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
        """Return the area of intersection of two circles.

        The circles have radii R and r, and their centres are separated by d.

        """

        R = self.R
        r = self.r
        d, BE = self.calc_d_vs_time(t)
        if d <= abs(R-r):
            # One circle is entirely enclosed in the other.
            return np.pi * min(R, r)**2
        if d >= r + R:
            # The circles don't overlap at all.
            return 0

        r2, R2, d2 = r**2, R**2, d**2
        alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
        beta = np.arccos((d2 + R2 - r2) / (2*d*R))
        return ( r2 * alpha + R2 * beta -
                 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta))
               ) 

    def makeIllustrationFigure(self, t, flag_sav=False, flag_show=True):
        d, BE = self.calc_d_vs_time(t)
        plt.figure(figsize=(6,6))
        plt.plot([-2, 2], [self.b * self.R, self.b * self.R], color="k", ls="--", lw=3)
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)

        c1 = plt.Circle((0.0, 0.0), self.R, color='orange', alpha=0.6)
        c2 = plt.Circle((BE, self.b * self.R), self.r, color='red', alpha=0.6)
        c3 = ax.arrow(0, 0, 1.5*self.R*np.cos(self.oblqt+np.pi/2.0), 1.5*self.R*np.sin(self.oblqt+np.pi/2.0), head_width=0.15, head_length=0.15, fc='k', ec='k')
        ax.add_artist(c1)
        ax.add_artist(c2)
        ax.add_artist(c3)
        plt.title("r={0:04.2f}, b={1:05.2f}, i={2:05.1f}, t0={3:05.2f}, t={4:05.2f}".format(self.r, self.b, self.oblqt/np.pi*180.0, self.t0, t))
        if flag_sav:
            plt.savefig("illus_"+"{0:05.2f}".format(t)+".png")
        if flag_show:
            plt.show()
        plt.clf()

    def makeIllustrationVideo(self, t_arr=None, t_int=0.4):
        if t_arr is None:
            t_arr = np.arange(0.1,self.t0,t_int)
        for t in t_arr:
            self.makeIllustrationFigure(t, flag_sav=True, flag_show=False)
        import imageio, glob
        images = []
        filenames = sorted(glob.glob("./illus_*png"))
        for filename in filenames:
            images.append(imageio.imread(filename))
            imageio.mimsave('./illus.gif', images)

    def calc_d_vs_time(self, t):
        R = self.R
        r = self.r
        AC = 2.0 * np.sqrt((r+R)**2 - (self.b*R)**2)
        v0 = AC / self.t0
        if (t<0.0) | (t>self.t0):
            return(1e9)
        else:
            AE = v0 * t
            cosACD = AC / 2.0 / (R+r)
            CE = AC - AE
            d = np.sqrt((CE)**2 + (r+R)**2 - 2.0*(CE)*(r+R)*cosACD)
            BE = AE - AC / 2.0 
            return([d, BE])

    def findMinMaxVelocity_disContinuityRemoval_Oblqt_0_2pi_b_minus_1_plus_1(self, t, verbal=False):
        if (self.b > (self.R + self.r)) | (self.b < (-1.0 * (self.R + self.r))):
            return([np.inf, np.inf])
        b_0 = self.b
        if self.b >= 0.0:
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval_Oblqt_0_2pi(t, verbal=verbal)
        else:
            self.b = -1.0 * self.b
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval_Oblqt_0_2pi(self.t0 - t, verbal=verbal)
            v_min = -1.0 * v_min
            v_max = -1.0 * v_max
        self.b = b_0
        if v_max >= v_min:
            return([v_min, v_max])
        else:
            return([v_max, v_min])

    def findMinMaxVelocity_disContinuityRemoval_Oblqt_0_2pi(self, t, verbal=False):
        """
        oblqt is between 0 and 90, for other values, use the mirror effect. E.g., 180 is the same as 0 except the v_min, v_max signs are reversed. 
        E.g., -45 is the same as 45 except that the v_min, v_max signs are reversed and the time sequence is reversed. 
        """
        oblqt_0 = self.oblqt
        if (self.oblqt < (np.pi / 2.0)) & (self.oblqt >= 0.0):
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval(t, verbal=verbal)
        if (self.oblqt < np.pi) & (self.oblqt >= (np.pi / 2.0)):
            self.oblqt = np.pi - self.oblqt
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval(self.t0 - t, verbal=verbal)
        if (self.oblqt < (3.0 * np.pi / 2.0)) & (self.oblqt >= np.pi):
            self.oblqt = self.oblqt - np.pi
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval(t, verbal=verbal)
            v_min = -1.0 * v_min
            v_max = -1.0 * v_max
        if (self.oblqt < (2.0 * np.pi)) & (self.oblqt >= (3.0 * np.pi / 2.0)):
            self.oblqt = 2.0 * np.pi - self.oblqt
            v_min, v_max = self.findMinMaxVelocity_disContinuityRemoval(self.t0 - t, verbal=verbal)
            v_min = -1.0 * v_min
            v_max = -1.0 * v_max
        self.oblqt = oblqt_0
        if v_max >= v_min:
            return([v_min, v_max])
        else:
            return([v_max, v_min])
 
    def findMinMaxVelocity_disContinuityRemoval(self, t, verbal=False):
        t_int = 1e-3
        d, BE = self.calc_d_vs_time(t)
        if d >= self.r + self.R:
            # The circles don't overlap at all.
            return([np.inf, np.inf])
        else:
            v_min, v_max = self.findMinMaxVelocity(t, verbal=verbal, flag_reverse=True)
            return([v_min, v_max])

    def findMinMaxVelocity(self, t, verbal=True, flag_reverse=True):
        d, BE = self.calc_d_vs_time(t)
        if d >= self.r + self.R:
            # The circles don't overlap at all.
            return([np.inf, np.inf])
        elif d <= np.abs(self.R-self.r):
            # One circle is entirely enclosed in the other.
            if d == 0.0:
                HDC = 0.0
            else:
                HDC = np.arccos(np.median(np.array([-1.0, 1.0, self.b * self.R / d])))
            if BE < 0:
                HDC = -1.0 * HDC
                ECD = np.pi / 2.0 + HDC + self.oblqt # ECD is between 0 and pi
                DE = np.sqrt(d**2 + self.r**2 - 2.0*d*self.r*(np.cos(ECD)))
                if d == 0.0:
                    EDC = 0.0
                else:
                    EDC = np.arccos((DE**2 + d**2 - self.r**2) / (2.0*DE*d)) # EDC is between 0 and pi
                GDC = self.oblqt + HDC # GDC is between -pi/2 and pi//2
                GDE = GDC + EDC 
                IE = DE * np.sin(GDE)
                v_E = self.vrot * IE / self.R
                v_F = self.vrot * (IE - 2.0*self.r) / self.R
            else:
                ECD = np.pi / 2.0 + HDC + self.oblqt # ECD is between pi / 2 an d 3*pi/2
                DE = np.sqrt(d**2 + self.r**2 - 2.0*d*self.r*(np.cos(ECD)))
                if d == 0.0:
                    EDC = 0.0
                else:
                    if (np.pi / 2.0 - self.oblqt) > HDC:
                        EDC = np.arccos((DE**2 + d**2 - self.r**2) / (2.0*DE*d)) # EDC is between 0 and pi
                    else:
                        EDC = -1.0 * np.arccos((DE**2 + d**2 - self.r**2) / (2.0*DE*d)) # EDC is between 0 and pi
                GDC = self.oblqt + HDC # GDC is between -pi/2 and pi//2
                GDE = GDC + EDC
                IE = DE * np.sin(GDE)
                v_E = self.vrot * IE / self.R
                v_F = self.vrot * (IE - 2.0*self.r) / self.R
            if verbal:
                print("EDC = ", EDC)
                print("GDC = ", GDC)
                print("GDE = ", GDE)
                print("self.oblqt, d, HDC, ECD, DE, IE", self.oblqt, d, HDC, ECD, DE, IE)
            return(np.sort(np.array([v_E, v_F])))
        else:
            # Two circles are partially overlapped
            if d == 0.0:
                HDC = 0.0
            else:
                HDC = np.arccos(np.median(np.array([-1.0, 1.0, self.b * self.R / d])))
                if BE < 0:
                    HDC = -1.0 * HDC
            CDA = np.arccos((d**2 + self.R**2 - self.r**2) / (2.0 * d * self.R)) # CDA is between 0 and pi
            if BE < 0:
                CDA = -1.0 * CDA
            HDA = HDC + CDA
            HDB = HDC - CDA
            if BE < 0:
                HDE = -np.pi / 2.0 - self.oblqt # if HDE is between HDA and HDB, then v_max or v_min is vrot
                v_E = -1.0 * self.vrot
            else:
                HDE = np.pi / 2.0 - self.oblqt # if HDE is between HDA and HDB, then v_max or v_min is vrot
                v_E = self.vrot
            v_A = self.vrot * np.sin(self.oblqt + HDC + CDA) 
            v_B = self.vrot * np.sin(self.oblqt + HDC - CDA)
            if BE < 0:
                if (-HDC) > self.oblqt:
                    # FD1
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (0, pi) extrema when oblqt and HDC differ by pi/2
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    FDC = np.arccos((FD**2 + d**2 - self.r**2) / (2.0 * FD * d)) # FDC is always positive
                    GDF = -HDC - self.oblqt + FDC
                    IF = FD * (np.sin(GDF)) 
                    v_F = -self.vrot * IF / self.R
                    HDF = GDF - self.oblqt
                    FD1 = FD
                    v_F1 = v_F  
                    if verbal:
                        print("left, 1, FCD, FD", FCD, FD)
                        print("GDF, HDC, FDC ", GDF, HDC, FDC) 
                    # FD2
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (0, pi) extrema when oblqt and HDC differ by pi/2
                    FCD = np.pi - FCD
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    v_F2 = v_F1 + 2.0 * self.r / self.R * self.vrot
                    FD2 = FD 
                else:
                    # FD1
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (0, pi) extrema when oblqt and HDC differ by pi/2
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    FDC = np.arccos((FD**2 + d**2 - self.r**2) / (2.0 * FD * d)) # FDC is always positive
                    GDF = -HDC - self.oblqt + FDC
                    IF = FD * (np.sin(GDF))
                    HDF = GDF + self.oblqt
                    #if (self.b * self.R - self.r) > (self.R * np.sin(np.pi / 2.0 - self.oblqt)):
                    if GDF < 0:
                        v_F = self.vrot * np.abs(IF) / self.R
                    else:
                        v_F = -self.vrot * np.abs(IF) / self.R 
                    FD1 = FD
                    v_F1 = v_F
                    if verbal:
                        print((self.b * self.R - self.r), (self.R * np.sin(np.pi / 2.0 - self.oblqt)))
                        print("left, 2, FCD, FD", FCD, FD)
                        print("GDF, HDC, FDC ", GDF, HDC, FDC)
                    # FD2
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (0, pi) extrema when oblqt and HDC differ by pi/2
                    FCD = np.pi - FCD
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    v_F2 = v_F1 + 2.0 * self.r / self.R * self.vrot
                    FD2 = FD                                                      
            else:
                if (np.pi / 2.0 - HDC) < self.oblqt:
                    # FD1
                    FCD = HDC + self.oblqt - np.pi / 2.0 # FCD (-pi/2, pi/2) extrema when oblqt and HDC differ by 0
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    FDC = np.arccos((FD**2 + d**2 - self.r**2) / (2.0 * FD * d)) # FDC is always positive
                    GDF = HDC + self.oblqt + FDC
                    IF = FD * (np.sin(GDF))
                    HDF = GDF - self.oblqt
                    #if (self.b * self.R - self.r) > (self.R * np.sin(np.pi / 2.0 - self.oblqt)):
                    if GDF < np.pi:
                        v_F = self.vrot * np.abs(IF) / self.R
                    else:
                        v_F = -self.vrot * np.abs(IF) / self.R
                    FD1 = FD
                    v_F1 = v_F
                    if verbal:
                        print("right, 1, FCD, FD", FCD, FD)
                        print("GDF, HDC, FDC ", GDF, HDC, FDC)
                    # FD2
                    FCD = HDC + self.oblqt - np.pi / 2.0 # FCD (-pi/2, pi/2) extrema when oblqt and HDC differ by 0
                    FCD = np.pi - FCD
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    v_F2 = v_F1 + 2.0 * self.r / self.R * self.vrot
                    FD2 = FD
                else:
                    # FD1
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (-pi/2, pi/2) extrema when oblqt and HDC differ by 0
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    FDC = np.arccos((FD**2 + d**2 - self.r**2) / (2.0 * FD * d)) # FDC is always positive
                    GDF = -HDC - self.oblqt + FDC
                    IF = FD * (np.sin(GDF))
                    HDF = GDF + self.oblqt
                    v_F = -self.vrot * IF / self.R
                    FD1 = FD
                    v_F1 = v_F
                    if verbal:
                        print("right, 2, FCD, FD", FCD, FD)
                        print("GDF, HDC, FDC ", GDF, HDC, FDC)
                    # FD2
                    FCD = np.pi / 2.0 - HDC - self.oblqt # FCD (0, pi) extrema when oblqt and HDC differ by 0
                    FCD = np.pi - FCD
                    FD = np.sqrt(d**2 + self.r**2 - 2.0 * d * self.r * np.cos(FCD)) # FD can be more than 2 * R 
                    v_F2 = v_F1 + 2.0 * self.r / self.R * self.vrot
                    FD2 = FD
            if np.abs(FD1) > self.R: 
                v_F1 = v_A
            if np.abs(FD2) > self.R:
                v_F2 = v_A
            if verbal:
                print("v_ABCD = ", v_A, v_B, v_E, v_F1, v_F2)
                print("HD_ABCD =", HDA, HDB, HDE, HDF)
                print("FCD, FDC, FD, IF, d = ", FCD, FDC, FD, IF, d)
                print("self.oblqt, HDC, GDF, HDC =", self.oblqt, HDC, GDF, HDC)
                print(self.vrot, IF, v_F, self.vrot * IF / self.R, self.R)
                print(self.intersection_area(t), np.pi * self.r**2)
            if ((HDE <= np.max([HDA, HDB])) & (HDE >= np.min([HDA, HDB]))): 
                # HDE needs to be between HDA and HDB, but HDF does not have to, so v_F will always be considered
                if (HDF <= np.max([HDA, HDB])) & (HDF >= np.min([HDA, HDB])):
                    v_min, v_max = np.min([v_A, v_B, v_E, v_F1, v_F2]), np.max([v_A, v_B, v_E, v_F1, v_F2])
                    if verbal: print("4")
                else:
                    v_min, v_max = np.min([v_A, v_B, v_E, v_F1, v_F2]), np.max([v_A, v_B, v_E, v_F1, v_F2])
                    if verbal: print("3")
            elif (((HDE + np.pi) <= np.max([HDA, HDB])) & ((HDE + np.pi) >= np.min([HDA, HDB]))) | (((HDE - np.pi) <= np.max([HDA, HDB])) & ((HDE - np.pi) >= np.min([HDA, HDB]))):
                if (HDF <= np.max([HDA, HDB])) & (HDF >= np.min([HDA, HDB])):
                    v_min, v_max = np.min([v_A, v_B, -v_E, v_F1, v_F2]), np.max([v_A, v_B, -v_E, v_F1, v_F2])
                    if verbal: print("4")
                else:
                    v_min, v_max = np.min([v_A, v_B, -v_E, v_F1, v_F2]), np.max([v_A, v_B, -v_E, v_F1, v_F2])
                    if verbal: print("3")
            else:
                if (HDF <= np.max([HDA, HDB])) & (HDF >= np.min([HDA, HDB])):
                    v_min, v_max = np.min([v_A, v_B, v_F1, v_F2]), np.max([v_A, v_B, v_F1, v_F2])
                    if verbal: print("2")
                else:
                    v_min, v_max = np.min([v_A, v_B, v_F1, v_F2]), np.max([v_A, v_B, v_F1, v_F2])              
                    if verbal: print("1") 
            return([v_min, v_max]) 
        
    def calcRMSeries(self, t_arr=None, flag_video=False, flag_plot=True):
        if t_arr is None:
            t_int = 0.2
            t_arr = np.arange(0.1,self.t0,t_int)
        v_arr = np.zeros((len(t_arr), 2))
        for i, t in enumerate(t_arr):
            v_arr[i,:] = self.findMinMaxVelocity_disContinuityRemoval_Oblqt_0_2pi_b_minus_1_plus_1(t, verbal=False)
        #print(v_arr)
        if flag_plot:
            plt.clf()
            plt.plot(t_arr, v_arr[:,0], "bo-")
            plt.plot(t_arr, v_arr[:,1], "rx-")
            plt.title("r={0:04.2f}, b={1:05.2f}, i={2:05.1f}, t0={3:05.2f}".format(self.r, self.b, self.oblqt/np.pi*180.0, self.t0)) 
            plt.savefig("./v_min_max.png")
            plt.show()
            plt.clf()
        if flag_video:
            self.makeIllustrationVideo(t_arr=t_arr)
        return([t_arr, v_arr])

    def createLineProfile(self, vsini=None, u1=1.0, u2=0.0, lambda_0=2.0, flag_plot=False):
        lambda_0 = self.lambda0
        spec_lsf = self.lsf
        # lsf is instrument profile or line spread function
        if vsini is None:
            vsini = self.vrot
        spec_lsf.flux = spec_lsf.flux - np.median(spec_lsf.flux)
        spec_lsf.flux = spec_lsf.flux / np.sum(spec_lsf.flux)
        kernel = spec_lsf.calcRotationalBroadeningKernel(u1=u1, u2=u2, vsini=vsini, lambda_0=lambda_0)
        kernel.resampleSpec(spec_lsf.wavelength[np.where((spec_lsf.wavelength>np.min(kernel.wavelength)) & (spec_lsf.wavelength<np.max(kernel.wavelength)))])
        lp = spec_lsf.convolveKernel(kernel.flux)
        lp.flux = lp.flux / np.sum(lp.flux)
        if flag_plot:
            plt.plot(spec_lsf.wavelength, spec_lsf.flux, "b")
            plt.plot(lp.wavelength, lp.flux, "r")
            plt.show()
        return(lp)

    def createRMLineProfile(self, t, v_min, v_max, u1=1.0, u2=0.0, lambda_0=2.0, flag_plot=False):
        spec_lsf = self.lsf
        lambda_0=self.lambda0
        if not self.self_luminous:
            f = self.calcBlockedFlux(t, v_min, v_max, u1=u1, u2=u2)
            #rv_t1, dvdt = self.calcRVChangeRateDuringEclipse(Msini=self.msini, P=self.orbitalPeriod, Mtotal=self.mtotal, e=self.eccentricity, om=self.argumentOfPeriastron, tp=self.timeAtPeriastron, tint=1e-3, flag_plot=False) # zero crossing is not necessarily where ecplise takes place, so this is wrong, should just use t - self.t0 / 2.0 to calculate rv
            lp_s = self.createLineProfile(vsini=None, u1=u1, u2=u2, lambda_0=lambda_0, flag_plot=flag_plot)
            if 1 == 0: # incorrect, discard
                dv_s = rv_t1 + dvdt * (t - self.t0 / 2.0)
                dv_p = -dv_s / self.massRatio
            else:
                dv_s = self.calcRVatAtime(Msini=self.msini, P=self.orbitalPeriod, Mtotal=self.mtotal, e=self.eccentricity, om=self.argumentOfPeriastron, tp=self.timeAtPeriastron, _t=np.array([t-self.t0/2.0])/24.0)
                dv_p = -dv_s / self.massRatio 
            lp_s.dopplerShift(rv_shift=-dv_s) 
            lp_p = self.createLineProfile(vsini=(v_max-v_min)/2.0-dv_s, u1=u1, u2=u2, lambda_0=lambda_0, flag_plot=flag_plot)
            lp_p.dopplerShift(rv_shift=-(v_max+v_min)/2.0).resampleSpec(lp_s.wavelength)
            lp_rm = lp_s.copy()
            lp_rm.flux = lp_s.flux - f * lp_p.flux
            lp_rm.flux = lp_rm.flux / np.sum(lp_rm.flux)
            if flag_plot:
                plt.plot(lp_s.converWavelengthtoVelocity(lp_s.wavelength), lp_s.flux, "b-", label="1")
                plt.plot(lp_p.converWavelengthtoVelocity(lp_p.wavelength), lp_p.flux * f, "r-", label="2")
                plt.plot(lp_rm.converWavelengthtoVelocity(lp_rm.wavelength), lp_rm.flux, "k-", label="f")
                plt.legend()
                plt.show()
        else:
            f = self.calcBlockedFlux(t, v_min, v_max, u1=u1, u2=u2)
            #rv_t1, dvdt = self.calcRVChangeRateDuringEclipse(Msini=self.msini, P=self.orbitalPeriod, Mtotal=self.mtotal, e=self.eccentricity, om=self.argumentOfPeriastron, tp=self.timeAtPeriastron, tint=1e-3, flag_plot=True) # zero crossing is not necessarily where ecplise takes place, so this is wrong, should just use t - self.t0 / 2.0 to calculate rv
            lp_s = self.createLineProfile(vsini=None, u1=u1, u2=u2, lambda_0=lambda_0, flag_plot=flag_plot)
            if 1 == 0: # incorrect, discard
                dv_s = rv_t1 + dvdt * (t - self.t0 / 2.0)
                dv_p = -dv_s / self.massRatio
            else:
                dv_s = self.calcRVatAtime(Msini=self.msini, P=self.orbitalPeriod, Mtotal=self.mtotal, e=self.eccentricity, om=self.argumentOfPeriastron, tp=self.timeAtPeriastron, _t=np.array([t-self.t0/2.0])/24.0)
                dv_p = -dv_s / self.massRatio
            #print("delta v = ", dv_p - dv_s)
            lp_s.dopplerShift(rv_shift=-dv_s) 
            lp_p = self.createLineProfile(vsini=(v_max-v_min)/2.0, u1=u1, u2=u2, lambda_0=lambda_0, flag_plot=flag_plot)
            lp_p.dopplerShift(rv_shift=-(v_max+v_min)/2.0-dv_s).resampleSpec(lp_s.wavelength)
            lp_rm = lp_s.copy()
            # primary star line profile when being eclipsed
            lp_rm.flux = lp_s.flux - f * lp_p.flux
            lp_1_rm = lp_rm.copy()
            # secondary star line profile, assuming the same limb darkening coefficient
            lp_2 = self.createLineProfile(vsini=self.vrot2, u1=u1, u2=u2, lambda_0=lambda_0, flag_plot=flag_plot)
            lp_2.dopplerShift(rv_shift=-dv_p).resampleSpec(lp_s.wavelength)
            # primary + secondary
            lp_rm.flux = lp_rm.flux + self.f12 * lp_2.flux
            # normalization
            lp_rm.flux = lp_rm.flux / np.sum(lp_rm.flux)
            #if flag_plot:
            if True:
                plt.plot(lp_s.converWavelengthtoVelocity(lp_s.wavelength), lp_s.flux, "b-", label="1")
                plt.plot(lp_p.converWavelengthtoVelocity(lp_p.wavelength), lp_p.flux * f, "r-", label="blocked")
                plt.plot(lp_2.converWavelengthtoVelocity(lp_2.wavelength), lp_2.flux * self.f12, color="orange", label="2")
                plt.plot(lp_rm.converWavelengthtoVelocity(lp_rm.wavelength), lp_rm.flux, "k-", label="f")
                plt.legend()
                plt.show()  
                lp_s.writeSpec(file_name="lp_s.dat") 
                lp_p.writeSpec(file_name="lp_p.dat") 
                lp_2.writeSpec(file_name="lp_2.dat")
                lp_rm.writeSpec(file_name="lp_rm.dat") 
                lp_1_rm.writeSpec(file_name="lp_1_rm.dat") 

        lp_rm.wavelength = lp_rm.converWavelengthtoVelocity(lp_rm.wavelength)         
        return(lp_rm)

    def calcRVatAtime(self, Msini=1.0, P=1.0, Mtotal=1.0, e=0.0, om=90.0, tp=0.0, _t=0.0):
        K = semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter')
        """
        :param Msini: mass of planet [Mjup]
        :param P: Orbital period [days]
        :param Mtotal: Mass of star + mass of planet [Msun]
        :param e: eccentricity
        :param om: in degree
        :param tp: in days
        :param tint in days
        """
        K = semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter')
        RV = RadialVelocity([P, tp, e, om, K])
        _rv = RV.rv_drive(_t)
        return(_rv)

    def calcRVChangeRateDuringEclipse(self, Msini=1.0, P=1.0, Mtotal=1.0, e=0.0, om=90.0, tp=0.0, tint=1e-2, flag_plot=False):
        K = semi_amplitude(Msini, P, Mtotal, e, Msini_units='jupiter')
        """
        :param Msini: mass of planet [Mjup]
        :param P: Orbital period [days]
        :param Mtotal: Mass of star + mass of planet [Msun]
        :param e: eccentricity
        :param om: in degree
        :param tp: in days
        :param tint in days
        """
        RV = RadialVelocity([P, tp, e, om, K])
        t_arr = np.arange(0, P, tint)
        v_arr = RV.rv_drive(t_arr)
        if flag_plot:
            plt.plot(t_arr, v_arr, "b-o")
            plt.show()
        t1 = self.fingZeroCrossing(t_arr, v_arr)
        dvdt = (RV.rv_drive(np.array([t1+tint])) - RV.rv_drive(np.array([t1-tint]))) / (2.0 * tint)
        if dvdt < 0.0:
            return(RV.rv_drive(np.array([t1])), dvdt / 24.0) # dvdt was m/s/day, now is at m/s/hour
        else:
            t_arr = np.arange(t1+0.1*P, t1+0.8*P+0.1*P, tint)
            v_arr = RV.rv_drive(t_arr)
            if flag_plot:
                plt.plot(t_arr, v_arr, "b-o")
                plt.show()
            t2 = self.fingZeroCrossing(t_arr, v_arr)
            dvdt = (RV.rv_drive(np.array([t2+tint])) - RV.rv_drive(np.array([t2-tint]))) / (2.0 * tint)
            return(RV.rv_drive(np.array([t2])), dvdt / 24.0) # dvdt was m/s/day, now is at m/s/hour

    def fingZeroCrossing(self, t_arr, v_arr):
        ind = np.where(np.abs(v_arr) == np.min(np.abs(v_arr)))[0][0]
        return(t_arr[ind])
            
    def calcBlockedFlux(self, t, v_min, v_max, u1=1.0, u2=0.0):
        cover_area = self.intersection_area(t)
        cover_frac = cover_area / (np.pi * self.R**2)
        ld = self.calcLimbDarkening(v_min, v_max, u1=u1, u2=u2)
        return(ld * cover_frac)

    def calcLimbDarkening(self, v_min, v_max, u1=1.0, u2=0.0):
        # use quadratic limb darkening
        # u1 = 0, u2 = 0, is no limb darkening, see David Sing, https://arxiv.org/pdf/0912.2274.pdf
        v_mean = (v_min + v_max) / 2.0
        mu = np.cos(np.arcsin(np.abs(v_mean) / self.vrot))
        return(1 - u1 * (1 - mu) - u2 * (1 - mu)**2)

    def calcLPSeries(self, t_arr=None, flag_plot=True, flag_save=True, vel_grid=None):
        t_arr, v_arr = self.calcRMSeries(t_arr=t_arr, flag_video=False, flag_plot=flag_plot)   
        lp_arr = [] 
        for i in np.arange(len(t_arr)): 
            lp_rm = self.createRMLineProfile(t_arr[i], v_arr[i,0], v_arr[i,1], u1=self.u1, u2=self.u2, lambda_0=self.lambda0, flag_plot=False)
            if vel_grid is not None:
                lp_rm.resampleSpec(vel_grid, left=0.0, right=0.0)
            lp_arr.append(lp_rm)
            if flag_plot:
                if i != 1e6:
                    plt.plot(lp_rm.wavelength, lp_rm.flux + i * 0.008, lw=3, alpha=0.5, label="{0:03.0f}".format(i))
                else:
                    plt.plot(lp_rm.wavelength - 3e3, lp_rm.flux + i * 0.006, lw=3, alpha=0.5, label="{0:03.0f}".format(i))
            if flag_save:
                file_name = "spec_{0:03.0f}.txt".format(i)
                lp_rm.writeSpec(file_name=file_name)
        if flag_plot:
            plt.legend()
            plt.show()
        return(lp_arr)

    def calcLPSeries_MCMC(self, t_arr=None, flag_plot=False, flag_save=False, vel_grid=None):
        t_arr, v_arr = self.calcRMSeries(t_arr=t_arr, flag_video=False, flag_plot=flag_plot)
        lp_arr = []
        for i in np.arange(len(t_arr)):
            lp_rm = self.createRMLineProfile(t_arr[i], v_arr[i,0], v_arr[i,1], u1=self.u1, u2=self.u2, lambda_0=self.lambda0, flag_plot=False)
            if vel_grid is not None:
                lp_rm.resampleSpec(vel_grid, left=0.0, right=0.0)
            if i == 0:
                lp_arr = np.zeros((len(t_arr), len(lp_rm.flux)))
            lp_arr[i,:] = lp_rm.flux
            if flag_plot:
                if i != 1e6:
                    plt.plot(lp_rm.wavelength, lp_rm.flux + i * 0.008, lw=3, alpha=0.5, label="{0:03.0f}".format(i))
                else:
                    plt.plot(lp_rm.wavelength - 3e3, lp_rm.flux + i * 0.006, lw=3, alpha=0.5, label="{0:03.0f}".format(i))
            if flag_save:
                file_name = "spec_{0:03.0f}.txt".format(i)
                lp_rm.writeSpec(file_name=file_name)
        if flag_plot:
            plt.legend()
            plt.show()
        return(lp_arr)

    def test_code(self):
        #r_arr = [0.2, 0.5, 1.0]
        #b_arr = [1e-6, 0.5, 1.0, 1.1]
        #o_arr = [135, 180, 225, 270, 315]
        r_arr = [0.2]
        b_arr = [0.5]
        o_arr = [45]
        for r in r_arr:
            for b in b_arr:
                for o in o_arr:
                    RossiterMcLaughlinEffect(b=b, t0=4.0, vrot=10e3, oblqt=o, R=1.0, r=r).calcRMSeries(flag_video=False, flag_plot=True)






