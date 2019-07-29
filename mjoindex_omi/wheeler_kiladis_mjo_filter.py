# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:14:10 2019

@author: ch
"""
import numpy as np
import matplotlib.pyplot as plt
import mjoindex_omi.olr_handling
import scipy
import scipy.fftpack
from scipy.io import FortranFile

def filterOLRForMJO_PC_CalculationWith1DSpectralSmoothing(olr):
    return filterOLRTemporallyWith1DSpectralSmoothing(olr, 20., 96.)

def filterOLRTemporallyWith1DSpectralSmoothing(olr, period_min, period_max):
    print("Smooth data temporally...")
    #FIXME: Don't use zeros in the follwing
    filteredOLR = np.zeros(olr.olr.shape)
    for idx_lat in range (0,olr.olr.shape[1]):
        for idx_lon in range (0,olr.olr.shape[2]):
            tempolr = np.squeeze(olr.olr[:,idx_lat, idx_lon])
            filteredOLR[:,idx_lat, idx_lon] = __performSpectralSmoothing(tempolr, period_min, period_max)
    #fig = plt.figure()
    #plt.contourf(np.squeeze(filteredOLR[0,:,:]))
    #plt.colorbar()
    #plt.title("Filtered OLR")
    return (mjoindex_omi.olr_handling.OLRData(filteredOLR, olr.time, olr.lat, olr.long))

def __performSpectralSmoothing(y, lowerCutOff, HigherCutOff ):
    #FIXME: calculate dt dynamically
    dt = 1 #day
    N=y.size
    #print(y.shape)
    w = scipy.fftpack.rfft(y)
    f = scipy.fftpack.rfftfreq(N, dt)
    P=1/f
    #spectrum = w**2

    w2 = w.copy()
    w2[P<lowerCutOff] = 0
    w2[P > HigherCutOff] = 0

#    fig = plt.figure()
#    plt.plot(P,spectrum)
#    plt.plot(P,w2**2)

    #w2[cutoff_idx] = 0

    y2 = scipy.fftpack.irfft(w2)
#    fig = plt.figure()
#    plt.plot(y)
#    plt.plot(y2)
    return y2

def filterOLRForMJO_PC_Calculation(olr, do_plot=0):
    return filterOLRTemporally(olr,  20., 96., do_plot=do_plot)

def filterOLRTemporally(olr, period_min, period_max, do_plot=0):
    return filterOLRTemporallyandLongitudinally(olr,  period_min, period_max, -720., 720, do_plot=do_plot)

def filterOLRForMJO_EOF_Calculation(olr, do_plot=0):
    return filterOLRTemporallyandLongitudinally(olr,  30., 96., 0., 720, do_plot=do_plot)

def filterOLRTemporallyandLongitudinally(olr, period_min, period_max, wn_min, wn_max, do_plot=0):
    print("Smooth data temporally and longitudally...")
        #FIXME: Don't use zeros in the follwing

        #nt,nlat,nlong = self.__olr_data_cube.shape

#        fig = plt.figure()
#        plt.contourf(np.squeeze(self.__olr_data_cube[0,:,:]))
#        plt.colorbar()
#        plt.title("Unfiltered OLR Data")
#
#
#        fig,axs = plt.subplots(3,1)
#        ax = axs[0]
#        ax.plot(np.squeeze(self.__olr_data_cube[:,10,10]))
#        ax.set_title("Unfiltered OLR Data Timeseries")
#
#        ax = axs[1]
#        ax.plot(np.squeeze(self.__olr_data_cube[10,10,:]))
#        ax.set_title("Unfiltered OLR Data Long-Evolution")
#
#
#        ax = axs[2]
#        ax.plot(np.squeeze(self.__olr_data_cube[10,:,10]))
#        ax.set_title("Unfiltered OLR Data lat-Evolution")
#

    filtered_olr = np.zeros(olr.olr.shape)



    #ilat= 10

    for ilat, lat in enumerate(olr.lat):
        print("Calculating for latitude: ", lat)
        time_spacing = (olr.time[1] - olr.time[0]).astype('timedelta64[s]')/np.timedelta64(1, 'D')  #time spacing in days
        print("Spacing: ",time_spacing)


        dataslice = np.squeeze(olr.olr[:,ilat,:])
        wkfilter = WKFilter()
        filtered_data = wkfilter.perform2dimSpectralSmoothing(dataslice, time_spacing, period_min, period_max, wn_min, wn_max, do_plot = do_plot, save_debug= 0)
        filtered_olr[:,ilat,:] = filtered_data

    return (mjoindex_omi.olr_handling.OLRData(filtered_olr, olr.time, olr.lat, olr.long))

#        fig = plt.figure()
#        plt.contourf(np.squeeze(filtered_olr[0,:,:]))
#        plt.colorbar()
#        plt.title("Filtered OLR")
#
#        fig,axs = plt.subplots(3,1)
#
#        ax = axs[0]
#        ax.plot(np.squeeze(filtered_olr[:,10,10]))
#        ax.set_title("Filtered OLR Data Timeseries")
#
#        ax = axs[1]
#        ax.plot(np.squeeze(filtered_olr[10,10,:]))
#        ax.set_title("Filtered OLR Data Long-Evolution")
#
#        ax = axs[2]
#        ax.plot(np.squeeze(filtered_olr[10,:,10]))
#        ax.set_title("Filtered OLR Data lat-Evolution")

#        fig = plt.figure()
#        plt.plot(np.squeeze(self.__olr_data_cube[:,10,70]))
#        plt.plot(np.squeeze(filtered_olr[:,10,70]))
#        plt.title("Original and fitered OLR for particular location")


def detrendTS(ts):

    x=np.arange(0, ts.size, 1)
    A = np.vstack([x, np.ones(len(x))]).T
    #FIXME: Remove FutureWarning  `rcond` parameter will change to the default of machine precision times ``max(M, N)``,...  pass `rcond=None`, to keep using the old, explicitly pass `rcond=-1`.
    m, b = np.linalg.lstsq(A, ts)[0]

    result = ts - (m*x+b)
#    plt.plot(ts)
#    plt.plot(m*x+b)
#    plt.plot(result)
    return result

def taperTSToZero(ts, window_length):
    startinds = np.arange(0,window_length,1)
    endinds = np.arange(-window_length-1, -1,1)+2

    result = ts
    result[0:window_length] =  result[0:window_length] *0.5*(1-np.cos(startinds*np.pi/window_length))
    result[ts.size-window_length:ts.size] =  result[ts.size-window_length:ts.size] *0.5*(1-np.cos(endinds*np.pi/window_length))
#    plt.figure()
#    plt.plot(ts)
#    plt.plot(result)
    return result


class WKFilter:
    def __init__(self):
        self.DebugInputOLR = []
        self.DebugFilterOLR = []
        self.DebugDetrendedOLR = []
        self.DebugPreprocessedOLR = []
        self.DebugFreqAxis = []
        self.DebugWNAxis = []
        self.DebugOriginalFourierSpectrum = []
        self.DebugFilteredFourierSpectrum = []
        self.DebugNoElementsInFilteredSpectrum = []


    def perform2dimSpectralSmoothing(self, data, time_spacing, period_min, period_max, wn_min, wn_max, do_plot = 0, save_debug= 0):
        """
        Bandpass-filters OLR data in time- and longitude direction according to
        the original Kiladis algoithm.

        Note that the temporal and longitudinal dimension have in principle
        different characteristics, so that they are in detail treated a bit
        differently:
        While the time is evolving into infinity (so that the number of data
        points and the time_spacing variable are needed to calculate the
        full temporal coverage), the longitude is periodic with the periodicity
        of one globe (so that it is assumed that the data covers eactly one
        globe and passing implicitely only the number
        of longitudes provides already the complete information)

        Parameters
        ----------
        data: numpy.array
            The OLR data as 2-dim array: first dimension time, second
            dimension longitude, both equally spaced.
            The longitudinal dimension has to cover the full globe.
            The time dimension is further described by the variable
            `time_spacing`.
        time_spacing:
            Temporal resolution of the data in days (often 1 or 0.5 (if two
            data points exist per day))
        period_min:
            Minimal period (in days) that remains in the dataset
        period_max:
            Maximal period (in days) that remains in the dataset
        wn_min:
            Minimal wavenumber (in cycles per globe) that remains in the dataset
        wn_max:
            Maximal wavenumber (in cycles per globe) that remains in the dataset

        """

        ####################### Process input data #######################
        if save_debug:
            self.DebugInputOLR = np.copy(data)

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_DataIn",clear=True)
            plt.contourf(data)
            plt.colorbar()
            plt.title("Original Data")

        dataperday = 1/time_spacing
        freq_min = 1/period_max
        freq_max = 1/period_min

        ######################### Detrend #################################
        #"orig" refers to the original size in the time dimension in the following, i.e. not the zero-padded version.
        orig_data = data
        orig_nt,nl=orig_data.shape

        for idx_l in range(0,nl):
            orig_data[:,idx_l] = detrendTS(orig_data[:,idx_l])

        if save_debug:
            self.DebugDetrendedOLR = np.copy(orig_data)
        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_Detrended",clear=True)
            plt.contourf(orig_data)
            plt.colorbar()
            plt.title("Detrended Data")

        ######################### Zero Padding ############################
        nt = 2**17 # Zero padding for performance and resolution optimization, as well as consistency with origininal Kiladis code

        if orig_nt > nt:
            raise ValueError('Time series is longer than hard-coded value for zero-padding!')

        data = np.zeros([nt, nl])
        data[0:orig_nt, :] = orig_data

        ######################### Tapering to zero ########################
        #10 days tapering according ot Kiladis Code
        #only relevant at beginning of time series as it is zero-padded in the end
        for idx_l in range(0,nl):
            data[:,idx_l] = taperTSToZero(data[:,idx_l],int(10*dataperday))

        if save_debug:
            self.DebugPreprocessedOLR = np.copy(data)

        ########################### Forward Fourier transform ############
        fourier_fft = np.fft.fft2(data)
        #reordering of spectrum is done to be consistent with the original kiladis ordering.
        fourier_fft = np.fft.fftshift(fourier_fft, axes=(0,1))
        fourier_fft = np.roll(fourier_fft, int(nt/2), axis=0)
        fourier_fft = np.roll(fourier_fft, int(nl/2),axis=1)

        ### Calculation of the frequency grid in accordance with Kiladis code
        freq_axis=np.zeros(nt)
        for i_f in range(0,nt):
            if(i_f <= nt/2):
                freq_axis[i_f]=i_f*dataperday/nt
            else:
                freq_axis[i_f]=-1*(nt-i_f)*dataperday/nt
        #the following code based on scipy function produces qualitatively the same grid.
        #However, numerical differeces seem to have larger effect for the filtering step.
        #freq_axis = np.fft.fftfreq(nt, d=time_spacing)
        #freq_axis = np.fft.fftshift(freq_axis)
        #freq_axis = np.roll(freq_axis, int(nt/2))

        ### Calculation of the wavenumber grid in accordance with Kiladis code
        wn_axis=np.zeros(nl)
        for i_wn in range(0,nl):
            if(i_wn <= nl/2):
                wn_axis[i_wn]= -1*i_wn
                #note: to have this consistent with the time-dimension, one could write wn_axis[i_wn]= -1*i_wn*dataperglobe/nl
                #However, since data is required to cover always one globe nl will always be equal to dataperglobe
                #The sign is not consistent with the time dimension, which is for reasons of consitency with the original Kiladis implementation
            else:
                wn_axis[i_wn]= nl-i_wn
        #the following code based on scipy function produces qualitatively the same grid.
        #However, numerical differeces seem to have larger effect for the filtering step.
        #wn_axis = np.fft.fftfreq(nl, d=dy)
        #wn_axis = np.fft.fftshift(wn_axis)  #identical with  wn_axis=np.arange(-int(nlong/2), int(nlong/2),1.)
        #wn_axis = -1 *wn_axis #FIXME: can this be justified? Is needed to reprodice Kiladis FFT. Otherwise mirrored...
        #wn_axis = np.roll(wn_axis, int(nl/2))

        if save_debug:
            self.DebugFreqAxis = np.copy(freq_axis)
            self.DebugWNAxis = np.copy(wn_axis)
            self.DebugOriginalFourierSpectrum = np.copy(fourier_fft)

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_freqAxis",clear=True)
            plt.plot(freq_axis)
            plt.title("Calc freq axis")

            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_WNAxis",clear=True)
            plt.plot(wn_axis)
            plt.title("Calc wn axis")

            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_Spectrum",clear=True)
            plt.contourf(wn_axis,freq_axis,np.squeeze(fourier_fft))
            plt.colorbar()
            plt.title("Fourier Transformation")

        #################### Filtering of the Fourier Spectrum #############
        ### name filter boundaries like in Kiladis Fortran Code
        f1=freq_min
        f2=freq_min
        f3=freq_max
        f4=freq_max
        s1=wn_min
        s2=wn_max
        s3=wn_min
        s4=wn_max

        #### Very similar to original Kiladis Code
        fourier_fft_filtered =  fourier_fft
        count = 0
        for i_f in range(0,int(nt/2)+1):
            for i_wn in range(0,nl):
                ff=freq_axis[i_f]
                ss=wn_axis[i_wn]
                if ((ff >= ((ss*(f1-f2)+f2*s1-f1*s2) / (s1-s2))) and \
                    (ff <= ((ss*(f3-f4)+f4*s3-f3*s4) / (s3-s4))) and \
                    (ss >= ((ff*(s3-s1)-f1*s3+f3*s1) / (f3-f1))) and \
                    (ss <= ((ff*(s4-s2)-f2*s4+f4*s2) / (f4-f2)))):
                        count = count + 1
                else:
                   fourier_fft_filtered[i_f,i_wn]=0
                   if(i_wn == 0 and i_f == 0):
                       pass
                   elif(i_wn == 0):
                       ind_f = nt-i_f
                       if(ind_f < nt):
                           fourier_fft_filtered[ind_f,i_wn]=0
                   elif(i_f == 0):
                       ind_wn = nl-i_wn
                       if(ind_wn < nl):
                           fourier_fft_filtered[i_f,ind_wn]=0
                   else:
                       ind_f = nt-i_f
                       ind_wn = nl-i_wn
                       if(ind_f < nt and ind_wn < nl):
                           fourier_fft_filtered[ind_f,ind_wn]=0
        if save_debug:
            self.DebugFilteredFourierSpectrum = np.copy(fourier_fft_filtered)
            self.DebugNoElementsInFilteredSpectrum = count

        if do_plot:
            fig = plt.figure("WK_Filter_perform2dimSpectralSmoothing_FilteredSpectrum",clear=True)
            plt.contourf(wn_axis,freq_axis,np.squeeze(fourier_fft_filtered))
            plt.colorbar()
            plt.title("Filtered Fourier Transformation")
            print("Number of elements in filtered spectrum: ", count)

        ############################# FFT Backward transformation ############
        ##reorder spectrum back from kiladis ordering to python ordering
        fourier_fft_filtered = np.roll(fourier_fft_filtered, -int(nt/2), axis=0)
        fourier_fft_filtered = np.roll(fourier_fft_filtered, -int(nl/2),axis=1)
        fourier_fft_filtered = np.fft.ifftshift(fourier_fft_filtered, axes=(0,1))
        filtered_olr = np.fft.ifft2(fourier_fft_filtered)
        filtered_olr = np.real(filtered_olr)

        ############################## remove zero padding elements ##########
        result = np.zeros([orig_nt,nl])
        result = filtered_olr[0:orig_nt,:]

        if save_debug:
            self.DebugFilterOLR = np.copy(result)

        if do_plot:
            fig = plt.figure("perform2dimSpectralSmoothing_4",clear=True)
            plt.contourf(result)
            plt.colorbar()
            plt.title("Filtered Data")

        return result

class WKFilterValidator:

    def __init__(self, data_exchange_dir=".", do_plot=1):
        self.__data_exchange_dir = data_exchange_dir
        self.__do_plot = do_plot

    def validate_WKFilter_perform2dimSpectralSmoothing_MJOConditions(self):
        kiladis_olr = Datasets.OLR.loadKiladisBinaryOLRDataTwicePerDay(self.__data_exchange_dir + "/olr.2x.7918.b")
        testdata = np.squeeze(kiladis_olr.OLRData[:,0,:]) #select one latitude
        self.validate_WKFilter_perform2dimSpectralSmoothing(testdata, 0.5, 30., 96., 0., 720)

    def validate_WKFilter_perform2dimSpectralSmoothing(self, data, time_spacing, period_min, period_max, wn_min, wn_max,):
        testfilter = WKFilter()
        testfilter.perform2dimSpectralSmoothing(data, time_spacing, period_min, period_max, wn_min, wn_max, do_plot = 1, save_debug= 1)

        ############## Input OLR
        k_inputOLR = self.loadKiladisOriginalOLR(self.__data_exchange_dir + "/OLROriginal.b")
        new_inputOLR = testfilter.DebugInputOLR

        if self.__do_plot:
             fig, axs = plt.subplots(1,3,num="WKFilterValidator_InputOLR",clear=True,  figsize=(6, 4), dpi=150)
             fig.suptitle("Input OLR")

             ax = axs[0]
             c=ax.contourf(k_inputOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Kiladis")

             ax = axs[1]
             c=ax.contourf(new_inputOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Recalculation")

             ax = axs[2]
             c=ax.contourf(k_inputOLR-new_inputOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Difference (absolute Units)")

        ############## Preprocessed OLR
        k_preprocessed_olr=self.loadKiladisPreprocessedOLR(self.__data_exchange_dir +"/OLRBeforeFFT.b")
        new_preprocessedOLR = testfilter.DebugPreprocessedOLR

        k_nt = k_preprocessed_olr.shape[0]

        if self.__do_plot:
             fig, axs = plt.subplots(1,3,num="WKFilterValidator_PreprocessedOLR",clear=True,  figsize=(6, 4), dpi=150)
             fig.suptitle("Preprocessed OLR (zero padding not shown)")

             ax = axs[0]
             c=ax.contourf(k_preprocessed_olr)
             fig.colorbar(c,ax=ax)
             ax.set_title("Kiladis")

             ax = axs[1]
             c=ax.contourf(new_preprocessedOLR[0:k_nt,:])
             fig.colorbar(c,ax=ax)
             ax.set_title("Recalculation")

             ax = axs[2]
             c=ax.contourf(k_preprocessed_olr-new_preprocessedOLR[0:k_nt,:])
             fig.colorbar(c,ax=ax)
             ax.set_title("Difference (absolute Units)")

        ############## Freq Axis
        k_freq_axis = self.loadKiladisFF(self.__data_exchange_dir +"/ff.b")
        new_freq_axis = testfilter.DebugFreqAxis
        if self.__do_plot:
            fig, axs = plt.subplots(1,3,num="WKFilterValidator_FreqAxis",clear=True,  figsize=(6, 4), dpi=150)
            fig.suptitle("Frequency Axis")

            ax = axs[0]
            c=ax.plot(k_freq_axis)
            ax.set_title("Kiladis")

            ax = axs[1]
            c=ax.plot(new_freq_axis)
            ax.set_title("Kiladis")

            ax = axs[2]
            c=ax.plot(k_freq_axis-new_freq_axis)
            ax.set_title("Difference (absolute Units)")

        ############## Wavenumber Axis
        k_wn_axis = self.loadKiladisSS(self.__data_exchange_dir +"/ss.b")
        new_wn_axis = testfilter.DebugWNAxis
        if self.__do_plot:
            fig, axs = plt.subplots(1,3,num="WKFilterValidator_WNAxis",clear=True,  figsize=(6, 4), dpi=150)
            fig.suptitle("Wavenumber Axis")

            ax = axs[0]
            c=ax.plot(k_wn_axis)
            ax.set_title("Kiladis")

            ax = axs[1]
            c=ax.plot(new_wn_axis)
            ax.set_title("Kiladis")

            ax = axs[2]
            c=ax.plot(k_wn_axis-new_wn_axis)
            ax.set_title("Difference (absolute Units)")

        ############## Original Fourier Spectrum
        k_origFourierSpectrum=self.loadKiladisFFT(self.__data_exchange_dir +"/FFT.b")
        new_origFourierSpectrum= testfilter.DebugOriginalFourierSpectrum

        if self.__do_plot:
             fig, axs = plt.subplots(1,3,num="WKFilterValidator_OrigFourierSpectrum",clear=True,  figsize=(6, 4), dpi=150)
             fig.suptitle("Original Fourier Spectrum")

             ax = axs[0]
             c=ax.contourf(k_origFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Kiladis")

             ax = axs[1]
             c=ax.contourf(new_origFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Recalculation")

             ax = axs[2]
             c=ax.contourf(k_origFourierSpectrum/new_origFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Quotient")

        ############## Filtered Fourier Spectrum
        k_filteredFourierSpectrum=self.loadKiladisFFT(self.__data_exchange_dir +"/FFTfiltered.b")
        new_filteredFourierSpectrum= testfilter.DebugFilteredFourierSpectrum

        if self.__do_plot:
             fig, axs = plt.subplots(1,3,num="WKFilterValidator_FilteredFourierSpectrum",clear=True,  figsize=(6, 4), dpi=150)
             fig.suptitle("Filtered Fourier Spectrum")

             ax = axs[0]
             c=ax.contourf(k_wn_axis, k_freq_axis,k_filteredFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Kiladis")

             ax = axs[1]
             c=ax.contourf(new_wn_axis,new_freq_axis, new_filteredFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Recalculation")

             ax = axs[2]
             c=ax.contourf(k_wn_axis, k_freq_axis, new_filteredFourierSpectrum/k_filteredFourierSpectrum)
             fig.colorbar(c,ax=ax)
             ax.set_title("Quotient")

             print("Number of elements in filtered Spectrum:", testfilter.DebugNoElementsInFilteredSpectrum)


        ############## Filtered OLR
        k_filteredOLR = self.loadKiladisFilteredOLR(self.__data_exchange_dir + "/OLRfiltered.b")
        new_filteredOLR = testfilter.DebugFilterOLR

        if self.__do_plot:
             fig, axs = plt.subplots(1,3,num="WKFilterValidator_FilteredOLR",clear=True,  figsize=(8, 4), dpi=150)

             ax = axs[0]
             c=ax.contourf(k_filteredOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Kiladis")

             ax = axs[1]
             c=ax.contourf(new_filteredOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Recalculation")

             ax = axs[2]
             c=ax.contourf(k_filteredOLR-new_filteredOLR)
             fig.colorbar(c,ax=ax)
             ax.set_title("Difference (absolute Units)")

    ################### Loading procedure for output of Kiladis debug data.

    def loadKiladisOriginalOLR(self,filename):
        nl = 144
        nt = 28970
        f = FortranFile(filename, 'r')
        olr=np.zeros([nt,nl])
        for i_l in range(0,nl):
            record1 = np.squeeze(f.read_record('(1,28970)<f4'))
            olr[:,i_l] = record1
        return olr

    def loadKiladisFilteredOLR(self,filename):
        f = FortranFile(filename, 'r')
        record1 = np.squeeze(f.read_record('(28970,144)<complex64')) #
        return record1

    def loadKiladisPreprocessedOLR(self, filename):
        nl = 144
        nt = 28970
        f = FortranFile(filename, 'r')
        olr=np.zeros([nt,nl])
        for i_l in range(0,nl):
            record1 = np.squeeze(f.read_record('(1,28970)<f4'))
            olr[:,i_l] = record1
        return olr

    def loadKiladisSS(self, filename):
        f = FortranFile(filename, 'r')
        record1 = np.squeeze(f.read_record('(1,144)<f4'))
        return record1

    def loadKiladisFF(self, filename):
        f = FortranFile(filename, 'r')
        record1 = np.squeeze(f.read_record('(1,131072)<f4'))
        return record1

    def loadKiladisFFT(self, filename):
        f = FortranFile(filename, 'r')
        record1 = np.squeeze(f.read_record('(131072,144)<complex64'))
        return record1



if __name__ == '__main__':

    #official calidation is done with file Christoph/MJO/MJOIndexRecalculation/ValidateWKFilterWithOriginalFortranCodeOutput.py

    validator = WKFilterValidator(data_exchange_dir="/home/ch/UPSoftware/Christoph/MJO/MJOIndexRecalculation/GKiladisFiltering")
    validator.validate_WKFilter_perform2dimSpectralSmoothing_MJOConditions()