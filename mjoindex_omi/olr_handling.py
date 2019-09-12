# -*- coding: utf-8 -*-

""" """

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindex_omi

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Contact: christoph.hoffmann@uni-greifswald.de


from pathlib import Path

import numpy as np
import scipy
import warnings
import scipy.interpolate
from scipy.io import netcdf
import matplotlib.pyplot as plt

import mjoindex_omi.tools as tools

"""
Created on Tue Dec  4 14:05:34 2018

@author: ch
"""


class OLRData:
    def __init__(self, olr, time, lat, long):
        if( olr.shape[0] == time.size):
            self._olr = olr.copy()
            self._time = time.copy()
            self._lat = lat.copy()
            self._long = long.copy()
        else:
            raise ValueError('Length of time grid does not fit to first dimension of OLR data cube')

    @property
    def olr(self):
        return self._olr

    @property
    def time(self):
        return self._time

    @property
    def lat(self):
        return self._lat

    @property
    def long(self):
        return self._long

    def get_olr_for_date(self, date):
        #FIXME: Check if date is in time range
        #FIXME: Unit-tested?
        cand = self.time == date
        #print(cand)
        #print(self.__olr_data_cube.shape)
        return np.squeeze(self.olr[cand,:,:])

    def extract_olr_matrix_for_doy_range(self, center_doy: int, window_length: int = 0) -> np.ndarray:
        """
        Extracts the olr data, which belongs to all doys around one center (center_doy +/- windowlength).
        Keep in mind that the OLR time series might span several years. In this case the center DOy is found more than
        once and the respective window in considered for each year.
        Example: 3 full years of data, centerdoy = 20, and window_length = 4 results in 3*(2*4+1) = 27 entries in the
        time axis
        :param center_doy: The center DOY
        :param window_length: The window length in DOYs on both sides of the center DOY. Hence, if the window is fully
        covered by the data, one gets 2*window_length + 1 entries per year in the result.
        :return: A matrix: 1. index doys, 2. index lat, 3 index long.
        """
        inds, doys  = tools.find_doy_ranges_in_dates(self.time, center_doy, window_length=window_length)
        return self.olr[inds, :, :]

    def save_to_npzfile(self, filename: Path) -> None:
        """
        Saves the data array contained in the OLRData object to a numpy file
        :param filename: The full filename
        """
        np.savez(filename, olr=self.olr, time=self.time, lat=self.lat, long=self.long)


def resample_spatial_grid_to_original(olr: OLRData) -> OLRData:
    """Resamples the data in an OLRData object spatially according to the original OMI EOF grid.

    Afterwards, the data corresponds to the original spatial calculation grid:
    Latitude: 2.5 deg sampling in the tropics from -20 to 20 deg (20S to 20 N)
    Longitude: Whole globe with 2.5 deg sampling

    :param olr: The OLRData object
    :return:  A new OLRData object with the resampled data
    """
    # FIXME Combine with definition in empirical_or....py
    orig_lat = np.arange(-20., 20.1, 2.5)
    orig_long = np.arange(0., 359.9, 2.5)
    return resample_spatial_grid(olr, orig_lat, orig_long)


def resample_spatial_grid(olr: OLRData, target_lat: np.array, target_long: np.array) -> OLRData:
    """ Resamples the OLR data according to the given grids and returns a new OLRData object

    :param olr: The OLR data to resample
    :param target_lat: The new latitude grid
    :param target_long: the new longitude grid
    :return: an OLRData object containing the resampled OLR data
    """
    no_days = olr.time.size
    olr_interpol = np.empty((no_days, target_lat.size, target_long.size))
    for idx in range(0,no_days):
        f = scipy.interpolate.interp2d(olr.long, olr.lat, np.squeeze(olr.olr[idx,:,:]), kind='linear')
        olr_interpol[idx,:,:] = f(target_long, target_lat)
    return OLRData(olr_interpol, olr.time, target_lat, target_long)

#FIXME: doc, unitttest
def restrict_time_coverage(olr: OLRData, start: np.datetime64, stop: np.datetime64) -> OLRData:
    windowInds = (olr.time >= start) & (olr.time <= stop)
    return OLRData(olr.olr[windowInds, :, :], olr.time[windowInds], olr.lat, olr.long)


def load_noaa_interpolated_olr(filename: Path) -> OLRData:
    """Loads the standard OLR data product provided by NOAA

    The dataset can be obtained from
    ftp://ftp.cdc.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc

    A description is found at
    https://www.esrl.noaa.gov/psd/data/gridded/data.interp_OLR.html

    :param filename: full filename of local copy of OLR data file
    :return: The OLR data
    """
    f = netcdf.netcdf_file(str(filename), 'r')
    lat = f.variables['lat'].data.copy()
    lon = f.variables['lon'].data.copy()
    # scaling and offset as given in meta data of nc file
    olr = f.variables['olr'].data.copy()/100.+327.65
    hours_since1800 = f.variables['time'].data.copy()
    f.close()

    temptime = []
    for item in hours_since1800:
        delta = np.timedelta64(int(item/24), 'D')
        day = np.datetime64('1800-01-01') + delta
        temptime.append(day)
    time = np.array(temptime, dtype=np.datetime64)
    result = OLRData(np.squeeze(olr), time, lat, lon)

    return result


def restore_from_npzfile(filename: Path) -> OLRData:
    """
    Loads an OLRData object from the array in a numpy file.
    :param filename: The filename to the .npz file.
    :return: The OLRData object
    """
    with np.load(filename) as data:
        olr = data["olr"]
        time = data["time"]
        lat = data["lat"]
        long = data["long"]
    return OLRData(olr, time, lat, long)

def plot_olr_map_for_date(olr: OLRData, date: np.datetime64):
    # TODO: Plot underlying map

    mapdata = olr.get_olr_for_date(date)

    fig, axs = plt.subplots(1, 1, num="plot_olr_map_for_date", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)

    ax = axs

    c = ax.contourf(olr.long, olr.lat, mapdata)
    fig.colorbar(c, ax=ax, label="OLR")
    ax.set_title("OLR")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    return fig