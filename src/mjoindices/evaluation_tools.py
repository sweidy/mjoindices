# -*- coding: utf-8 -*-

""" """

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindices

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

import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import re
from scipy.stats import norm

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.principal_components as pc
import mjoindices.tools as tools

#FIXME: typing
#FIXME: Comments

def compute_vector_difference_quantity(ref_vec, vec, percentage=True):
    result = vec - ref_vec
    if percentage is True:
        result = result / np.mean(np.abs(ref_vec)) *100
    return result


def calc_vector_agreement(ref_vec, vec, percentage=True, do_print=False):
    if not np.all(ref_vec.size == vec.size):
        raise AttributeError("Vectors do not have the same lenths.")
    corr = np.corrcoef(ref_vec, vec)[0, 1]

    diff_vec = compute_vector_difference_quantity(ref_vec, vec, percentage=percentage)
    diff_mean = np.mean(diff_vec)
    diff_std = np.std(diff_vec)
    diff_abs_percent68 = np.percentile(np.abs(diff_vec), 68.)
    diff_abs_percent95 = np.percentile(np.abs(diff_vec), 95.)
    diff_abs_percent99 = np.percentile(np.abs(diff_vec), 99.)

    if do_print:
        print("CorrelationCoefficient: %1.4f" % corr)
        print("Mean of difference: %1.4f" % diff_mean)
        print("Stddev. of difference: %1.4f" % diff_std)
        print("68%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent68)
        print("95%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent95)
        print("99%% Percentile (abs. value of differences: %1.4f" % diff_abs_percent99)

    return corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99


def calc_comparison_stats_for_eofs_all_doys(eofs_ref: eof.EOFDataForAllDOYs, eofs: eof.EOFDataForAllDOYs, eof_number, exclude_doy366=False, percentage=False, do_print=False) -> typing.Tuple:
    if eof_number != 1 and eof_number != 2:
        raise AttributeError("Argument eof_number must be 1 or 2.")
    doys = eof.doy_list()
    if exclude_doy366:
        doys = doys[:-1]
    corr = np.empty(doys.size)
    diff_mean= np.empty(doys.size)
    diff_std= np.empty(doys.size)
    diff_abs_percent68 = np.empty(doys.size)
    diff_abs_percent95 = np.empty(doys.size)
    diff_abs_percent99 = np.empty(doys.size)

    for idx, doy in enumerate(doys):
        if eof_number == 1:
            eof_ref = eofs_ref.eof1vector_for_doy(doy)
            eof_test = eofs.eof1vector_for_doy(doy)
        elif eof_number == 2:
            eof_ref = eofs_ref.eof2vector_for_doy(doy)
            eof_test = eofs.eof2vector_for_doy(doy)
        corr_single, diff_mean_single, diff_std_single, diff_vec_single, diff_abs_percent68_single, diff_abs_percent95_single, diff_abs_percent99_single = calc_vector_agreement(eof_ref,eof_test,percentage=percentage, do_print=False)
        corr[idx] = corr_single
        diff_mean[idx] = diff_mean_single
        diff_std[idx] = diff_std_single
        diff_abs_percent68[idx] = diff_abs_percent68_single
        diff_abs_percent95[idx] = diff_abs_percent95_single
        diff_abs_percent99[idx] = diff_abs_percent99_single

    if do_print:
        print("########## Summary of EOF comparison for all DOYs (EOF %i)" % eof_number)
        print("Worst Correlation (at DOY %i): %1.4f" % (doys[np.argmin(corr)], np.amin(corr)))
        print("Worst 99%% percentile (at DOY %i): %1.4f" % (doys[np.argmax(diff_abs_percent99)], np.amax(diff_abs_percent99)))
        print("Worst 68%% percentile (at DOY %i): %1.4f" % (doys[np.argmax(diff_abs_percent68)], np.amax(diff_abs_percent68)))

    return corr, diff_mean, diff_std, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99


def plot_comparison_stats_for_eofs_all_doys(recalc_eof: eof.EOFDataForAllDOYs, orig_eof: eof.EOFDataForAllDOYs, exclude_doy366=False, do_print=False) -> Figure:
    """

    :param recalc_eof: The object containing the calculated EOFs
    :param orig_eof: The object containing the ortiginal EOFs
    :return: Handle to the figure
    """

    doys = eof.doy_list()
    if exclude_doy366:
        doys = doys[:-1]
    corr_1, diff_mean_1, diff_std_1, diff_abs_percent68_1, diff_abs_percent95_1, diff_abs_percent99_1 = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=1, percentage=False, do_print=do_print)
    corr_2, diff_mean_2, diff_std_2, diff_abs_percent68_2, diff_abs_percent95_2, diff_abs_percent99_2 = calc_comparison_stats_for_eofs_all_doys(orig_eof, recalc_eof, exclude_doy366=exclude_doy366, eof_number=2, percentage=False, do_print=do_print)

    fig_id = "plot_comparison_stats_for_eofs_all_doys"
    fig, axs = plt.subplots(4, 1, num=fig_id, clear=True, figsize=(12, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3)

    fig.suptitle("Comparison of EOFs for all DOYs")

    ax = axs[0]
    ax.set_title("Correlation")
    p11, = ax.plot(doys, corr_1, label="EOF1", color="blue")
    p12, = ax.plot(doys, corr_2, label="EOF2", color="green")

    ax = axs[1]
    ax.set_title("Mean")
    p21, = ax.plot(doys, diff_mean_1, label="EOF1", color="blue")
    p22, = ax.plot(doys, diff_mean_2, label="EOF2", color="green")

    ax = axs[2]
    ax.set_title("Std Dev")
    p31, = ax.plot(doys, diff_std_1, label="EOF1", color="blue")
    p32, = ax.plot(doys, diff_std_2, label="EOF2", color="green")

    ax = axs[3]
    ax.set_title("Percentiles")
    p31, = ax.plot(doys, diff_abs_percent99_1, label="EOF1", color="blue")
    p32, = ax.plot(doys, diff_abs_percent99_2, label="EOF2", color="green")
    p33, = ax.plot(doys, diff_abs_percent95_1, label="EOF1", color="blue", linestyle="--")
    p34, = ax.plot(doys, diff_abs_percent95_2, label="EOF2", color="green", linestyle="--")
    p35, = ax.plot(doys, diff_abs_percent68_1, label="EOF1", color="blue", linestyle=":")
    p36, = ax.plot(doys, diff_abs_percent68_2, label="EOF2", color="green", linestyle=":")

    return fig

def plot_individual_eof_map_comparison(orig_eof: eof.EOFData, compare_eof: eof.EOFData, doy: int=None):

    # TODO: Print correlation values into figure
    print(np.corrcoef(orig_eof.eof1vector, compare_eof.eof1vector))
    print(np.corrcoef(orig_eof.eof2vector, compare_eof.eof2vector))

    fig, axs = plt.subplots(2, 3, num="ReproduceOriginalOMIPCs_ExplainedVariance_EOF_Comparison", clear=True,
                            figsize=(10, 5), dpi=150, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.35, hspace=0.35)
    if doy is not None:
        fig.suptitle("EOF Recalculation for DOY %i" % doy)

    ax = axs[0, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF1")
    ax.set_ylabel("Latitude [°]")

    ax = axs[0, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF1")

    # FIXME: Check that grids are equal
    ax = axs[0, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof1map - compare_eof.eof1map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 1")

    ax = axs[1, 0]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Original EOF2")
    ax.set_ylabel("Latitude [°]")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 1]
    c = ax.contourf(compare_eof.long, compare_eof.lat, compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Recalculated EOF2")
    ax.set_xlabel("Longitude [°]")

    ax = axs[1, 2]
    c = ax.contourf(orig_eof.long, orig_eof.lat, orig_eof.eof2map - compare_eof.eof2map)
    fig.colorbar(c, ax=ax, label="OLR Anomaly [W/m²]")
    ax.set_title("Difference 2")
    ax.set_xlabel("Longitude [°]")

    return fig

def calc_comparison_stats_for_explained_variance(ref_var, calc_var, do_print=False, exclude_doy366=False):
    ref_data = ref_var.copy()
    calc_data = calc_var.copy()
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        ref_data = ref_data[:-1]
        calc_data = calc_data[:-1]
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = calc_vector_agreement(ref_data, calc_data, percentage=False, do_print=do_print)
    return corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99

def plot_comparison_stats_for_explained_variance(ref_var, calc_var, title=None, do_print=False, exclude_doy366=False):
    if do_print:
        print("##########")
        if title is not None:
            print(title)
    ref_data = ref_var.copy()
    calc_data = calc_var.copy()
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        ref_data = ref_data[:-1]
        calc_data = calc_data[:-1]
    fig = plot_vector_agreement(ref_data, calc_data, title=title, do_print=do_print)
    return fig


def calc_timeseries_agreement(ref_data, ref_time, data, time, exclude_doy366=False, do_print=False):
    if not np.all(ref_time == time):
        raise AttributeError("Time series do not cover the same periods.")
    if exclude_doy366:
        if do_print:
            print("###### DOY 366 excluded")
        doys = tools.calc_day_of_year(ref_time)
        inds_used = np.nonzero(doys != 366)
        inds_not_used = np.nonzero(doys == 366)
        calc_ref_data = ref_data[inds_used]
        calc_data = data[inds_used]
    else:
        if do_print:
            print("##### Complete time series")
        calc_ref_data = ref_data
        calc_data = data
        inds_used = (np.arange(0, ref_time.size, 1),)
        inds_not_used = (np.array([], dtype="int64"),)

    corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = calc_vector_agreement(calc_ref_data, calc_data, percentage=False, do_print=do_print)

    return inds_used, inds_not_used, corr, diff_mean, diff_std, diff_ts, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99

def plot_timeseries_agreement(ref_data, ref_time, data, time, title = None, do_print=False):

    if do_print:
        print("##########")
        if title is not None:
            print(title)
    tempa, tempb, corr_complete, diff_mean_complete, diff_std_complete, diff_ts_abs_complete, diff_abs_percent68_complete, diff_abs_percent95_complete, diff_abs_percent99_complete= calc_timeseries_agreement(ref_data, ref_time, data, time, exclude_doy366=False, do_print=do_print)
    doy_not_366_inds, doy366_inds, corr_not366, diff_mean_not366, diff_std_not366, diff_ts_abs_not366, diff_abs_percent68_not366, diff_abs_percent95_not366, diff_abs_percent99_not366 = calc_timeseries_agreement(ref_data, ref_time, data, time, exclude_doy366=True, do_print=do_print)


    ref_data_not366 = ref_data[doy_not_366_inds]
    data_not366 = data[doy_not_366_inds]

    fig_id = "evaluate_timeseries_agreement"
    if title is not None:
        fig_id = fig_id + "_" + title
    fig, axs = plt.subplots(2, 2, num=fig_id, clear=True, figsize=(12, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    if title is not None:
        fig.suptitle(title)

    ax = axs[0, 0]
    ax.set_title("Time series")
    p11, = ax.plot(time, ref_data, label="Reference")
    p12, = ax.plot(time, data, label="Recalculation")

    ax = axs[1, 0]
    ax.set_title("Difference")
    p21, = ax.plot(time, diff_ts_abs_complete, color="red")
    p22, = ax.plot(time[doy_not_366_inds], diff_ts_abs_not366, color="blue")

    ax = axs[0, 1]
    ax.set_title("Scatterplot")
    p31, = ax.plot(ref_data_not366, data_not366, linestyle='None', marker="x", color="blue")
    p32, = ax.plot(ref_data[doy366_inds], data[doy366_inds], linestyle='None', marker="x", color="red")

    ax = axs[1, 1]
    (n2, bins2, patches) = ax.hist(diff_ts_abs_not366, bins=100, density=True)

    sigma = diff_std_not366
    mu = diff_mean_not366
    ax.plot(bins2, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu) ** 2 / (2 * sigma ** 2)), linewidth = 2, color = 'r')

    return fig


def plot_vector_agreement(ref_data, data, title=None, do_print=False):

    if do_print:
        print("##########")
        if title is not None:
            print(title)
    corr, diff_mean, diff_std, diff_vec, diff_abs_percent68, diff_abs_percent95, diff_abs_percent99 = calc_vector_agreement(ref_data, data, percentage=False, do_print=do_print)

    fig_id = "evaluate_vector_agreement"
    if title is not None:
        fig_id = fig_id + "_" + title
    fig, axs = plt.subplots(2, 2, num=fig_id, clear=True, figsize=(12, 9), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    if title is not None:
        fig.suptitle(title)

    ax = axs[0, 0]
    ax.set_title("Data")
    p11, = ax.plot(ref_data, label="Reference")
    p12, = ax.plot(data, label="Recalculation")

    ax = axs[1, 0]
    ax.set_title("Difference")
    p21, = ax.plot( diff_vec, color="red")

    ax = axs[0, 1]
    ax.set_title("Scatterplot")
    p31, = ax.plot(ref_data, data, linestyle='None', marker="x", color="blue")

    ax = axs[1, 1]
    (n2, bins2, patches) = ax.hist(diff_vec, bins=100, density=True)

    sigma = diff_std
    mu = diff_mean
    ax.plot(bins2, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins2 - mu) ** 2 / (2 * sigma ** 2)), linewidth = 2, color = 'r')

    return fig


def plot_comparison_orig_calc_pcs(calc_pcs: pc.PCData, orig_pcs: pc.PCData, startDate=None, endDate=None):
    fig, axs = plt.subplots(2, 1, num="ReproduceOriginalOMIPCs_PCs", clear=True, figsize=(8, 6), dpi=150)
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("PC Recalculation")

    ax = axs[0]
    ax.set_title("PC1")
    p1, = ax.plot(orig_pcs.time, orig_pcs.pc1, label="Original")
    p2, = ax.plot(calc_pcs.time, calc_pcs.pc1, label="Recalculation")
    if startDate != None and endDate != None:
        ax.set_xlim((startDate, endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
    ax.legend(handles=(p1, p2))

    #corr = (np.corrcoef(orig_pcs.pc1, calc_pcs.pc1))[0, 1]
    # FIXME: Calculate correlation only for wanted period
    # FIXME: Check that periods covered by orig_omi and recalc_omi are actually the same
    #plt.text(0.1, 0.1, "Correlation over complete period: %.3f" % corr, transform=ax.transAxes)

    ax = axs[1]
    ax.set_title("PC2")
    p3, = ax.plot(orig_pcs.time, orig_pcs.pc2, label="Original")
    p4, = ax.plot(calc_pcs.time, calc_pcs.pc2, label="Recalculation")
    if startDate is not None and endDate is not None:
        ax.set_xlim((startDate, endDate))
    plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment="right")
    ax.legend(handles=(p3, p4))

    #corr = (np.corrcoef(orig_pcs.pc2, calc_pcs.pc2))[0, 1]
    #plt.text(0.1, 0.1, "Correlation over complete period: %.3f" % corr, transform=ax.transAxes)

    return fig


def load_omi_explained_variance(filename:str) -> np.ndarray:
    c = lambda s: float(re.sub("[ \[;\]]", "", s.decode("utf-8")))
    data = np.genfromtxt(filename, converters={0: c, 1: c}, skip_header=7)
    var1 = data[:, 0]
    var2 = data[:, 1]
    return var1, var2


