"""
Test file for EOF rotation procedure
"""

from pathlib import Path
from typing import Tuple
import os.path
import inspect

import numpy as np
import pandas as pd
import pytest

import mjoindices.empirical_orthogonal_functions as eof
import mjoindices.olr_handling as olr
import mjoindices.omi.omi_calculator as omi
import mjoindices.principal_components as pc
import mjoindices.tools as tools
import mjoindices.omi.eof_rotation as omir

reference_eofs_rotated_filename = Path(os.path.abspath('')) / "testdata" / "rotation_reference" / "EOFs_rotated.npz"
reference_pcs_rotated_filename = Path(os.path.abspath('')) / "testdata" / "rotation_reference" / "PCs_rotated.txt"
mjoindices_reference_eofs_filename = Path(os.path.abspath('')) / "testdata" / "mjoindices_reference" / "EOFs.npz"

def test_calculate_angle_btwn_vectors():
    # create two sample vectors, compute the angle between them (angle_btwn_vectors)
    # compute the angle between two reference eofs (angle_btwn_eofs)
    errors = []

    sample_vect1 = [1,2,3]
    sample_vect2 = [3,-2,1]
    result_sample = omir.angle_btwn_vectors(sample_vect1, sample_vect2)
    expected_sample = 1.4274487578895312
    if not np.isclose(result_sample, expected_sample):
        errors.append("calculated angle between vectors does not match") 

    rotated_eofs = eof.restore_all_eofs_from_npzfile(reference_eofs_rotated_filename)
    doy1 = rotated_eofs.eofdata_for_doy(1)
    doy50 = rotated_eofs.eofdata_for_doy(50)
    result_d50 = omir.angle_between_eofs(doy1, doy50)
    expected_d50 = (0.17841331424520707, 0.18532438957723069)
    if not np.allclose(result_d50, expected_d50):
        errors.append("calculated angle for EOFs on DOY 50 do not match")

    result_d1 = omir.angle_between_eofs(doy1, doy1)
    expected_d1 = (0.,0.)
    if not np.allclose(result_d1, expected_d1):
        errors.append("calculated angle between identical EOFs do not match") 

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

def test_create_rotation_matrix():
    # check against a sample rotation matrix
    errors = []

    theta = 0
    result_0 = omir.rotation_matrix(theta)
    expected_0 = np.array([[1,0], [0,1]])
    if not np.allclose(result_0, expected_0):
        errors.append("rotation matrix for theta = 0 do not match")

    theta = np.pi
    result_pi = omir.rotation_matrix(theta)
    expected_pi = np.array([[-1,0],[0,-1]])
    if not np.allclose(result_pi, expected_pi):
        errors.append("rotation matrix for theta = pi do not match")

    theta = -0.00016395825437902885 
    result_disc = omir.rotation_matrix(theta)
    expected_disc = np.array([[ 9.99999987e-01,  1.63958254e-04],[-1.63958254e-04,  9.99999987e-01]])
    if not np.allclose(result_disc, expected_disc):
        errors.append("rotation matrix for theta = pi do not match")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors)) 

@pytest.mark.slow
def test_calculate_discontinuity():

    errors = []

    # use original reference eofs to check discontinuity function gets the
    # right value, using 366 and 365 days in a year
    original_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename)

    discont_test_366 = omir.calculate_angle_from_discontinuity(original_eofs, no_leap=False)
    discont_expected_366 = -0.00016395825437902885
    discont_test_365 = omir.calculate_angle_from_discontinuity(original_eofs, no_leap=True)
    discont_expected_365 = -0.00016440745507596277 
    if not np.isclose(discont_test_366, discont_expected_366):
        errors.append("discontinuity after initial rotation does not match")
    if not np.isclose(discont_test_365, discont_expected_365):
        errors.append("discontinuity calculated with no_leap = True does not match")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_normalization():
    # normalize a set of sample EOFs

    errors = []

    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    no_leap = False
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    eofs = eof.EOFDataForAllDOYs(eofs, no_leap)
    
    expected_norm = (1.,1.)

    eofs_norm = omir.normalize_eofs(eofs, no_leap=False)
    for idx, target_eof in enumerate(eofs_norm.eof_list):
        if not np.allclose(expected_norm, (np.linalg.norm(target_eof.eof1vector), 
                                            np.linalg.norm(target_eof.eof2vector))):
            errors.append("eofs were not normalized for index %i" % idx)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.slow
def test_rotate_eofs():
    # pass in original reference eofs with correct delta and make sure it matches
    # with rotated reference eofs. 

    errors = []

    original_eofs = eof.restore_all_eofs_from_npzfile(mjoindices_reference_eofs_filename)
    discont_366 = -0.00016395825437902885
    
    rotated_test = omir.rotate_each_eof_by_delta(original_eofs, discont_366, no_leap=False)

    # validate rotated EOFs against reference EOFs
    rotated_eofs = eof.restore_all_eofs_from_npzfile(reference_eofs_rotated_filename)
    for idx, target_eof in enumerate(rotated_test.eof_list):
        if not rotated_test.eof_list[idx].close(target_eof):
            errors.append("rotation-reference-validation: EOF data at index %i is incorrect" % idx)

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))

    # also test PCs? 