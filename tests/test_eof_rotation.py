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
    pass

def test_create_rotation_matrix():
    # check against a sample rotation matrix
    pass

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
    # normalize a sample vector
    pass

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