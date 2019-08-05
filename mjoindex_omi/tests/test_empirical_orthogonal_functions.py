# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:29:17 2019

@author: ch
"""
import math
from pathlib import Path

import numpy as np
import pytest

import mjoindex_omi.empirical_orthogonal_functions as eof

# FIXME: Define paths as fixture at a central point
originalOMIDataDirname = Path(__file__).parent / "testdata" / "OriginalOMI"
eof1Dirname = originalOMIDataDirname / "eof1"
eof2Dirname = originalOMIDataDirname / "eof2"

def test_reshape_to_vector_and_reshape_to_map():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eof1 = np.full((3, 2), 4)
    eof2 = np.full((3, 2), 2)
    target = eof.EOFData(lat, long, eof1, eof2)

    test_vec = np.array([1, 2, 3, 4, 5, 6])
    test_map = target.reshape_to_map(test_vec)
    transform_test_vec = target.reshape_to_vector(test_map)

    errors = []
    if not np.all(test_map == np.array([(1, 2), (3, 4), (5, 6)])):
        errors.append("Result of reshape_to_map does not meet the expectation.")

    if not np.all(test_vec == transform_test_vec):
        errors.append("Result of reshape_to_vector does not meet the expectations (might be a follow-up error)")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_reshape_to_vector_exceptions():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eof1 = np.full((3, 2), 4)
    eof2 = np.full((3, 2), 2)
    target = eof.EOFData(lat, long, eof1, eof2)

    errors = []
    with pytest.raises(AttributeError) as e:
        target.reshape_to_vector(np.array([1, 2, 3]))
    if "2 dimensions" not in str(e.value):
        errors.append("Check 2 dim failed")
    with pytest.raises(AttributeError) as e:
        target.reshape_to_vector(np.array([(1, 2), (3, 4), (5, 6), (7, 8)]))
    if "Length of first dimension" not in str(e.value):
        errors.append("Length check failed")
    with pytest.raises(AttributeError) as e:
        target.reshape_to_vector(np.array([(1, 2, 8), (3, 4, 9), (5, 6, 10)]))
    if "Length of first dimension" not in str(e.value):
        errors.append("Length check failed")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_reshape_to_map_exceptions():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eof1 = np.full((3, 2), 4)
    eof2 = np.full((3, 2), 2)
    target = eof.EOFData(lat, long, eof1, eof2)

    errors = []
    with pytest.raises(AttributeError) as e:
        target.reshape_to_map(np.array([(1, 2), (3, 4), (5, 6)]))
    if "only 1 dimension" not in str(e.value):
        errors.append("Check 1 dim failed")
    with pytest.raises(AttributeError) as e:
        target.reshape_to_map(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    if "lat.size*long.size" not in str(e.value):
        errors.append("Length check failed")
    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_EOFData_basic_properties():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eof1 = np.array([1, 2, 3, 4, 5, 6])
    eof2 = np.array([10, 20, 30, 40, 50, 60])
    target = eof.EOFData(lat, long, eof1, eof2)

    errors = []
    if not np.all(target.lat == lat):
        errors.append("Lat property not correct")
    if not np.all(target.long == long):
        errors.append("Long property not correct")
    if not np.all(target.eof1vector == eof1):
        errors.append("eof1vector property not correct")
    if not np.all(target.eof2vector == eof2):
        errors.append("eof2vector property not correct")
    if not np.all(target.eof1map == np.array([(1, 2), (3, 4), (5, 6)])):
        errors.append("eof1map property not correct")
    if not np.all(target.eof2map == np.array([(10, 20), (30, 40), (50, 60)])):
        errors.append("eof2map property not correct")

    eof1 = np.array([(1, 2), (3, 4), (5, 6)])
    eof2 = np.array([(10, 20), (30, 40), (50, 60)])
    target = eof.EOFData(lat, long, eof1, eof2)

    if not np.all(target.lat == lat):
        errors.append("Lat property not correct")
    if not np.all(target.long == long):
        errors.append("Long property not correct")
    if not np.all(target.eof1vector == np.array([1, 2, 3, 4, 5, 6])):
        errors.append("eof1vector property not correct")
    if not np.all(target.eof2vector == np.array([10, 20, 30, 40, 50, 60])):
        errors.append("eof2vector property not correct")
    if not np.all(target.eof1map == eof1):
        errors.append("eof1map property not correct")
    if not np.all(target.eof2map == eof2):
        errors.append("eof2map property not correct")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_initialization_exceptions():
    lat = np.array([-10., 0., 10., 20])
    long = np.array([0., 5.])

    errors = []

    eof1 = np.zeros((2, 2, 2))
    eof2 = np.ones((2, 2, 2))
    with pytest.raises(AttributeError) as e:
        target = eof.EOFData(lat, long, eof1, eof2)
    if "dimension of 1 or 2" not in str(e.value):
        errors.append("Dimenesion check failed")

    eof1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    eof2 = np.array([(10, 20), (30, 40), (50, 60), (70, 80)])
    with pytest.raises(AttributeError) as e:
        target = eof.EOFData(lat, long, eof1, eof2)
    if "same shape" not in str(e.value):
        errors.append("Check same shape failed")

    eof1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    eof2 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    with pytest.raises(AttributeError) as e:
        target = eof.EOFData(lat, long, eof1, eof2)
    if "lat.size*long.size" not in str(e.value):
        errors.append("size check failed")

    eof1 = np.array([(1, 2), (3, 4), (5, 6), (7, 8)]).T
    eof2 = np.array([(10, 20), (30, 40), (50, 60), (70, 80)]).T
    with pytest.raises(AttributeError) as e:
        target = eof.EOFData(lat, long, eof1, eof2)
    if "correspond to latitude axis" not in str(e.value):
        errors.append("axis check failed")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_save_eofs_to_txt_file_load_eofs_from_txt_file(tmp_path):
    filename = tmp_path / "out.txt"
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eof1 = np.array([1, 2, 3, 4, 5, 6])
    eof2 = np.array([10, 20, 30, 40, 50, 60])
    target = eof.EOFData(lat, long, eof1, eof2)

    target.save_eofs_to_txt_file(filename)

    target_reloaded = eof.load_single_eofs_from_txt_file(filename)

    errors=[]
    if not np.all(target_reloaded.lat == lat):
        errors.append("Latitude grid does not fit")
    if not np.all(target_reloaded.long == long):
        errors.append("Longitude grid does not fit")
    if not np.all(target_reloaded.eof1vector == eof1):
        errors.append("EOF1 does not fit")
    if not np.all(target_reloaded.eof2vector == eof2):
        errors.append("EOF2 does not fit")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available")
def test_save_eofs_to_txt_file_load_eofs_from_txt_filetmp_path_original_data(tmp_path):
    filename = tmp_path / "out.txt"
    target1 = eof.load_original_eofs_for_doy(originalOMIDataDirname, 1)

    target1.save_eofs_to_txt_file(filename)
    target1_reloaded = eof.load_single_eofs_from_txt_file(filename)

    errors = []
    # First some sample tests comparable to test_load_original_eofs_for_doy
    # Tolerance of 1e-7 because float format of new file format truncates at the 7th digit
    if not np.isclose(target1_reloaded.eof1vector[0], 0.00022178496, atol=1e-7):
        errors.append("EOF1 of DOY 1 is incorrect (Position 0)")
    if not np.isclose(target1_reloaded.eof1vector[10], -0.0023467445, atol=1e-7):
        errors.append("EOF1 of DOY 1 is incorrect (Position 10)")
    if not np.isclose(target1_reloaded.eof1vector[-1], 0.013897266, atol=1e-7):
        errors.append("EOF1 of DOY 1 is incorrect (Last position)")
    if not np.isclose(target1_reloaded.eof2vector[0], 0.0042107304, atol=1e-7):
        errors.append("EOF2 of DOY 1 is incorrect (Position 0)")
    if not np.isclose(target1_reloaded.eof2vector[10], 0.015404793, atol=1e-7):
        errors.append("EOF2 of DOY 1 is incorrect (Position 10)")
    if not np.isclose(target1_reloaded.eof2vector[-1], 0.012487547, atol=1e-7):
        errors.append("EOF2 of DOY 1 is incorrect (Last position)")

    # Test complete data
    if not np.all(np.isclose(target1_reloaded.eof1vector,target1.eof1vector,atol=1e-7)):
        errors.append("EOF1 does not fit")
    if not np.all(np.isclose(target1_reloaded.eof2vector, target1.eof2vector, atol=1e-7)):
        errors.append("EOF2 does not fit")
    if not np.all(np.isclose(target1_reloaded.lat,target1.lat)):
        errors.append("Latitude Grid does not fit")
    if not np.all(np.isclose(target1_reloaded.long,target1.long)):
        errors.append("Longitude Grid does not fit")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available")
def test_load_original_eofs_for_doy():
    target1 = eof.load_original_eofs_for_doy(Path(originalOMIDataDirname), 1)
    target10 = eof.load_original_eofs_for_doy(Path(originalOMIDataDirname), 10)
    target366 = eof.load_original_eofs_for_doy(Path(originalOMIDataDirname), 366)

    errors = []
    if not math.isclose(target1.eof1vector[0], 0.00022178496):
        errors.append("EOF1 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target1.eof1vector[10], -0.0023467445):
        errors.append("EOF1 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target1.eof1vector[-1], 0.013897266):
        errors.append("EOF1 of DOY 1 is incorrect (Last position)")
    if not math.isclose(target1.eof2vector[0], 0.0042107304):
        errors.append("EOF2 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target1.eof2vector[10], 0.015404793):
        errors.append("EOF2 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target1.eof2vector[-1], 0.012487547):
        errors.append("EOF2 of DOY 1 is incorrect (Last position)")

    if not math.isclose(target10.eof1vector[0], 0.00016476621):
        errors.append("EOF1 of DOY 10 is incorrect (Position 0)")
    if not math.isclose(target10.eof2vector[0], 0.0044616843):
        errors.append("EOF2 of DOY 10 is incorrect (Position 0)")

    if not math.isclose(target366.eof1vector[-1], 0.013874311):
        errors.append("EOF1 of DOY 366 is incorrect (Last position)")
    if not math.isclose(target366.eof2vector[-1], 0.012473147):
        errors.append("EOF2 of DOY 366 is incorrect (Last position)")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_EOFDataForAllDOYs_basic_properties():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    target = eof.EOFDataForAllDOYs(eofs)

    errors = []
    if not target.eof_list == eofs:
        errors.append("List of EOFData objects incorrect")
    if not np.all(target.lat == lat):
        errors.append("Lat is incorrect")
    if not np.all(target.long == long):
        errors.append("Long is incorrect")
    if not target.eofdata_for_doy(1) == eofs[0]:
        errors.append("Sample EOF data is incorrect")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_EOFDataForAllDOYs_doy_gettfunctions():
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    target = eof.EOFDataForAllDOYs(eofs)

    errors = []
    if not target.eofdata_for_doy(1).eof1vector[0] == 1:
        errors.append("EofData for DOY 1 incorrect")
    if not target.eofdata_for_doy(12).eof1vector[1] == 24:
        errors.append("EofData for DOY 1 incorrect")

    if not target.eof1vector_for_doy(1)[0] == 1:
        errors.append("EOF1Vector for DOY 1 incorrect")
    if not target.eof1vector_for_doy(12)[1] == 24:
        errors.append("EOF1Vector for DOY 1 incorrect")

    if not target.eof2vector_for_doy(1)[0] == 10:
        errors.append("EOF2Vector for DOY 1 incorrect")
    if not target.eof2vector_for_doy(12)[1] == 240:
        errors.append("EOF2Vector for DOY 1 incorrect")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_save_all_eofs_to_dir(tmp_path):
    lat = np.array([-10., 0., 10.])
    long = np.array([0., 5.])
    eofs = []
    for doy in range(1, 367):
        eof1 = np.array([1, 2, 3, 4, 5, 6]) * doy
        eof2 = np.array([10, 20, 30, 40, 50, 60]) * doy
        eofs.append(eof.EOFData(lat, long, eof1, eof2))
    target = eof.EOFDataForAllDOYs(eofs)

    print(tmp_path)

    errors = []
    with pytest.raises(FileNotFoundError) as e:
        target.save_all_eofs_to_dir(tmp_path / "eofs_dir_not_exisiting", create_dir=False)
    if "No such file or directory" not in str(e.value):
        errors.append("Test target should raise error, because directory does not exist.")

    target.save_all_eofs_to_dir(tmp_path / "eofs")
    target_reloaded = eof.load_all_eofs_from_directory(tmp_path / "eofs")
    if not target_reloaded.eof_list == eofs:
        errors.append("List of EOFData objects incorrect")
    if not np.all(target_reloaded.lat == lat):
        errors.append("Lat is incorrect")
    if not np.all(target_reloaded.long == long):
        errors.append("Long is incorrect")
    if not target_reloaded.eofdata_for_doy(1) == eofs[0]:
        errors.append("Sample EOF data is incorrect")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.skipif(not eof1Dirname.is_dir(), reason="EOF1 data not available")
@pytest.mark.skipif(not eof2Dirname.is_dir(), reason="EOF2 data not available")
def test_load_all_original_eofs_from_directory():
    target = eof.load_all_original_eofs_from_directory(originalOMIDataDirname)

    errors = []
    if not math.isclose(target.eofdata_for_doy(1).eof1vector[0], 0.00022178496):
        errors.append("EOF1 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target.eofdata_for_doy(1).eof1vector[10], -0.0023467445):
        errors.append("EOF1 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target.eofdata_for_doy(1).eof1vector[-1], 0.013897266):
        errors.append("EOF1 of DOY 1 is incorrect (Last position)")
    if not math.isclose(target.eofdata_for_doy(1).eof2vector[0], 0.0042107304):
        errors.append("EOF2 of DOY 1 is incorrect (Position 0)")
    if not math.isclose(target.eofdata_for_doy(1).eof2vector[10], 0.015404793):
        errors.append("EOF2 of DOY 1 is incorrect (Position 10)")
    if not math.isclose(target.eofdata_for_doy(1).eof2vector[-1], 0.012487547):
        errors.append("EOF2 of DOY 1 is incorrect (Last position)")

    if not math.isclose(target.eofdata_for_doy(10).eof1vector[0], 0.00016476621):
        errors.append("EOF1 of DOY 10 is incorrect (Position 0)")
    if not math.isclose(target.eofdata_for_doy(10).eof2vector[0], 0.0044616843):
        errors.append("EOF2 of DOY 10 is incorrect (Position 0)")

    if not math.isclose(target.eofdata_for_doy(366).eof1vector[-1], 0.013874311):
        errors.append("EOF1 of DOY 366 is incorrect (Last position)")
    if not math.isclose(target.eofdata_for_doy(366).eof2vector[-1], 0.012473147):
        errors.append("EOF2 of DOY 366 is incorrect (Last position)")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


#test_save_eofs_to_txt_file(pathlib.Path("/tmp/"))
    #target = pc.load_pcs_from_txt_file(filename)
test_EOFDataForAllDOYs_basic_properties()