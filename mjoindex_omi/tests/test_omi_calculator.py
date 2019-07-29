# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:42:52 2019

@author: ch
"""

import os
import pytest
import numpy as np
import mjoindex_omi.omi_calculator as omi
import mjoindex_omi.io as omiio
import mjoindex_omi.olr_handling as olr

olrDataFilename = (os.path.dirname(__file__)
                   + os.path.sep
                   + "testdata"
                   + os.path.sep
                   + "olr.day.mean.nc")
originalOMIDataDirname = (os.path.dirname(__file__)
                          + os.path.sep
                          + "testdata"
                          + os.path.sep
                          + "OriginalOMI")
eof1Dirname = (originalOMIDataDirname
               + os.path.sep
               + "eof1")
eof2Dirname = (originalOMIDataDirname
               + os.path.sep
               + "eof2")
origOMIPCsFilename = (originalOMIDataDirname
                      + os.path.sep
                      + "omi.1x.txt")

setups = [(True, 0.99, 0.99), (False, 0.999, 0.999)]
@pytest.mark.long
@pytest.mark.parametrize("useQuickTemporalFilter, expectedCorr1, expectedCorr2", setups)
@pytest.mark.skipif(not os.path.isfile(olrDataFilename),
                    reason="OLR data file not available")
@pytest.mark.skipif(not os.path.isdir(eof1Dirname),
                    reason="EOF1 data not available")
@pytest.mark.skipif(not os.path.isdir(eof2Dirname),
                    reason="EOF2 data not available")
@pytest.mark.skipif(not os.path.isfile(origOMIPCsFilename),
                    reason="Original OMI PCs not available for comparison")
def test_calculatePCsFromOLRWithOriginalConditions_Quickfilter(useQuickTemporalFilter, expectedCorr1, expectedCorr2):

    (orig_dates, orig_pc1, orig_pc2) = omiio.loadOriginalPCsFromTxt(origOMIPCsFilename)
    olrData = olr.loadNOAAInterpolatedOLR(olrDataFilename)
    resultFilename= (os.path.dirname(__file__)
                     + os.path.sep
                     + "tempdata"
                     + os.path.sep
                     + "PCs_test_calculatePCsFromOLRWithOriginalConditions.txt")

    (target_pc1, target_pc2) = omi.calculatePCsFromOLRWithOriginalConditions(olrData,
                                     originalOMIDataDirname,
                                     np.datetime64("1979-01-01"),
                                     np.datetime64("2018-08-28"),
                                     resultFilename,
                                     useQuickTemporalFilter = useQuickTemporalFilter)
    errors = []
    corr1 = (np.corrcoef(orig_pc1,target_pc1))[0,1]
    if not corr1 > expectedCorr1:
        errors.append("Correlation of PC1 too low!")

    corr2 = (np.corrcoef(orig_pc2,target_pc2))[0,1]
    if not corr2 > expectedCorr2:
        errors.append("Correlation of PC2 too low!")

    # FIXME: Define more test criteria: e.g., max deviation etc.

    assert not errors, "errors occured:\n{}".format("\n".join(errors))