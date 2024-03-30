import os
from pathlib import Path

import sys
ISST_DIR = str(Path(os.getcwd()).parent.parent)
sys.path.append(ISST_DIR)

import ISST

import pymc as pm
import numpy as np
import arviz as az


#-------------------------------------------------------------------------------
# Model Dot Test
#-------------------------------------------------------------------------------

def test_model_dot():


    with pm.Model() as model:

        R1 = pm.Beta('R1', mu=0.5, sigma=0.2)
        R2 = pm.Beta('R2', mu=0.9, sigma=0.1)

        baseline = np.array([0.1, 0.5])

        total_impact = pm.Deterministic('Total', pm.math.dot([R1, R2], baseline))

        idata = pm.sample()

        return idata

if __name__ == '__main__':

    results = test_model_dot()
    az.plot_trace(results, combined=True, show=True)
    print(az.summary(results, round_to=2))