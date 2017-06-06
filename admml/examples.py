"""Basic Examples

Copyright (c) 2017 The ADMML Authors.

All rights reserved. Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at :
   
   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""


import numpy as np

from . import utils
from . import mlalgs
from . import admml


def example(trndata, tstdata, example=1):
    """This module shows three basic examples to use the tool.

    Parameters
    ----------
    trndata : RDD
        This is the training data on which the models are trained.
        This is an RDD of numbers only created by reading a comma separated file through ``sc.textFile()``.
        The 1st field is y-labels
        i.e. ``trndata = sc.textFile('<URI>\\trndata.csv')``
    tstdata : RDD
        This is the test data on which the models are scored.
        This is an RDD of numbers only created by reading a comma separated file through ``sc.textFile()``.
        The 1st field is y-labels.
        i.e. ``trndata = sc.textFile('<URI>\\tstdata.csv')``
    example : int {1,2,3}
        1 = L2-Linear Regression, 2 = L2-Logistic Regression, 3 = Group - Linear Regression

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)

    """
    trndata = trndata.map(utils.parse_vector)      #(NEED DATA IN VSM) ARRAY of RDDS
    tstdata = tstdata.map(utils.parse_vector)

    if example == 1: #USE PREBUILT FUNCTIONS
        algoParam = mlalgs.setDefaultAlgoParams()
        algoParam['PROBLEM'] = 'regression'
        algoParam['VERBOSE'] = 1
        w, output, fval = mlalgs.ADMMLeastSquares(trndata, algoParam)

        test_err = mlalgs.predict(tstdata, w, algoParam)
        train_err = mlalgs.predict(trndata, w, algoParam)

        print('Test Error = {0}'.format(test_err))
        print('Train Error = {0}'.format(train_err))
        print('Minimized Function Value = {0}'.format(fval))

    elif example == 2:
        algoParam = mlalgs.setDefaultAlgoParams()
        algoParam['PROBLEM'] = 'binary'
        algoParam['VERBOSE'] = 1
        w, output, fval = mlalgs.ADMMLogistic(trndata, algoParam)

        test_err = mlalgs.predict(tstdata, w, algoParam)
        train_err = mlalgs.predict(trndata, w, algoParam)

        print('Test Error = {0}'.format(test_err))
        print('Train Error = {0}'.format(train_err))
        print('Minimized Function Value = {0}'.format(fval))

    elif example == 3:   # PLUG AND PLAY

        algoParam = mlalgs.setDefaultAlgoParams()
        trn = trndata.map(utils.append_one)
        N = trn.count()
        D = trn.first().size-1  # Data format is [y,X,1]
        algoParam['D'] = D
        algoParam['N'] = N
        algoParam['PROBLEM'] = 'regression'

        # GROUP REGULARIZER
        algoParam['REG'] = 'group'
        algoParam['VERBOSE'] = 0
        # CHANGE DELTA and G for different data. CURRENT SETTING IS FOR DATA LINK: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer (ONLY FOR ILLUSTRATION)
        algoParam['G'] = np.array([1.0, 1.0, 1.0, 0]/np.sqrt(3))
        algoParam['DELTA'] = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])

        #LOGISTIC LOSS
        algoParam['LOSS'] = 'leastsq'
        algoParam['LAMBDA'] = 0.1     # REGULARIZATION PARAMETER
        algoParam['RHO_ADAPTIVE_FLAG'] = False

        w, output = admml.ADMM(trn, algoParam)
        train_err = mlalgs.predict(trndata, w, algoParam)

        print('Train Error = {0}'.format(train_err))

    return w
