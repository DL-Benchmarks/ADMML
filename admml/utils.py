"""This module contains various helper functions used in ADMM

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
from pyspark.mllib.stat import Statistics
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row



def rdd_to_df(line):
    """Converts a numpy array to Row (pyspark.sql.types.Row)

    Parameters
    ----------
    line : numpy.ndarray
        A numpy array where the first element is the label

    Returns
    -------
    pyspark.sql.types.Row
        A ``Row`` with label and features attributes compatible with pyspark.ml libraries

    """
    return Row(label=float(line[0]),features=Vectors.dense(line[1:]))

    
    
def df_to_rdd(line):
    """Converts a pyspark ``Row`` element to numpy array 

    Parameters
    ----------
    line : pyspark.sql.types.Row
        A line dataframe.rdd 

        .. note::

           The dataframe should contain only numbers. Also the method can be invoked on ``dataframe.rdd`` as ``dataFrame`` objects have no attribute 'map'
    
           Returns
    -------
    numpy.ndarray
        A numpy array representation of the data contained in ``line``

    """
    return np.append(line.label,np.array(line.features))


def parse_vector(line):
    """Read a line from a CSV file into a numpy array

    Parameters
    ----------
    line : str
        A line from a CSV file.

        .. note::

           The CSV file should contain only numbers

    Returns
    -------
    numpy.ndarray
        A numpy array representation of the data contained in ``line``

    """
    return np.fromstring(line, sep=',')


def uniform_scale(vector, col_min, col_max):
    """Scale the features uniformly in the range of [0,1]

    Parameters
    ----------
    vector : numpy.ndarray
        A vector (numpy array) representation of a row of the data
    col_min : numpy.ndarray
        Min values over all the columns of data
    col_max : numpy.ndarray
        Max values over all the columns of data

    Returns
    -------
    numpy.ndarray
        Uniformly scaled representation of the vector.

    """
    # Scale only the features
    return np.append(vector[0], (vector[1:]-col_min[1:])/(col_max[1:]-col_min[1:]))


def normalize_scale(vector, col_mean, col_var):
    """Scale the features to a normal distribution

    Parameters
    ----------
    vector : numpy.ndarray
        A vector (numpy array) representation of a row of the data
    col_mean : numpy.ndarray
        Mean over all the columns of data
    col_var : numpy.ndarray
        Variance over all the columns of data

    Returns
    -------
    numpy.ndarray
        Normally scaled representation of the vector.

    """
    # Scale only the features
    return np.append(vector[0], (vector[1:]-col_mean[1:])/np.sqrt(col_var[1:]))


def log1pexpx(x):
    """Transformation :math:`\\mathbf{x} \\mapsto \\log(\\mathbf{1} + \\mathbf{x})`

    Parameters
    ----------
    x : numpy.ndarray
        input

    Returns
    -------
    numpy.ndarray
        :math:`\\log(\\mathbf{1} + \\mathbf{x})`
    """
    y = np.exp(x)
    return np.log1p(y)


def sigmoid(x):
    """Transformation to :math:`\\mathbf{x} \\mapsto sigmoid(\\mathbf{x})`

    Parameters
    ----------
    x : numpy.ndarray
        input

    Returns
    -------
    numpy.ndarray
        :math:`sigmoid(\\mathbf{x})`
    """
    return np.exp(-log1pexpx(-x))


def scale_data(trndata, tstdata, algoParam):
    """Scales the data as specified by the ``algoParam['SCALE']``.

    The ``tstdata`` is scaled in the same range as ``trndata``.

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    tstdata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``
        Here ``algoParam['SCALE']`` is either ``'Uniform'`` or ``'Normalize'``

    Returns
    -------
    trndata : RDD
        the scaled training data
    tstdata : RDD
        the scaled test data

    """
    # SCALE THE DATA
    if algoParam['SCALE'] == 'Uniform':
        # Uniformly Scale in [0,1]
        cStats = Statistics.colStats(trndata)
        c_min = cStats.min()
        c_max = cStats.max()
        trndata = trndata.map(lambda x: uniform_scale(x, c_min, c_max))
        tstdata = tstdata.map(lambda x: uniform_scale(x, c_min, c_max))

    elif algoParam['SCALE'] == 'Normalize':
        cStats = Statistics.colStats(trndata)
        c_mean = cStats.mean()
        c_var = cStats.variance()
        trndata = trndata.map(lambda x: normalize_scale(x, c_mean, c_var))
        tstdata = tstdata.map(lambda x: normalize_scale(x, c_mean, c_var))

    else:
        print('No Scaling selected, returning original data.')

    return trndata, tstdata


def append_one(vector):
    """ Appends a column of ones to the end of the vector. This takes care of bias term.

    Parameters
    ----------
    vector : numpy.ndarray
        input

    Returns
    -------
    numpy.ndarray
        the input vector appended with one -- [input,1]

    """
    return np.append(vector, 1.0)


def combineHv(x1, x2):
    """ Combine/Reduce function for the mappers

    Parameters
    ----------
    x1 : list
        element1
    x2 : list
        element2

    Returns
    -------
    tuple
        ``(x1[0]+x2[0], x1[1]+x2[1])``
    """
    return (x1[0]+x2[0], x1[1]+x2[1])
