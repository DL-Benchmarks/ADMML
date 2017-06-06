"""This module contains the mappers to compute the Hessian/gradients in a distributed fashion

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


from math import exp

import numpy as np


def mapHv(algoParam, vector, w, mu):
    """Map function to compute the appropriate Hessian/Gradient based on
    ``algoParam['LOSS']`` and ``algoParam['PROBLEM']``.

    Parameters
    ----------
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``.
        Here ``algoParam['LOSS']`` and ``algoParam['PROBLEM']``  should be
        specified
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.
    mu : float
        The :math:`\\mu` value only for Huber/Pseudo-Huber Loss. Else it's None.

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """
    if algoParam['LOSS'] == 'logistic' and algoParam['PROBLEM'] == 'binary':
        return _mapHv_logistic_binary(vector, w)

    elif algoParam['LOSS'] == 'hinge' and algoParam['PROBLEM'] == 'binary':  # SHALL BE PROVIDED IN NEXT VERSION
        raise ValueError('ERROR: Currently Unsupported.')

    elif algoParam['LOSS'] == 'sq_hinge' and algoParam['PROBLEM'] == 'binary':
        return _mapHv_sq_hinge_binary(vector, w, algoParam['D'])

    elif algoParam['LOSS'] == 'smooth_hinge' and algoParam['PROBLEM'] == 'binary':
        return _mapHv_smooth_hinge_binary(vector, w, algoParam['D'])

    elif algoParam['LOSS'] == 'pseudo_huber' and algoParam['PROBLEM'] == 'regression':
        return _mapHv_pseudo_huber_regression(vector, w, mu)

    elif algoParam['LOSS'] == 'huber' and algoParam['PROBLEM'] == 'regression':
        return _mapHv_huber_regression(vector, w, mu)

    else:
        raise ValueError('ERROR: Unsupported Loss Function.')


def _mapHv_logistic_binary(vector, w):
    """COMPUTE DISTRIBUTED HESSIAN/GRADIENT

    Parameters
    ----------
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """
    y = vector[0]
    x = vector[1:]
    eta = y * (np.dot(x, w))
    p = 1 / (1 + exp(-eta))
    v_i = -y * (1 - p) * x
    H_i = (1 - p) * p * (np.outer(x, x))
    return H_i, v_i


def _mapHv_sq_hinge_binary(vector, w, D):
    """COMPUTE DISTRIBUTED HESSIAN/GRADIENT

    Parameters
    ----------
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.
    D : int
        the dimension of the sample

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """
    y = vector[0]
    x = vector[1:]
    z = y * (np.dot(w, x))

    if z >= 1.0:
        H_i = np.zeros([D, D])
        v_i = np.zeros(D)

    else:
        H_i = np.outer(x, x)
        v_i = (-y * (1.0 - z)) * x

    return H_i, v_i


def _mapHv_smooth_hinge_binary(vector, w, D):
    """COMPUTE DISTRIBUTED HESSIAN/GRADIENT (DO NOT USE!!)

    Parameters
    ----------
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.
    D : int
        the dimension of the sample

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """
    y = vector[0]
    x = vector[1:]
    z = y * (np.dot(w, x))

    if z >= 1.0:
        H_i = np.zeros([D, D])
        v_i = np.zeros(D)

    elif 0.0 < z < 1.0:
        H_i = np.outer(x, x)
        v_i = (-y * (1.0 - z)) * x

    else:
        H_i = np.zeros([D, D])
        v_i = -y * x

    return H_i, v_i


def _mapHv_pseudo_huber_regression(vector, w, mu):
    """COMPUTE DISTRIBUTED HESSIAN/GRADIENT

    Parameters
    ----------
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.
    mu : float
        The :math:`\\mu` value only for Pseudo-Huber Loss.

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """
    y = vector[0]
    x = vector[1:]
    s = (y - np.dot(x, w))
    v_i = -(s / np.sqrt(mu ** 2 + s ** 2)) * x
    H_i = ((mu ** 2) / np.sqrt(mu ** 2 + s ** 2) ** 3) * np.outer(x, x)
    return H_i, v_i


def _mapHv_huber_regression(vector, w, mu):
    """COMPUTE DISTRIBUTED HESSIAN/GRADIENT

    Parameters
    ----------
    vector : numpy.ndarray
        Data sample (a row of data matrix) which contributes to Hessian/Gradient
    w : numpy.ndarray
        weight vector at the current internal newtonian iteration.
    mu : float
        The :math:`\\mu` value only for Huber Loss.

    Returns
    -------
    H_i : numpy.ndarray
        The i-th sample's contribution to the Hessian
    v_i : numpy.ndarray
        The i-th sample's contribution to the gradient

    """

    def trimg(s, mu):
        return max(-mu, min(mu, s))

    def trimH(s, mu):
        return float(abs(s) <= mu)

    y = vector[0]
    x = vector[1:]
    s = (y - np.dot(x, w))
    v_i = -trimg(s, mu) * x
    H_i = trimH(s, mu) * np.outer(x, x)
    return H_i, v_i
