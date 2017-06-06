"""ML Algorithms

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
from numpy import sign
from numpy.linalg import norm

from . import admml
from . import utils


def setDefaultAlgoParams():
    """This module creates a default template of the algorithm parameters.

    Returns
    -------
    algoParam : dict
        a dictionary with keys and values (default values are in bold):

        * **PROBLEM: 'binary'**,'regression'
        * **REG: 'elasticnet'**, 'group'
        * **LOSS: 'logistic'** (classification), 'sq_hinge' (classification), 'leastsq' (classification/regression),
          'huber' (regression), 'pseudo_huber' (regression)
        * **LAMBDA: 1.0** , (Regularization parameter)
        * **ALPHA: 0.0**, (Elastic-Net parameter)
        * **MU: 0.1**, ((Pseudo)-Huber Threshold parameter)
        * **SCALE: 'Uniform'**, 'Normalize', 'None'
        * **RHO: 1.0**, (ADMM augmented lagrangian penalty term)
        * **RHO_INITIAL: 0**, 1  (0 = Constant (RHO = LAMBDA) , 1 = Goldstein (pending))
        * **RHO_ADAPTIVE_FLAG: False**, True
        * **MAX_ITER: 100**, (Max. iterations for ADMM updates)
        * **PRIM_TOL: 1e-4**, (Error tolerance of relative primal residual)
        * **DUAL_TOL: 1e-4**, (Error tolerance of relative dual residual)
        * **MAX_INNER_ITER: 10**, (Max. iteration for internal newton updates)
        * **INNER_TOL: 1e-6**, (Relative error tolerance of internal iterations)
        * **VERBOSE: 0** (no print), 1 (print)

    """
    algoParam = {'PROBLEM': 'binary',           # 'binary'(default),'regression', multiclass (pending)
                 'REG': 'elasticnet',           # 'elasticnet'(default), 'group' , 'scad (non-convex pending)'
                 'LOSS': 'logistic',            # Classification :: 'logistic'(default),'sq_hinge','leastsq'
                                                # Regression :: 'leastsq','huber','pseudo_huber'
                 'LAMBDA': 1.0,                 # Regularization parameter 'lambda'
                 'ALPHA': 0.0,                  # Alpha (Elastic net param). Formulation:- (Alpha)*norm(w,1) + (1-Alpha)/2 * norm(w,2)^2
                 'SCALE': 'Uniform',            # 'Uniform'(default), 'Normalize', 'None'
                 'MAX_ITER': 100,               # Max outer iteration
                 'MAX_INNER_ITER': 10,          # Max Inner iteration for Newton Updates:- Logistic , Huber, Pseudo-Huber
                 'RHO': 1.0,                    # RHO FOR ADMM
                 'RHO_INITIAL': 0,              # 0 = Constant (RHO) , 1 = Goldstein
                 'RHO_ADAPTIVE_FLAG': False,    # This Flag sets the RHO Adaptive per step. True= Adaptive RHO, False = Non-Adaptive RHO
                 'PRIM_TOL': 1e-4,              # Relative Tolerance Primal Residual
                 'DUAL_TOL': 1e-4,              # Relative Tolerance Dual Residual
                 'INNER_TOL': 1e-6,             # Inner Newton update Tolerance Level
                 'N': 0,                        # No of Samples. Has to be set.
                 'D': 0,                        # No of Dimension. Has to be set
                 'K': 1,                        # No of classes (binary = 1, Multiclass > 1 (pending))
                 'EIG_VALS_FLAG': 0,            # 0 = SCIPY (exact), 1 = APPROX (pending)
                 'MU': 0.1,                     # Threshold for Huber loss
                 'MU_MAX': 1.0,                 # HUBER START FROM EASIER PROBLEM
                 'VERBOSE': 0                   # 0 = No Prints, 1 = Prints
                 }

    return algoParam


def _ADMM_ML_Algorithm(trndata, algoParam, algorithm):
    """Generic internal function that calls ADMM to solve ML algorithms

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``
    algorithm : str
        the machine learning algorithm to be used. Currently supported ML
        Algorithms:

        * Classification : 'logistic', 'l2svm', 'lssvm'
        * Regression : 'leastsq','pseudo_huber','huber'

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    # APPEND ONES FOR BIAS TERM
    trndata = trndata.map(utils.append_one)   # THIS TAKES CARE OF BIAS TERM.

    # DATA PROPERTIES
    N = trndata.count()
    D = trndata.first().size-1  # FIRST COLUMN IS y
    algoParam['D'] = D
    algoParam['N'] = N

    # LOSS FUNCTION
    if algorithm.lower() == 'logistic':
        algoParam['LOSS'] = 'logistic'
    elif algorithm.lower() == 'l2svm':
        algoParam['LOSS'] = 'sq_hinge'
    elif algorithm.lower() in ['lssvm', 'leastsq']:
        algoParam['LOSS'] = 'leastsq'
    elif algorithm.lower() == 'pseudo_huber':
        algoParam['LOSS'] = 'pseudo_huber'
    elif algorithm.lower() == 'huber':
        algoParam['LOSS'] = 'huber'

    #REGULARIZER
    if algoParam['REG'] == 'elasticnet': # AUTOMATICALLY TAKE CARE OF ELASTIC NET. GROUP REGULARIZER NEEDS DELTA, G TO BE SET BY USER
        algoParam['DELTA'] = np.ones(D-1)
        algoParam['DELTA'] = np.append(algoParam['DELTA'], 0.)  # NO REGULARIZATION IN BIAS SPACE

    w, output = admml.ADMM(trndata, algoParam)
    fval = functionVals(trndata, w, algoParam)

    return w, output, fval


######################          BINARY LOG REG      ###########################
def ADMMLogistic(trndata, algoParam):
    """Solves the (Elastic-Net + Group) Logistic Regression problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'Logistic')


######################      L2-SVM (BINARY)      ##############################
def ADMML2SVM(trndata,algoParam):
    """Solves the (Elastic-Net + Group) L2-SVM classification problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'L2SVM')


######################     LS SVM (BINARY)        #############################
def ADMMLSSVM(trndata,algoParam):
    """Solves the (Elastic-Net + Group) LS-SVM classification problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'LSSVM')


######################      LINEAR REGRESSION     #############################
def ADMMLeastSquares(trndata,algoParam):
    """Solves the (Elastic-Net + Group) Linear Regression problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'leastsq')


######################      PSEUDO-HUBER REGRESSION    ########################
def ADMMPseudoHuber(trndata,algoParam):
    """Solves the (Elastic-Net + Group) Pseudo-Huber Regression problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'pseudo_huber')


######################           HUBER REGRESSION      ########################
def ADMMHuber(trndata,algoParam):
    """Solves the (Elastic-Net + Group) Huber Regression problem

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    fval : numpy.float64
        Function Value at the optimal solution

    """
    return _ADMM_ML_Algorithm(trndata, algoParam, 'huber')


######################        ML FORMULATIONS        ##########################
def functionVals(trndata, w, algoParam):
    """Provides function values for the supported algorithms at specified w-values.

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y,x]
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    fval :numpy.float64
        Function Value at the specified w-value

    """
    loss = 1e6
    reg = 1e6

    def Huber(vector, w, mu):

        y = vector[0]
        x = vector[1:]
        z = y-np.dot(w, x)

        if np.abs(z) <= mu:
            val = 0.5*(z**2)
        else:
            val = mu*np.abs(z) - 0.5*(mu**2)

        return val

    # LOSS
    if algoParam['PROBLEM'] == 'binary':
        if algoParam['LOSS'] == 'logistic':
            loss = (1.0/algoParam['N'])*(trndata.map(lambda x: np.log1p(np.exp(-x[0]*np.dot(x[1:], w)))).reduce(lambda x, y: x+y))
        elif algoParam['LOSS'] == 'sq_hinge':
            loss = (0.5/algoParam['N'])*(trndata.map(lambda x: (max(0,1-x[0]*(np.dot(x[1:], w))))**2).reduce(lambda x, y: x+y))
        elif algoParam['LOSS'] == 'leastsq':
            loss = (0.5/algoParam['N'])*(trndata.map(lambda x: (x[0]-np.dot(x[1:], w))**2).reduce(lambda x, y: x+y))
        else:
            raise ValueError('Inconsistent LOSS function.')

    elif algoParam['PROBLEM'] == 'regression':
        if algoParam['LOSS'] == 'leastsq':
            loss = (0.5/algoParam['N'])*(trndata.map(lambda x: (x[0]-np.dot(x[1:], w))**2).reduce(lambda x, y: x+y))
        elif algoParam['LOSS'] == 'pseudo_huber':
            loss = (1.0/algoParam['N'])*(trndata.map(lambda x: np.sqrt(algoParam['MU']**2 + (x[0]-np.dot(x[1:], w))**2)-algoParam['MU']).reduce(lambda x, y: x+y))
        elif algoParam['LOSS'] == 'huber':
            loss = (1.0/algoParam['N'])*(trndata.map(lambda x: Huber(x, w, algoParam['MU'])).reduce(lambda x, y: x+y))
        else:
            raise ValueError('Inconsistent LOSS function.')

    else:
        raise ValueError('Inconsistent Problem Type.')

    if algoParam['REG'] == 'elasticnet':
        z = w*algoParam['DELTA']
        reg = algoParam['LAMBDA']*(algoParam['ALPHA']*norm(z, ord=1) + (1-algoParam['ALPHA'])*((norm(z, ord=2))**2)/2)
    elif algoParam['REG'] == 'group':
        Gmax = len(algoParam['G'])
        w_sum = 0
        for j in range(Gmax):
            gind = np.where(algoParam['DELTA'] == j)[0]
            w_sum += (algoParam['G'][j])*norm(w[gind], 2)
        reg = algoParam['LAMBDA']*w_sum
    else:
        raise ValueError('Inconsistent Regularizer')

    fval = loss + reg

    return fval


######################        PREDICTION ROUTINE        #######################
def predict(tstdata, w, algoParam):
    """ Predict on Future Test Data

    Parameters
    ----------
    tstdata : RDD
        RDD with each entry as numpy array [y,x]
    w : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    error_rate : numpy.float64
        Misclassification Error Rate (binary classification)
        Mean Square Error (regression)

    """
    tstdata = tstdata.map(utils.append_one)
    N = tstdata.count()

    if algoParam['PROBLEM'] == 'binary':
        error_rate = (1.0/N)*tstdata.map(lambda x: abs(x[0]-sign(np.dot(x[1:], w)))/2).reduce(lambda x, y: x+y)
    elif algoParam['PROBLEM'] == 'regression':
        error_rate = (1.0/N)*tstdata.map(lambda x: (x[0]-np.dot(x[1:], w))**2).reduce(lambda x, y: x+y)
    else:
        raise ValueError('Unsupported Problem type')

    return error_rate
