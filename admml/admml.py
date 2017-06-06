"""ADMM CORE ALGORITHM

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

from copy import copy

import numpy as np
from numpy.linalg import inv, norm

from . import mappers
from . import utils


##########################      ADMM STEPS     ################################


def ADMM(trndata, algoParam):
    """This is the main engine that implements the ADMM steps.

    Parameters
    ----------
    trndata : RDD
        RDD with each entry as numpy array [y, **x**]
    algoParam : dict
        This contains the algorithm parameters for ML/ADMM algorithm.
        The default can be set as: ``algoParam = setDefaultAlgoParams()``

    Returns
    -------
    z : numpy.ndarray
        Weight vector [w,b] (bias term appended to end)
    output : dict
        Dictionary of the algorithm outputs

        * ``W_k``: each row correspond to w vectors at k_th ADMM iteration
        * ``norm_pri``: :math:`\\ell^2`-norm of primal residual
        * ``norm_dual``: :math:`\\ell^2`-norm of dual residual
        * ``rel_norm_pri``: relative norm of primal residual
        * ``rel_norm_dual``: relative norm of dual residual

    """
    def rho_initialize(algoParam):
        """This function initializes the algoParam['RHO'] value.

        There are several strategies, and this is an ongoing research topic.
        Current implementation use a fixed RHO = LAMBDA .
        Additional methods to be included later. For further details see:

        1. Boyd, Stephen, et al. "Distributed optimization and statistical
           learning via the alternating direction method of multipliers."
        2. Goldstein, Tom, et al. "Fast alternating direction optimization
           methods." SIAM Journal on Imaging Sciences 7.3 (2014): 1588-1623.
        3. Ghadimi, Euhanna, et al. "Optimal parameter selection for the
           alternating direction method of multipliers (ADMM): quadratic
           problems." IEEE Transactions on Automatic Control 60.3 (2015):
           644-658.

        Parameters
        ----------
        algoParam : dict
            This contains the algorithm parameters for ML/ADMM algorithm.
            The default can be set as: ``algoParam = setDefaultAlgoParams()``

        Returns
        -------
        algoParam : dict
            The ``algoParam`` dict with correctly initialized rho

        """
        if algoParam['RHO_INITIAL'] == 0:
            algoParam['RHO'] = algoParam['LAMBDA']
        else:
            raise ValueError('Inconsistent RHO Selection!')
        return algoParam

    def rho_adjustment(rho, norm_pri, norm_dual):
        """This function defines the adaptive RHO updates.

        There are several strategies, and this is an ongoing research topic.
        We use the version of the strategy in [Boyd, 2011] pg. 20

        1. Boyd, Stephen, et al. "Distributed optimization and statistical
           learning via the alternating direction method of multipliers."

        Parameters
        ----------
        rho : float
            Current RHO value
        norm_pri : float
            Current normed primal residual
        norm_dual : float
            Current normed dual residual

        Returns
        -------
        rho : float
            Updated RHO

        """
        if norm_pri > 10*norm_dual:
            rho *= 2
        elif norm_dual > 10*norm_pri:
            rho /= 2

        return rho

    def z_update(w, u, algoParam):
        """**z** update steps in ADMM Algorithm. Changes only for different regularizers.

        Parameters
        ----------
        w : numpy.ndarray
            w-vector obtained from the previous w-update step
        u : numpy.ndarray
            u-vector (current state)
        algoParam : dict
            This contains the algorithm parameters for ML/ADMM algorithm.
            The default can be set as: ``algoParam = setDefaultAlgoParams()``

        Returns
        -------
        z : numpy.ndarray
            z-update step

        """
        D = algoParam['D']
        z = np.zeros(D)

        # PROXIMAL OPERATORS
        # ELASTIC-NET REGULARIZER
        if algoParam['REG'] == 'elasticnet':
            for j in range(D):
                if algoParam['RHO']*(w[j]+u[j]) > algoParam['LAMBDA']*algoParam['DELTA'][j]*algoParam['ALPHA']:
                    z[j] = (algoParam['RHO']*(w[j]+u[j])-algoParam['LAMBDA']*algoParam['DELTA'][j]*algoParam['ALPHA'])/(algoParam['LAMBDA']*algoParam['DELTA'][j]*(1-algoParam['ALPHA'])+algoParam['RHO'])
                elif algoParam['RHO']*(w[j]+u[j]) < -algoParam['LAMBDA']*algoParam['DELTA'][j]*algoParam['ALPHA']:
                    z[j] = (algoParam['RHO']*(w[j]+u[j])+algoParam['LAMBDA']*algoParam['DELTA'][j]*algoParam['ALPHA'])/(algoParam['LAMBDA']*algoParam['DELTA'][j]*(1-algoParam['ALPHA'])+algoParam['RHO'])

        # GROUP-REGULARIZER
        elif algoParam['REG'] == 'group':
            G = algoParam['G']
            Gmax = len(G)
            for j in range(Gmax):
                gind = np.where(algoParam['DELTA'] == j)[0]
                wG = w[gind]
                uG = u[gind]
                zG = max(norm(wG+uG)-(algoParam['LAMBDA']*G[j])/algoParam['RHO'], 0.)*((wG+uG)/norm(wG+uG))
                z[gind] = zG
        else:
            raise ValueError('ERROR: Unsupported Regularizer.')

        return z

    def w_update(trndata, z, u, algoParam):
        """**w** update steps in ADMM Algorithm. Changes only for different loss functions

        Parameters
        ----------
        trndata : RDD
            RDD with each entry as numpy array [y,x]
        z : numpy.ndarray
            z-vector from previous state
        u : numpy.ndarray
            u-vector from previous state
        algoParam : dict
            This contains the algorithm parameters for ML/ADMM algorithm.
            The default can be set as: ``algoParam = setDefaultAlgoParams()``

        Returns
        -------
        w : numpy.ndarray
            w-update step

        """
        N = algoParam['N']
        D = algoParam['D']

        if algoParam['LOSS'] == 'leastsq':
            P = trndata.map(lambda x: np.outer(x[1:], x[1:])).reduce(lambda x, y: np.add(x, y))/N + algoParam['RHO']*np.identity(D)
            q = trndata.map(lambda x: np.dot(x[0], x[1:])).reduce(lambda x, y: np.add(x, y))/N + algoParam['RHO']*(z - u)
            return np.dot(inv(P), q)

        # HUBER & PSEUDO-HUBER: initialize with the least-squares solution
        if algoParam['LOSS'] in ['huber', 'pseudo_huber'] and algoParam['PROBLEM'] == 'regression':
            P = trndata.map(lambda x: np.outer(x[1:], x[1:])).reduce(lambda x, y: np.add(x, y))/N + algoParam['RHO']*np.identity(D)
            q = trndata.map(lambda x: np.dot(x[0], x[1:])).reduce(lambda x, y: np.add(x, y))/N + algoParam['RHO']*(z - u)
            w = np.dot(inv(P), q)
            mu_max = algoParam['MU_MAX']   # GRADUALLY OBTAIN FINAL SOLUTION
        else:
            w = np.zeros(D)
            mu_max = None

        # NEWTON UPDATES
        for j in range(algoParam['MAX_INNER_ITER']):
            if algoParam['LOSS'] in ['huber', 'pseudo_huber'] and algoParam['PROBLEM'] == 'regression':
                mu_max = max(mu_max, algoParam['MU'])

            H_v = trndata.map(lambda x: mappers.mapHv(algoParam, x, w, mu_max)).reduce(utils.combineHv)

            H = H_v[0]/N
            v = H_v[1]/N
            P = H + algoParam['RHO']*(np.identity(D))
            Pinv = inv(P)
            q = v + algoParam['RHO']*(w-z+u)

            w_old = copy(w)
            w = w - np.dot(Pinv, q)
            in_residual = norm(w-w_old)/norm(w)

            if mu_max is not None:
                if in_residual < algoParam['INNER_TOL'] and mu_max-algoParam['MU'] <= 1e-6:
                    break
                else:
                    mu_max /= 2.

            elif in_residual < algoParam['INNER_TOL']:
                break

        return w

    # ADMM ALGORITHM
    D = algoParam['D']
    z = u = np.zeros(D)
    W_k = z
    rel_norm_pri = []
    rel_norm_dual = []

    # INITIALIZE RHO
    algoParam = rho_initialize(algoParam)

    for k in range(algoParam['MAX_ITER']):
        z_old = copy(z)
        w = w_update(trndata, z, u, algoParam)
        z = z_update(w, u, algoParam)
        u = u+w-z
        norm_pri = norm(w-z)
        norm_dual = norm(z-z_old)
        W_k = np.vstack((W_k, z))
        rel_norm_pri.append(norm_pri/norm(w))
        rel_norm_dual.append(norm_dual/norm(z))

        if algoParam['RHO_ADAPTIVE_FLAG']:
            algoParam['RHO'] = rho_adjustment(algoParam['RHO'], rel_norm_pri[k], rel_norm_dual[k])

        converged = rel_norm_pri[k] < algoParam['PRIM_TOL'] and rel_norm_dual[k] < algoParam['DUAL_TOL']
        if converged:
            break
        elif algoParam['VERBOSE'] == 1:
            # print the convergence results
            print("Iteration: {0} Current RHO : ({1})  Primal Residual (Rel.): {2}({3}) Dual Residual (Rel.): {4}({5})".format(k+1, algoParam['RHO'], norm_pri, rel_norm_pri[k], norm_dual, rel_norm_dual[k]))

    output = {'W_k': W_k,
              'norm_pri': norm_pri,
              'norm_dual': norm_dual,
              'rel_norm_pri': rel_norm_pri,
              'rel_norm_dual': rel_norm_dual}

    return z, output
