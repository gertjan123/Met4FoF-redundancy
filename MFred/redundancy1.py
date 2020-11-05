# -*- coding: utf-8 -*-
"""
The module :mod:`redundancy1` implements methods for analysing redundant estimates provided by
redundant measurement data from a sensor network.

The main functions included in the file *redundancy1.py* are:

#. :func:`calc_best_estimate`: Calculation of the best estimate for a given set of estimates with associated uncertainty matrix.
#. :func:`calc_lcs`: Calculation of the largest subset of consistent estimates of a measurand.
#. :func:`calc_lcss`: Calculation of the largest subset of sensor values that yield consistent estimates of a measurand linked to the sensor values by a linear system of equations.

The scientific publication giving more information on this topic is:

    G. Kok and P. Harris, "Uncertainty Evaluation for Metrologically Redundant Industrial Sensor Networks,"
    2020 IEEE International Workshop on Metrology for Industry 4.0 & IoT, Roma, Italy, 2020,
    pp. 84-88, doi: `10.1109/MetroInd4.0IoT48571.2020.9138297
    <https://dx.doi.org/10.1109/MetroInd4.0IoT48571.2020.9138297>`_.
"""

import itertools
import numpy as np
from scipy.stats import chi2
from scipy.stats import multivariate_normal as mvn
from scipy.special import comb

def calc_consistent_estimates_no_corr(y_arr2d, uy_arr2d, prob_lim):
    """
    Calculation of consistent estimate for n_sets of estimates y_ij (contained in
    y_arr2d) of a quantity Y, where each set contains n_estims estimates.
    The uncertainties are assumed to be independent and given in uy_arr2d.
    The consistency test is using limit probability limit prob_lim.
    For each set of estimates, the best estimate, uncertainty,
    observed chi-2 value and a flag if the
    provided estimates were consistent given the model are given as output.

    Parameters
    ----------
    y_arr2d:    np.ndarray of size (n, m)
                each row contains m independent estimates of a measurand
    uy_arr2d:   np.ndarray of size (n, m)
                each row contains the standard uncertainty u(y_ij) of y_ij = y_arr2d[i,j]
    prob_lim:   limit probability used in consistency test. Typically 0.95.

    Returns
    -------
    isconsist_arr:  np.ndarray of size(n) ind
    ybest_arr
    uybest_arr
    chi2obs_arr

    """

    if len(y_arr2d.shape) > 1:
        n_sets = y_arr2d.shape[0]
    else:
        n_sets = 1

    n_estims = y_arr2d.shape[-1]  # last dimension is number of estimates
    chi2_lim = chi2.ppf(1 - prob_lim, n_estims - 1)
    uy2inv_arr2d = 1 / np.power(uy_arr2d, 2)
    uy2best_arr = 1 / np.sum(uy2inv_arr2d, -1)
    uybest_arr = np.sqrt(uy2best_arr)

    print(n_sets)
    print(n_estims)
    print(y_arr2d)
    print(uy2inv_arr2d)
    print(y_arr2d * uy2inv_arr2d)

    print(np.sum(y_arr2d * uy2inv_arr2d, -1))
    print(uy2best_arr)
    print(uy_arr2d.shape)

    ybest_arr = np.sum(y_arr2d * uy2inv_arr2d, -1) * uy2best_arr
    chi2obs_arr = np.sum(np.power((y_arr2d - np.broadcast_to(ybest_arr, (n_estims, n_sets))) / uy_arr2d, 2), -1)

    print(np.power((y_arr2d - np.broadcast_to(ybest_arr, (n_sets, n_estims))) / uy_arr2d, 2))
    # print(chi2obs_arr)
    isconsist_arr = (chi2obs_arr <= chi2_lim)
    return isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr


def print_output_single(isconsist, ybest, uybest, chi2obs):
    """
    Function to print the output of a single row of the calculate_best_estimate function.

    Parameters
    ----------
    isconsist
    ybest
    uybest
    chi2obs

    Returns
    -------

    """
    print('\tThe observed chi-2 value is %3.3f.' % chi2obs)
    if not isconsist:
        print("\tThe provided estimates (input) were not consistent.")
    else:
        print("\tThe provided estimates (input) were consistent.")
    print("\tThe best estimate is %3.3f with uncertainty %3.3f.\n" % (ybest, uybest))


def print_output_cbe(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr):
    """
    Function to print the full output of calc_best_estimate.

    Parameters
    ----------
    isconsist_arr
    ybest_arr
    uybest_arr
    chi2obs_arr

    Returns
    -------

    """
    if len(ybest_arr.shape) == 0:
        print_output_single(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)
    else:
        n_sets = ybest_arr.shape[0]
        print('There are %d sets with estimates of the measurand.' % n_sets)
        for i_set in range(n_sets):
            print('The result of set %d is:' % i_set)
            print_output_single(isconsist_arr[i_set], ybest_arr[i_set], uybest_arr[i_set], chi2obs_arr[i_set])



def test_calc_consistent_estimates_no_corr():
    """
    Test function for calc_consistent_estimates_no_corr(), implementing two test cases.

    """
    # case with only one set of estimates
    print('Testing case with single set of estimates.')
    # input
    y_arr = np.array([20.2, 21.3, 20.5])
    uy_arr = np.array([0.5, 0.8, 0.3])
    prob_lim = 0.05
    # function
    isconsist, ybest, uybest, chi2obs = calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)
    # print of output
    print_output_cbe(isconsist, ybest, uybest, chi2obs)

    # case with two sets of estimates
    print('Testing case with two sets of estimates.')
    # input
    y_arr = np.array([[20.2, 21.3, 20.5], [19.5, 19.7, 20.3]])
    uy_arr = np.array([[0.5, 0.8, 0.3], [0.1, 0.2, 0.3]])
    prob_lim = 0.05
    # function
    isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr = calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)
    # print of output
    print_output_cbe(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)


def calc_best_estimate(y_arr, vy_arr2d, problim):
    """Calculate the best estimate for a set of estimates with associated uncertainty matrix,
    and determine if the set of estimates are consistent using a provided limit probability.

    Parameters
    ----------
        y_arr:      np.ndarray of shape (n)
                    vector of estimates of a measurand Y
        vy_arr2d:   np.ndarray of shape (n, n)
                    uncertainty matrix associated with y_arr
        problim:    float
                    probability limit used for assessing the consistency of the estimates. Typically, problim equals 0.95.

    Returns
    -------
        isconsist:  bool
                    indicator whether provided estimates are consistent in view of *problim*
        ybest:      float
                    best estimate of measurand
        uybest:     float
                    uncertainty associated with *ybest*
        chi2obs:    float
                    observed value of chi-squared, used for consistency evaluation
    """
    n_estims = y_arr.shape[-1]
    if n_estims == 1:
        isconsist = True
        ybest = y_arr[0]
        uybest = np.sqrt(vy_arr2d[0, 0])
        chi2obs = 0.0
    else:
        e_arr = np.ones(n_estims)
        vyinve_arr = np.linalg.solve(vy_arr2d, e_arr)
        uy2 = 1 / np.dot(e_arr, vyinve_arr)
        uybest = np.sqrt(uy2)
        ybest = np.dot(vyinve_arr, y_arr) * uy2
        yred_arr = y_arr - ybest
        chi2obs = np.dot(yred_arr, np.linalg.solve(vy_arr2d, yred_arr))
        chi2lim = chi2.ppf(problim, n_estims - 1)
        isconsist = (chi2obs <= chi2lim)
    return isconsist, ybest, uybest, chi2obs



# test function for calc_best_estimate
def test_calc_best_estimate():
    print('\n-------------------------------------------------------------------\n')
    print('TESTING FUNCTION calc_best_estimate()')
    # Test case 0
    print('TEST CASE 0')
    n_reps = 1000
    y_arr = np.array([20.2, 20.5, 20.8])
    vy_arr2d = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 4]])
    problim = 0.95
    isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
    print_output_cbe(isconsist, ybest, uybest, chi2obs)

    # Test case 1: check limit probability limprob
    print('TEST CASE 1')
    n_reps = 10000
    print('Repeating the procedure %d times in order to check the acceptance statistics.' % n_reps)
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    problim = 0.95
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
        if isconsist == True:
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print('Repeating the procedure %d times, data generated by the assumed model is accepted with probability %4.4f'
          ', whereas %4.4f is expected.\n' % (n_reps, frackeep, problim))


# function that returns a list with the values corresponding to combination indcomb
def get_combination(val_arr, n_keep, indcomb):
    subsets = itertools.combinations(val_arr, n_keep)
    i_subset = -1
    for subset in subsets:
        i_subset += 1
        if i_subset == indcomb:
            return np.array(list(subset))
    return -1  # error, index indcomb is probably out of range or not an integer


def calc_lcs(y_arr, vy_arr2d, problim):
    isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
    n_estims = len(y_arr)
    estim_arr = np.arange(n_estims)
    n_remove = 0
    if isconsist == True:  # set the other return variables
        n_sols = 1
        indkeep = estim_arr
    while (isconsist == False):
        n_remove += 1
        subsets = itertools.combinations(estim_arr, n_estims - n_remove)
        n_subsets = comb(n_estims, n_remove, exact=True)
        isconsist_arr = np.full(n_subsets, np.nan)
        ybest_arr = np.full(n_subsets, np.nan)
        uybest_arr = np.full(n_subsets, np.nan)
        chi2obs_arr = np.full(n_subsets, np.nan)
        i_subset = -1
        for subset in subsets:
            i_subset += 1
            sublist = list(subset)
            yred_arr = y_arr[sublist]
            vyred_arr2d = vy_arr2d[np.ix_(sublist, sublist)]
            isconsist_arr[i_subset], ybest_arr[i_subset], uybest_arr[i_subset], chi2obs_arr[i_subset] = \
                calc_best_estimate(yred_arr, vyred_arr2d, problim)
        # Find smallest chi2obs value amongst all subsets. If multiple possibilities exist, return them all
        indmin = np.argmin(chi2obs_arr)
        if isconsist_arr[indmin] == True:
            # consistent solution found (otherwise isconsist remains false and the while loop continues)
            isconsist = True
            chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
            indmin = np.where(chi2obs_arr == chi2obs)[0]  # list with all indices with minimum chi2obs value
            n_sols = len(indmin)
            if n_sols == 1:
                ybest = ybest_arr[indmin[0]]
                uybest = uybest_arr[indmin[0]]
                indkeep = get_combination(estim_arr, n_estims - n_remove, indmin)  # indices of kept estimates
            else:  # multiple solutions exist, the return types become arrays
                ybest = np.full(n_sols, np.nan)
                uybest = np.full(n_sols, np.nan)
                indkeep = np.full((n_sols, n_estims - n_remove), np.nan)
                for i_sol in range(n_sols):
                    ybest[i_sol] = ybest_arr[indmin[i_sol]]
                    uybest[i_sol] = uybest_arr[indmin[i_sol]]
                    indkeep[i_sol] = get_combination(estim_arr, n_estims - n_remove, indmin[i_sol])
    return n_sols, ybest, uybest, chi2obs, indkeep


# test function for calc_lcs()
def test_calc_lcs():
    print('\n-------------------------------------------------------------------\n')
    print('TESTING FUNCTION calc_lcs()')
    # Test case 0:
    print('TEST CASE 0')
    # input
    y_arr = np.array([20, 20.6, 20.5, 19.3])
    vy_arr2d = np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 1:
    print('TEST CASE 1')
    # input
    y_arr = np.array([20, 23.6, 20.5, 19.3])
    vy_arr2d = np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 2 with two optimal solutions
    print('TEST CASE 2')
    # input
    y_arr = np.array([10, 11, 20, 21])
    vy_arr2d = 5 * np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 3: check limit probability limprob
    print('TEST CASE 3')
    n_reps = 10000
    print('Repeating the procedure %d times in order to check the acceptance statistics.' % n_reps)
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    problim = 0.95
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
        if indkeep.shape[-1] == len(y_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print('Repeating the procedure %d times, data generated by the assumed model is accepted with probability %4.4f'
          ', whereas %4.4f is expected.' % (n_reps, frackeep, problim))


def print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr):
    n_estims = len(y_arr)
    n_keep = indkeep.shape[-1]  # number of retained estimates in the best solution(s)
    if n_sols == 1:
        print('calc_lcs found a unique solution with chi2obs = %4.4f using %d of the provided %d estimates.'
              % (chi2obs, n_keep, n_estims))
        print('\ty = %4.4f, u(y) = %4.4f' % (ybest, uybest))
        print('\tIndices and values of retained provided estimates:', end=' ')
        for ind in indkeep[:-1]:
            indint = int(ind)
            print('y[%d]= %2.2f' % (indint, y_arr[indint]), end=', ')
        indint = int(indkeep[-1])
        print('y[%d]= %2.2f.\n' % (indint, y_arr[indint]))
    else:
        print('calc_lcs found %d equally good solutions with chi2obs = %4.4f using %d of the provided %d estimates.'
              % (n_sols, chi2obs, n_keep, n_estims))
        for i_sol in range(n_sols):
            print('\tSolution %d is:' % i_sol)
            print('\ty = %4.4f, u(y) = %4.4f' % (ybest[i_sol], uybest[i_sol]))
            print('\tIndices and values of retained provided estimates:', end=' ')
            for ind in indkeep[i_sol][:-1]:
                indint = int(ind)
                print('y[%d]= %2.2f' % (indint, y_arr[indint]), end=', ')
            indint = int(indkeep[i_sol][-1])
            print('y[%d]= %2.2f.\n' % (indint, y_arr[indint]))
    return



# Function that returns the index of a row of A that can be written as a linear combination of the others.
# This row does not contribute any new information to the system.
def reduce_a(a_arr2d, epszero):
    if a_arr2d.shape[0] <= np.linalg.matrix_rank(a_arr2d):
        print('ERROR: A cannot be reduced!')
        return 1/0
    # Remove one row from A that is a linear combination of the other rows.
    # Find a solution of A' * b = 0.
    u, s, vh = np.linalg(np.transpose(a_arr2d))
    # singVals = diag(S)%;
    b = vh[-1, :]
    indrem = np.where(abs(b) > epszero)[0] # remove a row corresponding to a non-zero entry in b.
    if len(indrem) == 0:
        print('ERROR: b is zero vector!')
        1/0 # Create error
    indrem = indrem[-1]  # return the last row that can be taken out
    # print('ReduceA: Identified row %d to be removed from a and A.\n', indRem);
    return indrem

# Reduced the system if matrix Vx is not of full rank.
# This might be ambiguous, as constant sensor values or offsets have to be estimated and are not known.
def reduce_vx(x_arr, vx_arr2d, a_arr, a_arr2d, epszero):
    if vx_arr2d.shape[0] <= np.linalg.matrix_rank(vx_arr2d, epszero):
        print('Vx cannot be reduced!')
        return
    # Remove one sensor from Vx, A and x that is a linear combination of the other sensors.
    # Find a solution of Vx * b = 0. This
    u, s, vh = np.linalg.svd(vx_arr2d)
    b = vh[-1, :] # bottom row of vh is orthogonal to Vx
    if abs(np.dot(b, x_arr))> epszero :
        print('ERROR: Sensors in x should be linearly dependent with b^T * x = 0, but this is not the case!')
        1/0 # Raise error here
    indrem = np.where(abs(b) > epszero)[0] # remove a sensor corresponding to a non-zero entry in b.
    if len(indrem) == 0:
        print('ERROR: b is the zero vector!')
        1/0 # Raise error here
    indrem = indrem[-1] # take out the last sensor
    indsenskeep = np.concatenate(np.arange(indrem), np.arange(indrem, vx_arr2d.shape[0]))
    vxred_arr2d = vx_arr2d[indsenskeep, indsenskeep]
    xred_arr = x_arr(indsenskeep)
    # Update A by removing the sensor and updating the system of equations

    ared_arr2d = a_arr2d - a_arr2d[:, indrem]/ b[indrem] * np.transpose(b)
    if max(abs(ared_arr2d[:, indrem])) > epszero :
        print(ared_arr2d)
        print('ERROR: Column ' + str(indrem) + ' should be zero by now!')
        1/ 0 # create error
    ared_arr2d = a_arr2d[:, indsenskeep] # remove the zero column from A
    ared_arr = a_arr + np.dot(a_arr2d - ared_arr2d, x_arr) # adapt vector a_arr such that the vector of estimates y = a + A*x remains the same
    return xred_arr, vxred_arr2d, ared_arr, ared_arr2d



def calc_best_est_lin_sys(a_arr, a_arr2d, x_arr, vx_arr2d, problim):
    epszero = 1e-10  # some small constant used for some checks

    # The main procedure only works when vy_arr2d has full rank. Therefore first a_arr, a_arr2d and vx_arr2d need to be
    # reduced such that vy_arr2d will have full rank.
    xred_arr = x_arr
    vxred_arr2d = vx_arr2d
    ared_arr = a_arr
    ared_arr2d = a_arr2d

  #  print('cbels: x_arr = ', x_arr)
 #   print('cbels: a_arr2d = ', a_arr2d)

    # Reduce the system if the covariance matrix vx_arr2d is rank deficient.
    while np.linalg.matrix_rank(vxred_arr2d) < vxred_arr2d.shape[0]:
        print('Reducing Vx. No of rows = ', vxred_arr2d.shape[0], ', rank = ', np.linalg.matrix_rank(vxred_arr2d))
        [xred_arr, vxred_arr2d, ared_arr, ared_arr2d] = reduce_vx(xred_arr, vxred_arr2d, ared_arr, ared_arr2d, epszero)

    # Reduce the system if a_arr2d has more rows than its rank.
    while ared_arr2d.shape[0] > np.linalg.matrix_rank(ared_arr2d):
        print('Reducing A. No of rows = ', ared_arr2d.shape[0], ', rank = ', np.linalg.matrix_rank(ared_arr2d))
        ind_rem = reduce_a(ared_arr, ared_arr2d, epszero)
        n_rows = ared_arr2d.shape[0]
        indrowskeep = np.concatenate((np.arange(0, ind_rem - 1), np.arange(ind_rem + 1, n_rows)))
        ared_arr = ared_arr[indrowskeep]
        ared_arr2d = ared_arr2d[indrowskeep,]

    # calculate y vector and Vy matrix
    y_arr = ared_arr + np.dot(ared_arr2d, xred_arr)
    vy_arr2d = np.matmul(np.matmul(ared_arr2d, vxred_arr2d), np.transpose(ared_arr2d))

    # try to calculate a consistent solution with these y and Vy.
    isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
    # print('y_arr = ', y_arr)
    # print('chi2obs = ', chi2obs)
    return isconsist, ybest, uybest, chi2obs


# function to calculate lcss
def calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim):
    epszero = 1e-7 # epsilon for rank check
    eps_chi2 = 1e-7 # epsilon for chi2 equivalence check

    isconsist, ybest, uybest, chi2obs = calc_best_est_lin_sys(a_arr, a_arr2d, x_arr, vx_arr2d, problim)
    n_sens = len(x_arr)
    sens_arr = np.arange(n_sens)
    n_remove = 0
    if isconsist == True:  # set the other return variables
        n_sols = 1
        indkeep = sens_arr

    # no consistent solution, remove sensors 1 by 1, 2 by 2, etc.
    while (isconsist == False):
        n_remove += 1
        subsets = itertools.combinations(sens_arr, n_sens - n_remove)
        n_subsets = comb(n_sens, n_remove, exact=True)
        isconsist_arr = np.full(n_subsets, np.nan)
        ybest_arr = np.full(n_subsets, np.nan)
        uybest_arr = np.full(n_subsets, np.nan)
        chi2obs_arr = np.full(n_subsets, np.nan)
        i_subset = -1
        for subset in subsets:
            i_subset += 1
            sublistsenskeep = list(subset)
            xred_arr = x_arr[sublistsenskeep]
            vxred_arr2d = vx_arr2d[:, sublistsenskeep]
            vxred_arr2d = vxred_arr2d[sublistsenskeep, :]
            boolremove_arr = np.full(n_sens, True)
            boolremove_arr[sublistsenskeep] = False
            if np.linalg.matrix_rank(np.concatenate((a_arr2d[:, boolremove_arr], np.ones((a_arr2d.shape[0], 1))), axis=1), epszero) == \
                                                    np.linalg.matrix_rank(a_arr2d[:, boolremove_arr], epszero):
                # there is no vector c such that c' * ones = 1 and c' * ai = 0 at the same time.
                # Thus this combination of sensors cannot be removed as a group from the matrix A.
                isconsist_arr[i_subset] = False
                continue  # continue with next subset

            ared_arr2d = np.concatenate(
                (a_arr2d[:, boolremove_arr], a_arr2d[:, sublistsenskeep]), axis=1)  # move the columns corresponding to sensors to be taken out to the front
            q,r = np.linalg.qr(ared_arr2d)
            q1 = q[:, n_remove:]  # these (n_sens-n_remove) columns are orthogonal to the first n_remove columns of ared_arr2d
            # print('n_sens = ', n_sens)
            # print('n_remove = ', n_remove)
            # print(q1)
            s = np.sum(q1, axis=0)  # column sums might be zero which is a problem for normalization to unit sum
            indzero = np.where(np.abs(s) < epszero)[0]
            if len(indzero) > 0:
                indnonzero = np.full(n_sens-n_remove, True) # all True array
                indnonzero[indzero] = False # positions that are zero are false
                indnonzero = np.where(indnonzero) # conversion to indices instead of boolean array
                if len(indnonzero) == 0:
                    print("ERROR: All columns have zero sum!")
                b = q1[:, indnonzero[0]] # b is column vector with no zero column sum
                for i_zero in range(len(indzero)):
                    q1[:, indzero[i_zero]] = q1[:, indzero[i_zero]] + b  # add b to prevent zero column sum
            q1 = q1 / np.sum(q1, axis=0) # unit column sums, in order not to introduce a bias in the estimate of the measurand
            # print('normalized q1 = ', q1)
            ared_arr2d = np.matmul(np.transpose(q1), ared_arr2d)
            ared_arr   = np.matmul(np.transpose(q1), a_arr)
            # The columns of matrix ared_arr2d are still in the wrong order compared to the order of the sensors
            ared2_arr2d = np.full_like(ared_arr2d, np.nan)
            ared2_arr2d[:, boolremove_arr] = ared_arr2d[:, :n_remove] # columns 0, 1, ..., (n_remove-1)
            ared2_arr2d[:, np.invert(boolremove_arr)] = ared_arr2d[:, n_remove:] # columns n_remove, ..., (n_sens-1)
            ared_arr2d = ared2_arr2d
            # print('ared_arr2d = ', ared_arr2d)
            if np.linalg.norm(ared_arr2d[:, boolremove_arr]) > epszero:
                print('ERROR: These columns of A should be zero by now!')
                1/0  # just something to raise an error here
            ared_arr2d = ared_arr2d[:, sublistsenskeep] # np.invert(boolremove_arr)] # reduce the matrix A by removing the appropriate columns of A, which are zero anyway.

            isconsist_arr[i_subset], ybest_arr[i_subset], uybest_arr[i_subset], chi2obs_arr[i_subset] = \
                calc_best_est_lin_sys(ared_arr, ared_arr2d, xred_arr, vxred_arr2d, problim)

        # print('chi2obs_arr = ', chi2obs_arr)
        # After analyzing all subset,
        # find the smallest chi2obs value amongst all subsets. If multiple possibilities exist, return them all
        indmin = np.argmin(chi2obs_arr)
        if isconsist_arr[indmin] == True:
            # consistent solution found (otherwise isconsist remains false and the while loop continues)
            isconsist = True
            chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
            indmin = np.where(np.abs(chi2obs_arr - chi2obs) < eps_chi2)[0]  # list with all indices with minimum chi2obs value
            n_sols = len(indmin)
            if n_sols == 1:
                ybest = ybest_arr[indmin[0]]
                uybest = uybest_arr[indmin[0]]
                indkeep = get_combination(sens_arr, n_sens - n_remove, indmin)  # indices of kept estimates
            else:  # multiple solutions exist, the return types become arrays
                ybest = np.full(n_sols, np.nan)
                uybest = np.full(n_sols, np.nan)
                indkeep = np.full((n_sols, n_sens - n_remove), np.nan)
                for i_sol in range(n_sols):
                    ybest[i_sol] = ybest_arr[indmin[i_sol]]
                    uybest[i_sol] = uybest_arr[indmin[i_sol]]
                    indkeep[i_sol] = get_combination(sens_arr, n_sens - n_remove, indmin[i_sol])
    return n_sols, ybest, uybest, chi2obs, indkeep


def print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim):
    print('INPUT of lcss function call:')
    print('Vector a of linear system: a_arr = ', a_arr)
    print('Matrix A of linear system: a_arr2d = ', a_arr2d)
    print('Vector x with sensor values: x_arr = ', x_arr)
    print('Covariance matrix Vx of sensor values: vx_arr2d = ', vx_arr2d)
    print('Limit probability for chi-squared test: p = ', problim)


def print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d):
    n_sensors = len(x_arr)
    n_eq = a_arr2d.shape[0]
    n_keep = indkeep.shape[-1]  # number of retained estimates in the best solution(s)
    print('Provided number of sensors (or sensor values) was %d and number of equations was %d.' % (n_sensors, n_eq))
    if n_sols == 1:
        print('calc_lcss found a unique solution with chi2obs = %4.4f using %d of the provided %d sensor values.'
              % (chi2obs, n_keep, n_sensors))
        print('\ty = %4.4f, u(y) = %4.4f' % (ybest, uybest))
        print('\tIndices and values of retained provided sensor values:', end=' ')
        for ind in indkeep[:-1]:
            indint = int(ind)
            print('x[%d]= %2.2f' % (indint, x_arr[indint]), end=', ')
        indint = int(indkeep[-1])
        print('x[%d]= %2.2f.\n' % (indint, x_arr[indint]))
    else:
        print('calc_lcss found %d equally good solutions with chi2obs = %4.4f using %d of the provided %d sensor values.'
              % (n_sols, chi2obs, n_keep, n_eq))
        for i_sol in range(n_sols):
            print('\tSolution %d is:' % i_sol)
            print('\ty = %4.4f, u(y) = %4.4f' % (ybest[i_sol], uybest[i_sol]))
            print('\tIndices and values of retained provided sensor values:', end=' ')
            for ind in indkeep[i_sol][:-1]:
                indint = int(ind)
                print('x[%d]= %2.2f' % (indint, x_arr[indint]), end=', ')
            indint = int(indkeep[i_sol][-1])
            print('x[%d]= %2.2f.\n' % (indint, x_arr[indint]))
    return


# test function for calc_lcsc()
def test_calc_lcss():
    print('\n-------------------------------------------------------------------\n')
    print('TESTING FUNCTION calc_lcss()\n')
    # Test case 0:
    print('TEST CASE 0')
    print('Test case with A = identity matrix and a = zero, i.e. same as lcs.')
    # input
    x_arr = np.array([22.3, 20.6, 25.5, 19.3])
    vx_arr2d = np.identity(4) + np.ones((4, 4))
    a_arr = np.zeros(4)
    a_arr2d = np.identity(4)
    problim = 0.95
    # print input
    print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 1:
    print('TEST CASE 1')
    print('lcss which reduces to lcs after transformation of the linear system.')
    print('Input data is the same as in the last case and the results should be the same as well.')
    # input
    # x_arr = (same as above) np.array([20, 23.6, 20.5, 19.3])
    # vx_arr2d = (same as above) np.identity(4) + np.ones((4, 4))

    a_arr2d = np.array([[1, 2, 3, 4], [2, -5, 4, 1], [2, 9, 1, 0], [3, 5, -2, 4]])
   # a_arr2d = np.identity(4) + np.ones((4,4))
    s = np.sum(a_arr2d,1)
    s.shape = (s.shape[0], 1)  # set the second dimension to 1
    a_arr2d = a_arr2d/s  # make all row sums equal to 1

    # Manipulate input to create a non trivial vector a_arr
    dx_arr = np.array([1, 2, 3, 4])
    x_arr = x_arr - dx_arr
    a_arr = a_arr + np.matmul(a_arr2d, dx_arr)

    #problim = 0.95
    # print input
    print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 2 with two optimal solutions
    print('TEST CASE 2')
    # input
    x_arr = np.array([10, 11, 20, 21])
    vx_arr2d = np.identity(4) + np.ones((4, 4))

    # Manipulate input to create a non trivial vector a_arr
    a_arr = np.zeros(4)
    dx_arr = np.array([1, 20, 3, -44])
    x_arr = x_arr - dx_arr
    a_arr = a_arr + np.matmul(a_arr2d, dx_arr)

    # problim = 0.95
    # print input
    print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 3: check limit probability limprob
    print('TEST CASE 3')
    n_reps = 10000
    print('Repeating the procedure %d times in order to check the acceptance statistics.' % n_reps)
    xmean = 20.0
    vx_arr2d = np.random.rand(4, 4)
    vx_arr2d = vx_arr2d.transpose() @ vx_arr2d
    problim = 0.95
    n_casekeep = 0
    for i_rep in range(n_reps):
        x_arr = xmean + mvn.rvs(mean=None, cov=vx_arr2d)
        # Add an additional conversion to work with non-trivial vector a_arr
        dx_arr = np.random.standard_normal(4)
        x_arr = x_arr - dx_arr
        a_arr = np.matmul(a_arr2d, dx_arr)
        n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
        if indkeep.shape[-1] == len(x_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print('Repeating the procedure %d times, data generated by the assumed model is accepted with probability %4.4f'
           ', whereas %4.4f is expected.' % (n_reps, frackeep, problim))
