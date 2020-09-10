# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:41:49 2020

@author: KokG
"""

import numpy as np
from scipy.stats import chi2
import itertools
from scipy.special import comb
from scipy.stats import multivariate_normal as mvn


# Calculation of consistent estimate for n_sets of estimates y_ij (contained in
# y_arr2d) of a quantity Y, where each set contains n_estims estimates.
# The uncertainties are assumed to beindependent and given in uy_arr2d.
# The consistency test is using limit probability limit prob_lim.
# Fora each set of estimates, the best estimate, uncertainty, 
# observed chi-2 value and a flag if the 
# provided estimates were consistent given the model are given as output.
def calc_consistent_estimates_no_corr(y_arr2d, uy_arr2d, prob_lim):
    if len(y_arr2d.shape) > 1:
        n_sets = y_arr2d.shape[0]
    else:
        n_sets = 1

    n_estims = y_arr2d.shape[-1]  # last dimension is number of estimates
    chi2_lim = chi2.ppf(1 - prob_lim, n_estims - 1);
    uy2inv_arr2d = 1 / np.power(uy_arr2d, 2)
    uy2best_arr = 1 / np.sum(uy2inv_arr2d, -1);
    uybest_arr = np.sqrt(uy2best_arr)

    print(n_sets)
    print(n_estims)
    print(y_arr2d)
    print(uy2inv_arr2d)
    print(y_arr2d * uy2inv_arr2d)

    print(np.sum(y_arr2d * uy2inv_arr2d, -1))
    print(uy2best_arr)
    print(uy_arr2d.shape)

    ybest_arr = np.sum(y_arr2d * uy2inv_arr2d, -1) * uy2best_arr;
    chi2obs_arr = np.sum(np.power((y_arr2d - np.broadcast_to(ybest_arr, (n_estims, n_sets))) / uy_arr2d, 2), -1);

    print(np.power((y_arr2d - np.broadcast_to(ybest_arr, (n_sets, n_estims))) / uy_arr2d, 2))
    print(chi2obs_arr)
    isconsist_arr = (chi2obs_arr <= chi2_lim);
    return isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr


def print_output_single(isconsist, ybest, uybest, chi2obs):
    print('\tThe observed chi-2 value is %3.3f.' % chi2obs)
    if not isconsist:
        print("\tThe provided estimates (input) were not consistent.")
    else:
        print("\tThe provided estimates (input) were consistent.")
    print("\tThe best estimate is %3.3f with uncertainty %3.3f.\n" % (ybest, uybest))


# Function to print output of calc_best_estimate
def print_output_cbe(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr):
    if len(ybest_arr.shape) == 0:
        print_output_single(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)
    else:
        n_sets = ybest_arr.shape[0]
        print('There are %d sets with estimates of the measurand.' % n_sets)
        for i_set in range(n_sets):
            print('The result of set %d is:' % i_set)
            print_output_single(isconsist_arr[i_set], ybest_arr[i_set], uybest_arr[i_set], chi2obs_arr[i_set])


# Test function for calc_consistent_estimates_no_corr().
def test_calc_consistent_estimates_no_corr():
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

    # case with two set2 of estimates
    print('Testing case with single set of estimates.')
    # input
    y_arr = np.array([[20.2, 21.3, 20.5], [19.5, 19.7, 20.3]])
    uy_arr = np.array([[0.5, 0.8, 0.3], [0.1, 0.2, 0.3]])
    prob_lim = 0.05
    # function
    isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr = calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)
    # print of output
    print_output_cbe(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)


# Calculate best estimate for 1 set of estimates y_arr with covariance matrix
# vy_arr2d. 
# The consistency test is using limit probability limit prob_lim.
def calc_best_estimate(y_arr, vy_arr2d, problim):
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
    print('TESTING FUNCTION calc_best_estimate()')
    # Test case 0
    print('TEST CASE 0')
    n_reps = 1000
    y_arr = np.array([20.2, 20.5, 20.8])
    vy_arr2d = np.array([[2, 1, 1], [1,3, 1], [1, 1, 4]])
    problim = 0.95
    isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
    print_output_cbe(isconsist, ybest, uybest, chi2obs)

    # Test case 1: check limit probability limprob
    print('TEST CASE 1')
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    problim = 0.95
    n_reps = 10000
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
        if isconsist == True:
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print('Repeating the procedure %d times, data from a consistent model is accepted with probability %4.4f'
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
            ybest_arr, uybest_arr, chi2obs_arr
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
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    problim = 0.95
    n_reps = 10000
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
        if indkeep.shape[-1] == len(y_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print('Repeating the procedure %d times, data from a consistent model is accepted with probability %4.4f'
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


# function to calculate lcss
def calc_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, prob_lim):
    # dsfsdtwerwer
    1+1
    return
