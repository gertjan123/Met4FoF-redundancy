"""
The module :mod:`test_MFred.py` calls all test functions which are implemented in the
module :mod:`redundancy1`. These test functions are:

* :func:`test_calc_consistent_estimates_no_corr`
* :func:`test_calc_best_estimate`
* :func:`test_calc_lcs`
* :func:`test_calc_lcss`

"""

import pytest
import numpy as np
from Met4FoF_redundancy.MFred.redundancy1 import (
    calc_best_estimate,
    calc_consistent_estimates_no_corr,
    calc_lcs,
    calc_lcss,
    print_input_lcss,
    print_output_cbe,
    print_output_lcs,
    print_output_lcss,
)
from scipy.stats import multivariate_normal as mvn


@pytest.mark.cenc
def test_calc_consistent_estimates_no_corr():
    """
    Test function for calc_consistent_estimates_no_corr(), implementing two test cases.

    """
    # case with only one set of estimates
    print("Testing case with single set of estimates.")
    # input
    y_arr = np.array([20.2, 21.3, 20.5])
    uy_arr = np.array([0.5, 0.8, 0.3])
    prob_lim = 0.05
    # function
    isconsist, ybest, uybest, chi2obs = calc_consistent_estimates_no_corr(
        y_arr, uy_arr, prob_lim
    )
    # print of output
    print_output_cbe(isconsist, ybest, uybest, chi2obs)

    # case with two sets of estimates
    print("Testing case with two sets of estimates.")
    # input
    y_arr = np.array([[20.2, 21.3, 20.5], [19.5, 19.7, 20.3]])
    uy_arr = np.array([[0.5, 0.8, 0.3], [0.1, 0.2, 0.3]])
    prob_lim = 0.05
    # function
    (
        isconsist_arr,
        ybest_arr,
        uybest_arr,
        chi2obs_arr,
    ) = calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)
    # print of output
    print_output_cbe(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)


@pytest.mark.cbe
def test_calc_best_estimate():
    """
    Test function for calc_best_estimate.
    """
    print("\n-------------------------------------------------------------------\n")
    print("TESTING FUNCTION calc_best_estimate()")
    # Test case 0
    print("TEST CASE 0")
    n_reps = 1000
    y_arr = np.array([20.2, 20.5, 20.8])
    vy_arr2d = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 4]])
    problim = 0.95
    isconsist, ybest, uybest, chi2obs = calc_best_estimate(y_arr, vy_arr2d, problim)
    print_output_cbe(isconsist, ybest, uybest, chi2obs)

    # Test case 1: check limit probability limprob
    print("TEST CASE 1")
    n_reps = 10000
    print(
        "Starting with repeating the procedure %d times in order to check the acceptance statistics..."
        % n_reps
    )
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
    print(
        "After repeating the procedure %d times, the data generated by the assumed model was "
        "accepted with probability %4.4f, whereas %4.4f is expected.\n"
        % (n_reps, frackeep, problim)
    )
    difFracMax = 0.01
    if abs(frackeep - problim) > difFracMax:
        raise ValueError(f'The experimental fraction {frackeep:4.4f} deviated more than {difFracMax:4.4f} ' 
                         'from the expected fraction {problim:4.4f}')


@pytest.mark.lcs
def test_calc_lcs():
    """
    Test function for :func:`calc_lcs`.
    Implements 4 test cases.
    """
    print("\n-------------------------------------------------------------------\n")
    print("TESTING FUNCTION calc_lcs()")
    # Test case 0:
    print("TEST CASE 0")
    # input
    y_arr = np.array([20, 20.6, 20.5, 19.3])
    vy_arr2d = np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 1:
    print("TEST CASE 1")
    # input
    y_arr = np.array([20, 23.6, 20.5, 19.3])
    vy_arr2d = np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 2 with two optimal solutions
    print("TEST CASE 2")
    # input
    y_arr = np.array([10, 11, 20, 21])
    vy_arr2d = 5 * np.identity(4) + np.ones((4, 4))
    problim = 0.95
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcs(y_arr, vy_arr2d, problim)
    # print output
    print_output_lcs(n_sols, ybest, uybest, chi2obs, indkeep, y_arr)

    # Test case 3: check limit probability limprob
    print("TEST CASE 3")
    n_reps = 10000
    print(
        "Starting to repeating the procedure %d times in order to check the acceptance statistics..."
        % n_reps
    )
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
    print(f'After repeating the procedure {n_reps:.0f} times, the data generated by the assumed model was '
          f'accepted with probability {frackeep:4.4f}, whereas {problim:4.4f} is expected.')
    difFracMax = 0.01
    if abs(frackeep - problim) > difFracMax:
        raise ValueError(f'The experimental fraction {frackeep:4.4f} deviated more than {difFracMax:4.4f} ' 
                         'from the expected fraction {problim:4.4f}')


@pytest.mark.lcss
def test_calc_lcss():
    """
    Test function for method :func:`calc_lcss`.
    Implements 4 test cases.
    """
    print("\n-------------------------------------------------------------------\n")
    print("TESTING FUNCTION calc_lcss()\n")
    # Test case 0:
    print("TEST CASE 0")
    print("Test case with A = identity matrix and a = zero, i.e. same as lcs.")
    # input
    x_arr = np.array([22.3, 20.6, 25.5, 19.3])
    vx_arr2d = np.identity(4) + np.ones((4, 4))
    a_arr = np.zeros(4)
    a_arr2d = np.identity(4)
    problim = 0.95
    # print input
    print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(
        a_arr, a_arr2d, x_arr, vx_arr2d, problim
    )
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 1:
    print(
        "TEST CASE 1\nlcss which reduces to lcs after transformation of the linear "
        "system.\nInput data is the same as in the last case and the results should "
        "be the same as well."
    )
    # input
    # x_arr = (same as above) np.array([20, 23.6, 20.5, 19.3])
    # vx_arr2d = (same as above) np.identity(4) + np.ones((4, 4))

    a_arr2d = np.array([[1, 2, 3, 4], [2, -5, 4, 1], [2, 9, 1, 0], [3, 5, -2, 4]])
    # a_arr2d = np.identity(4) + np.ones((4,4))
    s = np.sum(a_arr2d, 1)
    s.shape = (s.shape[0], 1)  # set the second dimension to 1
    a_arr2d = a_arr2d / s  # make all row sums equal to 1

    # Manipulate input to create a non trivial vector a_arr
    dx_arr = np.array([1, 2, 3, 4])
    x_arr = x_arr - dx_arr
    a_arr = a_arr + np.matmul(a_arr2d, dx_arr)

    # problim = 0.95
    # print input
    print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim)
    # function
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(
        a_arr, a_arr2d, x_arr, vx_arr2d, problim
    )
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 2 with two optimal solutions
    print("TEST CASE 2")
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
    n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(
        a_arr, a_arr2d, x_arr, vx_arr2d, problim
    )
    # print output
    print_output_lcss(n_sols, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d)

    # Test case 3: check limit probability limprob
    print("TEST CASE 3")
    n_reps = 10000
    print(
        "Repeating the procedure %d times in order to check the acceptance statistics."
        % n_reps
    )
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
        n_sols, ybest, uybest, chi2obs, indkeep = calc_lcss(
            a_arr, a_arr2d, x_arr, vx_arr2d, problim
        )
        if indkeep.shape[-1] == len(x_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    print(f'After repeating the procedure {n_reps:.0f} times, the data generated by the assumed model was '
          f'accepted with probability {frackeep:4.4f}, whereas {problim:4.4f} is expected.')
    difFracMax = 0.01
    if abs(frackeep - problim) > difFracMax:
        raise ValueError(f'The experimental fraction {frackeep:4.4f} deviated more than {difFracMax:4.4f} ' 
                         f'from the expected fraction {problim:4.4f}!')


#if __name__ == '__main__':
#    test_calc_consistent_estimates_no_corr()
#    test_calc_best_estimate()
#    test_calc_lcs()
#    test_calc_lcss()
