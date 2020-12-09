"""
The module :mod:`test1` calls all test functions which are implemented in the module :mod:`redundancy1`.
These test functions are:

* :func:`test_calc_consistent_estimates_no_corr`
* :func:`test_calc_best_estimate`
* :func:`test_calc_lcs`
* :func:`test_calc_lcss`

"""

import Met4FoF_redundancy.MFred.redundancy1 as mfred1


def main():
    """
    Function that calls all test procedures implemented in :mod:`redundancy1`.
    """
    mfred1.test_calc_consistent_estimates_no_corr()
    mfred1.test_calc_best_estimate()
    mfred1.test_calc_lcs()
    mfred1.test_calc_lcss()


if __name__ == '__main__':
    main()