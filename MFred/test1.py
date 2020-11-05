"""
The module :mod:`test1.py` calls all test functions which are implemented in the module :mod:`redundancy1.py`.
These test functions are:

* :func:`test_calc_best_estimate`
* :func:`test_calc_lcs`
* :func:`test_calc_lcss`

"""

import MFred.redundancy1 as mfred1


def main():
    """
    Function that calls all test procedures implemented in redundancy1.py.
    """
    mfred1.test_calc_best_estimate()
    mfred1.test_calc_lcs()
    mfred1.test_calc_lcss()


if __name__ == '__main__':
    main()