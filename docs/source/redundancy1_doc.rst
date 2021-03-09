MFred: Python functions for best estimate calculation in a sensor network in the presence of redundancy
========================================================================================================

In this section some method for analysing redundant measurement data is presented. *Redundancy* means that there is more than one way to derive the value of the measurand *Y* from the values of the sensor data *X_i*. Following main cases are considered in the module:

#. Redundant measurement of the measurand *Y* by independent sensors directly measuring *Y*
#. Redundant measurement of the measurand *Y* by correlated sensors directly measuring *Y*
#. Redundant measurement of the measurand *Y* by correlated sensors *X_i* indirectly measuring *Y*, with a linear relationship **y** = **a** + *A* * **x** between the vector **x** of sensor values and the vector **y** containing the various (redundant) estimates of the  measurand *Y*, where **a** is a vector and *A* a matrix both of appropriate size.

Details of the different modules are presented in the next sections.

Details of the main module :mod:`redundancy1`
---------------------------------------------
.. automodule:: Met4FoF_redunacny.MFred.redundancy1
    :members:

Details of the test module :mod:`test1`
---------------------------------------
.. automodule:: test_redundancy.test_MFred
    :members:
