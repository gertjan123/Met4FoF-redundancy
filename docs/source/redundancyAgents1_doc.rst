agentMFred: Python software agents for processing redundant measurement data
==============================================================================

Some of the methods of the module :mod:`MFred` have been incorporated into a Redundancy Agent that can be used in the `Met4FoF agent framework <https://agentmet4fof.readthedocs.io>`_. The Redundancy Agent can be found in :mod:`redundancyAgents1`. It uses of a metrological datastream which can be found in :mod:`metrological_streams_v2`. The usage of the Redundancy Agent is illustrated with two examples contained in two tutorials.

In tutorial :mod:`redundancyAgents_tutorial_1` four independent signals are generated and the Redundancy Agent calculates the best estimate with associated uncertainty, respecting the input uncertainties, and rejecting sensor values that may be erroneous. In this case the sensors directly measure the measurand.

In tutorial :mod:`redundancyAgents_tutorial_2` a single signal containing redundant, correlated information is analyzed, and the best estimate with associated uncertainty, respecting all provided input uncertainties, and rejecting sensor values that may be erroneous. In this case the sensors do not directly measure the measurand, but the measurand is linked to the sensor values by means of four linear equations. The fact that there are four equations and not just one is the cause of the redundancy.

Details of the different modules are presented in the next sections.

Details of the module :mod:`metrological_streams_v2`
----------------------------------------------------
.. automodule:: Met4FoF_redundancy.agentMFred.metrological_streams_v2
    :members:


Details of the module :mod:`redundancyAgents1`
-------------------------------------------------
.. automodule:: Met4FoF_redundancy.agentMFred.redundancyAgents1
    :members:


Details of the module :mod:`redundancyAgents_tutorial_1`
---------------------------------------------------------
.. automodule:: Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_1
    :members:


Details of the module :mod:`redundancyAgents_tutorial_2`
---------------------------------------------------------
.. automodule:: Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_2
    :members:


