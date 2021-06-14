import pytest


from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_1 import demonstrate_redundancy_agent_four_signals as tut1
from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_2 import demonstrate_redundancy_agent_onesignal as tut2
from time import sleep


@pytest.mark.agent
def test_execution_of_tutorial_1(test_network_run_time):
    print('\nStarted test execution of agent tutorial 1.')
    tut1_network = tut1()
    sleep(test_network_run_time)
    tut1_network.shutdown()
    sleep(5)
    print('Finished test execution of agent tutorial 1.')


@pytest.mark.agent
def test_execution_of_tutorial_2(test_network_run_time):
    print('\nStarted test execution of agent tutorial 2.')
    tut2_network = tut2()
    sleep(test_network_run_time)
    tut2_network.shutdown()
    sleep(5)
    print('Finished test execution of agent tutorial 2.')
