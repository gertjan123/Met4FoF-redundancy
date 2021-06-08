from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_1 import demonstrate_redundancy_agent_four_signals as tut1
from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_2 import demonstrate_redundancy_agent_onesignal as tut2
from time import sleep as slp


def test_execution_of_tutorial_1():
    print('\nStarted test execution of agent tutorial 1.')
    tut1().shutdown()
    print('Finished test execution of agent tutorial 1.')


def test_execution_of_tutorial_2():
    print('\nStarted test execution of agent tutorial 2.')
    tut2().shutdown()
    print('Finished test execution of agent tutorial 2.')


if __name__ == "__main__":
    test_execution_of_tutorial_1()
    slp(5) # just to allow tutorial 1 to finish and to receive and display all log messages
    test_execution_of_tutorial_2()