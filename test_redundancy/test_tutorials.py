from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_1 import demonstrate_redundancy_agent_four_signals as tut1
from Met4FoF_redundancy_tutorials.redundancyAgents_tutorial_2 import demonstrate_redundancy_agent_onesignal as tut2


def test_execution_of_tutorial_1():
    tut1().shutdown()

def test_execution_of_tutorial_2():
    tut2().shutdown()
