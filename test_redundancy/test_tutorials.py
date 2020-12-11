from Met4FoF_redundancy.agentMFred.redundancyAgents_tutorial_1 import main as tut1
from Met4FoF_redundancy.agentMFred.redundancyAgents_tutorial_2 import main as tut2


def test_execution_of_tutorial_1():
    tut1().shutdown()

def test_execution_of_tutorial_2():
    tut2().shutdown()
