"""
Example 1 of using a Redundancy Agent.
Four signals are generated and data is supplied to the Redundancy Agent.
The Redundancy Agent calculates the best consistent estimate taking into account the supplied uncertainties.
"""

import numpy as np
from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import MetrologicalMonitorAgent


from Met4FoF_redundancy.agentMFred.metrological_streams_v2 import (
    MetrologicalMultiWaveGenerator,
)

from Met4FoF_redundancy.agentMFred.redundancyAgents1 import (
    MetrologicalMultiWaveGeneratorAgent,
    RedundancyAgent,
)


def demonstrate_redundancy_agent_four_signals():
    """
    At the start of the main module all important parameters are defined. Then the agents are defined and the network
    is started. The network and the calculated results can be monitored in a browser at the address http://127.0.0.1:8050/.
    """
    # parameters
    batch_size = 20
    n_pr = batch_size
    fsam = 100
    intercept = 10
    freqs = [6, 10, 8, 12]
    phases = [1, 2, 3, 4]
    ampls = [0.3, 0.2, 0.5, 0.4]
    exp_unc_abs = 0.2
    problim = 0.95

    # start agent network server
    agent_network: AgentNetwork = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal_arr = [MetrologicalMultiWaveGenerator(sfreq=fsam, freq_arr=np.array([freq]), intercept=intercept,
                                                 ampl_arr=np.array([ampl]), phase_ini_arr=np.array([phi]),
                                                 value_unc=exp_unc_abs) for freq, phi, ampl in
                  zip(freqs, phases, ampls)]

    # Data source agents.
    source_agents = []
    sensor_key_list = []
    for count, signal in enumerate(signal_arr):
        sensor_key_list += ["Sensor" + str(count + 1)]
        source_agents += [agent_network.add_agent(name=sensor_key_list[-1], agentType=MetrologicalMultiWaveGeneratorAgent)]
        source_agents[-1].init_parameters(signal=signal, batch_size=batch_size)

    # Redundant data processing agent
    redundancy_name1 = "RedundancyAgent1"
    redundancy_agent1 = agent_network.add_agent(name=redundancy_name1, agentType=RedundancyAgent)
    redundancy_agent1.init_parameters1(sensor_key_list=sensor_key_list, n_pr=n_pr, problim=problim,  calc_type="lcs")

    # Initialize metrologically enabled plotting agent.
    monitor_agent1 = agent_network.add_agent(name="MonitorAgent_SensorValues",      agentType=MetrologicalMonitorAgent)
    monitor_agent2 = agent_network.add_agent(name="MonitorAgent_RedundantEstimate", agentType=MetrologicalMonitorAgent)

    # Bind agents.
    for source_agent in source_agents:
        source_agent.bind_output(monitor_agent1)
        source_agent.bind_output(redundancy_agent1)

    redundancy_agent1.bind_output(monitor_agent2)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    demonstrate_redundancy_agent_four_signals()
