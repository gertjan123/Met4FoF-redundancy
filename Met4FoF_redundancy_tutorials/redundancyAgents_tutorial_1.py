"""
Example 1 of using a Redundancy Agent.
Four signals are generated and data is supplied to the Redundancy Agent.
The Redundancy Agent calculates the best consistent estimate taking into account the supplied uncertainties.
"""

import numpy as np
from agentMET4FOF.agents import AgentNetwork
from agentMET4FOF.metrological_agents import MetrologicalMonitorAgent

from Met4FoF_redundancy.agentMFred.metrological_streams_v2 import MetrologicalMultiWaveGenerator
from Met4FoF_redundancy.agentMFred.redundancyAgents1 import MetrologicalMultiWaveGeneratorAgent, RedundancyAgent


def main():
    """
    At the start of the main module all important parameters are defined. Then the agents are defined and the network
    is started. The network and the calculated results can be monitored in a browser at the address http://127.0.0.1:8050/.
    """
    # parameters
    batch_size = 10
    n_pr = batch_size
    fsam = 100
    intercept = 10
    f1 = 6
    f2 = 10
    f3 = 8
    f4 = 12
    phi1 = 1
    phi2 = 2
    phi3 = 3
    phi4 = 4
    ampl1 = 0.3
    ampl2 = 0.2
    ampl3 = 0.5
    ampl4 = 0.4
    expunc_abs = 0.2
    problim = 0.95

    # start agent network server
    agent_network = AgentNetwork(dashboard_modules=True)

    # Initialize signal generating class outside of agent framework.
    signal1 = MetrologicalMultiWaveGenerator(sfreq=fsam, intercept=intercept, freq_arr=np.array([f1]),
                                             ampl_arr=np.array([ampl1]), phase_ini_arr=np.array([phi1]),
                                             expunc_abs=expunc_abs)
    signal1.init_parameters(batch_size1=batch_size)

    signal2 = MetrologicalMultiWaveGenerator(sfreq=fsam, intercept=intercept, freq_arr=np.array([f2]),
                                             ampl_arr=np.array([ampl2]), phase_ini_arr=np.array([phi2]),
                                             expunc_abs=expunc_abs)
    signal2.init_parameters(batch_size1=batch_size)

    signal3 = MetrologicalMultiWaveGenerator(sfreq=fsam, intercept=intercept, freq_arr=np.array([f3]),
                                             ampl_arr=np.array([ampl3]), phase_ini_arr=np.array([phi3]),
                                             expunc_abs=expunc_abs)
    signal3.init_parameters(batch_size1=batch_size)

    signal4 = MetrologicalMultiWaveGenerator(sfreq=fsam, intercept=intercept, freq_arr=np.array([f4]),
                                             ampl_arr=np.array([ampl4]), phase_ini_arr=np.array([phi4]),
                                             expunc_abs=expunc_abs)
    signal4.init_parameters(batch_size1=batch_size)

    # Data source agents.
    source_name1 = "Sensor1"
    source_agent1 = agent_network.add_agent(name=source_name1, agentType=MetrologicalMultiWaveGeneratorAgent)
    source_agent1.init_parameters(signal=signal1)

    source_name2 = "Sensor2"
    source_agent2 = agent_network.add_agent(name=source_name2, agentType=MetrologicalMultiWaveGeneratorAgent)
    source_agent2.init_parameters(signal=signal2)

    source_name3 = "Sensor3"
    source_agent3 = agent_network.add_agent(name=source_name3, agentType=MetrologicalMultiWaveGeneratorAgent)
    source_agent3.init_parameters(signal=signal3)

    source_name4 = "Sensor4"
    source_agent4 = agent_network.add_agent(name=source_name4, agentType=MetrologicalMultiWaveGeneratorAgent)
    source_agent4.init_parameters(signal=signal4)

    # Redundant data processing agent
    sensor_key_list = [source_name1, source_name2, source_name3, source_name4]
    redundancy_name1 = "RedundancyAgent1"  # Name cannot contain spaces!!
    redundancy_agent1 = agent_network.add_agent(name=redundancy_name1, agentType=RedundancyAgent)
    redundancy_agent1.init_parameters1(sensor_key_list=sensor_key_list, calc_type="lcs", n_pr=n_pr, problim=problim)

    # Initialize metrologically enabled plotting agent.
    monitor_agent1 = agent_network.add_agent(name="MonitorAgent_SensorValues", agentType=MetrologicalMonitorAgent) # Name cannot contain spaces!!
    monitor_agent2 = agent_network.add_agent(name="MonitorAgent_RedundantEstimate", agentType=MetrologicalMonitorAgent)

    # Bind agents.
    source_agent1.bind_output(monitor_agent1)
    source_agent2.bind_output(monitor_agent1)
    source_agent3.bind_output(monitor_agent1)
    source_agent4.bind_output(monitor_agent1)
    source_agent1.bind_output(redundancy_agent1)
    source_agent2.bind_output(redundancy_agent1)
    source_agent3.bind_output(redundancy_agent1)
    source_agent4.bind_output(redundancy_agent1)
    redundancy_agent1.bind_output(monitor_agent2)

    # Set all agents states to "Running".
    agent_network.set_running_state()

    # Allow for shutting down the network after execution.
    return agent_network


if __name__ == "__main__":
    main()