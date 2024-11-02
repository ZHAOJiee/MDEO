# Multi-Domain Evolutionary Optimization of Network Structures
This is our implementation for the paper:

Jie Zhao, Kang Hao Cheong and Yaochu Jin. Multi-Domain Evolutionary Computation for Network Structure.

## Abstract

Multi-task evolutionary optimization, focusing on addressing complex problems through optimizing multiple tasks simultaneously, has attracted much attention. While this emerging paradigm has been primarily focusing on task similarity, there remains a hugely untapped potential in harnessing the shared characteristics between different domains. For example, real-world complex systems usually share the same characteristics, such as the power-law rule, small-world property and community structure, thus making it possible to transfer solutions optimized in one system to another to facilitate the optimization. Drawing inspiration from this observation of shared characteristics within complex systems, we propose a novel framework, multi-domain evolutionary optimization. To examine the performance of the proposed framework, we utilize a challenging combinatorial problem--community deception as the illustrative optimization task. In addition, we propose a community-level measurement of graph similarity to manage the knowledge transfer among domains. Furthermore, we develop a graph learning-based network alignment model that serves as the conduit for effectively transferring solutions between different domains. Moreover, we devise a self-adaptive mechanism to determine the number of transferred solutions from different domains and introduce a novel mutation operator based on the learned mapping to facilitate the utilization of knowledge from other domains. Experiments on multiple real-world networks of different domains demonstrate superiority of the proposed framework in efficacy compared to classical evolutionary optimization. 

## Experimental Enviroment

* igraph               0.10.4

* numpy                1.21.5

* networkx             2.6.3

* torch                1.13.1

* torch-geometric      2.3.1

## Usage

1. For multi-domain evolutionary optimization, please run MDEO_EVC.py.
2. FOr network alignment, please run Network_alignment.py (The trained mapping is already under the file of "result")


