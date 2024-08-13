# Efficient clustering of DHNs heat consumers

This repository contains the DHNs with their graph views, python codes and results post-processing used for our article about "efficient clustering of DHNs heat consumers with the objectives to be replaced by surrogate models".

## Training DHNs

Four different DHNs have been used to train and prior-evaluate the proposed clustering process. In this context, 'training' refers to consider and train all possible clusters obtained from the clustering process using every distance threshold in [0.1 - 0.99]. The training procedures of the clusters are described in our previous article ![link](https://www.sciencedirect.com/science/article/pii/S2666546824000594?via%3Dihub). Training codes and necessary data are also available ![here](https://github.com/drod-96/smart_clusters_v1).

### Training graph 1

![DHN 1](training_dhns_graphs/training_dhn_1.pdf)

This graph describes a DHN with 71 substations nodes with 69 heat consumers and 2 heat producers. Heat consumers nodes are in grey while heat producers are in red. It contains one major loop, commonly present in DHNs. 

### Training graph 2

![DHN 2](training_dhns_graphs/training_dhn_2.pdf)

This graph describes a smaller DHN with 21 substations (19 heat consumers and 2 heat producers). Those small DHNs refer to localized heat grid (e.g., university campus).

### Training graph 3

![DHN 3](training_dhns_graphs/training_dhn_3.pdf)

This graph describes a normal sized DHN with 61 substations (59 heat consumers and 2 heat producers). It is a pure tree-like DHN without loops.

### Training graph 4

![DHN 4](training_dhns_graphs/training_dhn_4.pdf)

This graph describes a relatively bigger DHN with 91 substations (89 heat consumers and 2 heat producers). It contains a major loop and a smaller loop with may indicate recent reconfiguration of the DHN.