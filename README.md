# LMFD: Latent Monotonic Feature Discovery

This repository contains the code used in the LMFD paper.

### Abstract
Many systems in our world age, degrade or otherwise move slowly but steadily in a certain direction. When monitoring such systems by means of sensors, one often assumes that some form of ‘age’ is latently present in the data, but perhaps the available sensors do not readily provide this useful information. The task that we study in this paper is to extract potential proxies for this ‘age’ from the available multi-variate time series without having clear data on what ‘age’ actually is. We argue that when we find a sensor, or more likely some discovered function of the available sensors, that is sufficiently monotonic, that function can act as the proxy we are searching for. Using a carefully defined grammar and optimising the resulting equations in terms of monotonicity, defined as the absolute Spearman’s Rank Correlation between time and the candidate formula, the proposed approach generates a set of candidate features which are then fitted and assessed on monotonicity. The proposed system is evaluated against an artificially generated dataset and two real-world datasets. In all experiments, we show that the system is able to combine sensors with low individual monotonicity into latent features with high monotonicity. For the real-world dataset of InfraWatch, a structural health monitoring project, we show that two features with individual absolute Spearman’s ρ values of 0.13 and 0.09 can be combined into a proxy with an absolute Spearman’s ρ of 0.95. This demonstrates that our proposed method can find interpretable equations which can serve as a proxy for the ‘age’ of the system.


### Experiments

The paper contains the following experiments:
1. Artificial dataset 
```bash
    python experiments.py --experiment artificial
```
2. Climate dataset 
```bash
    python experiments.py --experiment climate
```
3. InfraWatch dataset 
```bash
    python experiments.py --experiment InfraWatch
```