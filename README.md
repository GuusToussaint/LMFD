# LMFD: Latent Monotonic Feature Discovery

This repository contains the code used in the LMFD paper.

### Abstract
Many systems in our world age, degrade or otherwise move
slowly but steadily in a certain direction. When monitoring such systems
by means of sensors, one often assumes that some form of ‘age’ is latently
present in the data, but perhaps the available sensors do not readily pro-
vide this useful information. The task that we study in this paper is to
extract potential proxies for this ‘age’ from the available multi-variate
time series without having clear data on what ‘age’ actually is. We argue
that when we find a sensor, or more likely some discovered function of
the available sensors, that is sufficiently monotonic, that function can
act as the proxy we are searching for. Using a carefully defined gram-
mar and optimising the resulting equations in terms of monotonicity,
defined as the absolute Spearman’s Rank Correlation between time and
the candidate formula, the proposed approach generates a set of candi-
date features which are then fitted and assessed on monotonicity. The
proposed system is evaluated against an artificially generated dataset and
two real-world datasets. In all experiments, we show that the system is
able to combine sensors with low individual monotonicity into latent fea-
tures with high monotonicity. For the real-world dataset of InfraWatch, a
structural health monitoring project, we show that two features with in-
dividual absolute Spearman’s ρ values of 0.13 and 0.09 can be combined
into a proxy with an absolute Spearman’s ρ of 0.95. This demonstrates
that our proposed method can find interpretable equations which can
serve as a proxy for the ‘age’ of the system.
1 Introduction
Many systems in our world age, degrade, or slowly but steadily move in a certain
direction. For example, a highway bridge slowly degrades during its lifetime,
a cyclist in the Tour de France will tire over the course of a long stage, and
the battery charge of an electric vehicle will deplete as it drives. While in the
last example the continuous tracking of the state of charge is fairly doable, in
many other applications, the actual ‘age’ of the system may be hidden, and only
latently expressed in any data measured about the system. Often, any available
sensors will capture the easily measurable information that is often of dynamic
nature (what is currently happening in and around the system?), but the actual
quantity of interest is much harder to obtain. For example, there can be plenty
of measurements for the elite cyclist, including their heart rate, power output,
skin conductivity, etc., but the measure of fatigue is hard to define, let alone



