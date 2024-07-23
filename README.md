# Active Agents: An Active Inference Approach to Agent-Based Modeling in the Social Sciences
Landing page and repository for the 'Active Agents' tutorial held 17 July, 2024 at the 10th International Conference on Computational Social Science.
IC2S2 2024 Tutorials: https://ic2s2-2024.org/tutorials
__________________________
![](https://github.com/apashea/IC2S2-Active-Inference-Tutorial/blob/main/Single-Agent%20Inference%20-%20Simulation%201.jpg?raw=true =250x250)
__________________________
![](https://github.com/apashea/IC2S2-Active-Inference-Tutorial/blob/main/Multi-Agent%20Inference%20-%20Simulation%201.jpg?raw=true =250x250)
__________________________
In brief, this tutorial will cover:
- An overview of what is argued to be "traditional" rules-based Agent-Based Modeling (ABM) followed by a relatively recent shift towards "cognitive" modeling of agents with their own internal beliefs and mechanisms for autonomous action
- Reinforcement Learning as a popular paradigm for approaching this cognitive turn in ABM: its principles, capacity for low computational costs, but also its limitations for cognitive modeling.
  - RL lacks a *coherent* cognitive theory or empirical support for approximating *human* behavior, i.e. what social scientists aim to model. This is reasonable as most RL research advances are focused on task optimization, e.g., for building AI tools and bots.
  - RL's common issue of defining exploratory vs. exploitative behavior, for example epsilon-greedy as a performative yet *ad hoc* means of defining each.
- Introduction to **Active Inference** (ActInf) as an established framework derived from cognitive/neuroscience with the explicit goal of cognitive modeling and approximating human behavior.
  - ActInf provides what RL lacks in this case: foundations in the study of behavior in humans and biological organisms generally with empirical support for the assumptions it relies upon, e.g., neuronal dynamics, physiological substrates.
  - ActInf is explicitly a *beliefs-based framework*, well suited for the studying the relationship between *beliefs* and *behavior* with proxy parameters for interpretation of this relationship.
  - False inference: ActInf does not (but can) focus on task-optimization. Optimality is defined relative to the agent: what might be "rational" to the agent may not be the optimal solution in the task. (Despite this "realism" ActInf models nonetheless are highly competitive in comparison to RL models)
  - ActInf's applications to social science, ranging beyond the neuronal to the human and beyond to the study of collective behavior
  - ActInf provides a theory of exploratory and exploitative behavior and "automatically" balances them by relating them in gradient descent over expected free energy as the function to be optimized
- Single-agent inference: We build an Active Inference agent "from scratch" using the `pymdp` package via the provided Google Colab script
  - The core ideas of ActInf are described and then directly applied.
  - We cover how agents infer learn based on observations and elicit actions based on the inference process; inference with a single agent is demonstrated, including in code with plots for the results.
- Multi-Agent demonstration: We recreate a famous paradigm (Lazer & Friedman, 2007) for analyzing explore-exploit behavior in networks (and subnetworks) of $N$ agents
  - We can extract and plot the agents' performance- *and belief*-related metrics, providing further insights into this classic paradigm
  - The Google Colab script provides all of the relevant code, as well as options for further experimentation
 
*"The Network Structure of Exploration and Exploitation" (Lazer & Friedman 2007) https://ndg.asc.upenn.edu/wp-content/uploads/2016/04/Lazer-Friedman-2007-ASQ.pdf
__________________________
### Instructions
1. Open the Google Colab link below to open a read-only version of the tutorial code. With a valid GMail account, click `File` $\rightarrow$ `Save a copy in Drive` to save an editable copy.
2. Click `Runtime` $\rightarrow$ `Run all` to run the code. The simulations near the end will take a bit of time to play out (two simulations, 30 generative models each, roughly 4-5 minutes total), so better to have these ready to go so that you can follow along smoothly!

__________________________
### Code/slides - live links (subject to modification)

##### Slide presentation and follow-along:
- https://docs.google.com/presentation/d/1gHAX-5Ughdd47oUDm2oQKYfgjEW5XkAnTXUdXQG-XXc/edit?usp=sharing
##### Google Colab:
- https://colab.research.google.com/drive/14oMDEByadHRGmZ8MFQvuc4iPnX522GsY?usp=sharing
