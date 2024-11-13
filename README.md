# Active Agents: An Active Inference Approach to Agent-Based Modeling in the Social Sciences
This page is being expanded over time as I continue to develop and deliver instructional materials for the Active Inference Institute: https://www.activeinference.institute/

### Presentations and Resources:

## IC2S2: Landing page and repository for the 'Active Agents' tutorial held 17 July, 2024 at the 10th International Conference on Computational Social Science.
- IC2S2 2024 Tutorials: https://ic2s2-2024.org/tutorials
- [Tutorial Slideshow presentation and follow-along](https://docs.google.com/presentation/d/1gHAX-5Ughdd47oUDm2oQKYfgjEW5XkAnTXUdXQG-XXc/edit?usp=sharing)

- [Google Colab in-brower code for single-agent and multi-agent simulations](https://colab.research.google.com/drive/14oMDEByadHRGmZ8MFQvuc4iPnX522GsY?usp=sharing)

 
## "ActInf ModelStream 015.0: Andrew Pashea 'Active Agents: Agent-Based Modeling in the Social Sciences'": Livestreamed talk with the Active Inference Institute discussing the basic mechanics and code for developing POMDP Active Inference agents using the `pymdp` library
- [Code demonstration for constructing agents with clarifying Markdown instructional descriptions](https://colab.research.google.com/drive/1VWvwZFzQlNdL8w1X8UQ1VWNBESZ32_AW?usp=sharing)
- [Livestream link (YouTube, hosted by Active Inference Institute](https://www.youtube.com/watch?v=wAd-ARzquj8)

## Applied Active Inference Symposium ~ 2024 
- Main website: [symposium.activeinference.institute](symposium.activeinference.institute)
- [Slideshow presentation (under construction, revisions to include Biofirm updates in-progress)](https://docs.google.com/presentation/d/10ojJpwuVfuk7N0eMx0-ksN0BLPGgaO79I6e2B_H1ir4/edit?usp=sharing)
  
__________________________
### Instructions for Google Colab links
1. Open the Google Colab links to open a read-only version of the corresponding code. With a valid Google account, click `File` $\rightarrow$ `Save a copy in Drive` to save an editable copy.
2. Click `Runtime` $\rightarrow$ `Run all` to run the code. The simulations near the end of the IC2S2 script will take a bit of time to play out (two simulations, 30 generative models each), so better to have these ready to go so that you can follow along smoothly!
__________________________
<img src="https://github.com/apashea/IC2S2-Active-Inference-Tutorial/blob/main/Single-Agent%20Inference%20-%20Simulation%201.jpg?raw=true" width="640" height="360">

__________________________

<img src="https://github.com/apashea/IC2S2-Active-Inference-Tutorial/blob/main/Multi-Agent%20Inference%20-%20Simulation%201.jpg?raw=true" width="640" height="360">

__________________________

### IC2S2: In brief, this tutorial will cover:
- An overview of what is argued to be "traditional" rules-based Agent-Based Modeling (ABM) followed by a relatively recent shift towards "cognitive" modeling of agents with their own internal beliefs and mechanisms for autonomous action, e.g., action, perception, decision-making, planning, learning.
  - Autonomous agents and micro-level assumptions: Calls for "cognitive" modeling center on introducing more empirically-supportable realism to the assumptions made about agents at the micro-level. "Traditional" agents' simplicity of hard-coded conditional decision-making ('if X then do Y, else do Z', lacking flexibility and cognitive or memory mechanisms) sacrifices autonomy and direct connection to human *natural intelligence* -- in caricature, a level of simplicity which produces pre-programmed robots. On the other hand, incorporating research in cognitive/neuroscience, physics, and other fields combined with methods of mathematical formalization and modeling provides methods for coding agents in ABM settings as *autonomous and forward-looking generative agents* with interpretable parameters and behavioral outcomes.
  - Scientifically-informed policy design with a beliefs-based framework: Cognitive, autonomous agents with their own *mechanisms for  action, perception, decision-making, and planning*, guided in part by their own (internal representations of) *beliefs* expands possibilities for *policy, intervention design, and research methodology* expressly targeting beliefs and behavior, e.g., fostering students' sense of personal efficacy, mitigating financial overconfidence, or disincentivizing self-harming behaviors. Composing agents equipped with these cognitive mechanisms approximates the *natural intelligence* of humans, such that experimental simulations for counterfactual testing and modeling better approximate our social and physical reality.
- Reinforcement Learning as a popular paradigm for approaching this cognitive turn in ABM: its principles, potential for low computational cost, but also its limitations for cognitive modeling.
  - RL lacks a *coherent* cognitive theory/framework or empirical support for approximating *human* behavior, i.e. what social scientists aim to model. This is reasonable as most RL research advances are focused on task optimization, e.g., for building AI tools and bots.
  - RL's common issue regarding how to manually define exploratory vs. exploitative behavior--the famous *explore-exploit dilemma* centered on questions of what propels agents to seek information--for example epsilon-greedy as a performative yet *ad hoc* means of defining either type of behavior where information seeking is merely treated as a low probability action or policy space.
- Introduction to **Active Inference** (ActInf) as an established framework derived from cognitive/neuroscience with the explicit goal of cognitive modeling and approximating human behavior.
  - ActInf provides what RL lacks in this case: foundations in the study of behavior in humans and biological organisms generally with empirical support for the assumptions it relies upon, e.g., neuronal dynamics, physiological substrates, "natural intelligence" (as opposed to "artificial intelligence").
  - ActInf is explicitly a *beliefs-based framework*, well suited for the studying the relationship between *beliefs* and *behavior* with proxy parameters for interpretation of this relationship.
  - False inference: ActInf does not (but can) focus on task-optimization. Optimality is defined relative to the agent: what might be "rational" to the agent may not be the optimal solution in the task. (Despite this "realism" ActInf models nonetheless are highly competitive in comparison to RL models)
  - ActInf's applications to social science, ranging beyond the neuronal to the human and beyond to the study of collective behavior
  - ActInf provides a theory of exploratory and exploitative behavior and "automatically" balances them by relating them in gradient descent over expected free energy as the function to be optimized
- Single-agent inference: We build an Active Inference agent "from scratch" using the `pymdp` package via the provided Google Colab script
  - The core ideas of ActInf are described and then directly applied.
  - We cover how agents infer and learn based on observations and elicit actions based on the inference process; inference with a single agent is demonstrated, including in code with plots for the results.
- Multi-Agent demonstration: We recreate a famous paradigm (Lazer & Friedman, 2007) for analyzing explore-exploit behavior in networks (and subnetworks) of $N$ agents
  - We look at two particular simulations (parameter configurations/sweeps) and then extract and plot the agents' performance- *and belief*-related metrics for comparison, providing further insights into this classic paradigm.
  - The Google Colab script provides all of the relevant code, as well as options for further experimentation.
 
*"The Network Structure of Exploration and Exploitation" (Lazer & Friedman 2007) https://ndg.asc.upenn.edu/wp-content/uploads/2016/04/Lazer-Friedman-2007-ASQ.pdf

### Instructions
1. Open the Google Colab link below to open a read-only version of the tutorial code. With a valid GMail account, click `File` $\rightarrow$ `Save a copy in Drive` to save an editable copy.
2. Click `Runtime` $\rightarrow$ `Run all` to run the code. The simulations near the end will take a bit of time to play out (two simulations, 30 generative models each, roughly 4-5 minutes total), so better to have these ready to go so that you can follow along smoothly!


