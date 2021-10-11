# Residual Reinforcement Learning
## Notes1:
3 main causes of the success of RPL:

a) we take care to initialize the residual policy so that its output at first matches the initial policy. When the initial policy is strong, this initialization gives RPL a clear boost;

b) In learning from scratch with sparse rewards and long horizons, the first successful trajectory must be discovered by chance. Hindsight Experience Replay is designed to face this challenge, but RPL offers a more direct solution. RPL can discover successful trajectories immediately if the initial policy produces them with nontrivial frequency;

c) the residual reinforcement learning problem induced by the initial policy may be easier than the original problem

## Notes2:
a) Human controller specifies the general trajectory of the optimal policy, environment samples are required only to learn the corrective feedback behavior

## Notes3:
a) Residual RL improves the generalization abilities of DMPs

## Notes4:
a) They leverage visual features and proprioceptive inputs to learn the base policy and then improve this policy with residual reinforcement learning from sparse rewards.

## Notes5:
a) GMM-GMR+Residual Reinforcement Learning

## Notes6:
a) they incorporate operational space visual and haptic informatin into reinforcement learning 

b) proactive action to solve the partially observable MDP problem.

## Notes 7:
a) residual RL with traditional P controller as the base policy

## Notes 8:
a) online predictions given a noisy user input instead of learning to mimic a skill in an offline fashion

b) the residual actions works on top of a user input instead of a pre-trained policy

## Notes 9:
a) minimally adjust the human's actions such that a set of goal-agnostic constraints are satisfied

b) the unknown human policy plays the role of the nominal policy that the residual shared autonomy agent corrects to improve performance

## Notes 10 & 11:
a) using the uncertainty as the signal to switch between model-based method and RL method. 

## Refrence:
1. Residual Policy Learning
2. Residual Reinforcement Learning for Robot Control
3. Residual Learning from demonstration: adapting DMP for contact-rich insertion tasks
4. Residual reinforcement learning from demonstrations
5. Efficient Insertion control for precision assembly based on demonstration learning and reinforcement learning
6. proactive action visual residual reinforcement learning for contact-rich tasks using a torque-controlled robot
7. Deep reinforcement learning for industrial insertion tasks with visual inputs and natural rewards
8. physics-based dexterous manipulations with estimated hand poses and residual reinforcement learning
9. residual policy learning for shared autonomy
10. combining learning from demonstration with learning by exploration to facilitate contact-rich tasks
11. guided uncertainty-aware policy optimization: combining learning and model-based strategies for sample-efficient policy learning