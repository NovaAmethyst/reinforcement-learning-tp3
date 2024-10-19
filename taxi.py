"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

import matplotlib.pyplot as plt
import matplotlib.animation as animation


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r
        s = next_s
        if done:
            break
        # END SOLUTION

    return total_reward

# Get base results and animations

def get_animation(env, agent, filepath, t_max=int(1e4), save=False):
    fig = plt.figure()
    plt.axis('off')
    s, _ = env.reset()
    ims = [[plt.imshow(env.render(), animated=True)]]
    for _ in range(t_max):
        a = agent.get_best_action(s)
        s, _, done, _, _ = env.step(a)
        im = plt.imshow(env.render(), animated=True)
        ims.append([im])
        if done:
            break

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    if save:
        ani.save(filepath)
    else:
        plt.show()
    plt.close()

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/qlearning/training/base/base_qlearning_iteration_{i + 1}.gif", t_max=200, save=True)

get_animation(env, agent, f"animations/qlearning/base_qlearning.gif", t_max=200, save=True)


# TODO: créer des vidéos de l'agent en action

def get_graph(x, y, title, x_label, y_label, filepath, save=False, use_line=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    if use_line:
        ax.plot(x, y)
    else:
        ax.scatter(x, y, s=2 )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if save:
        fig.savefig(filepath)
    else:
        plt.show()
    plt.close()

x = np.arange(0, 1000)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_1/learning_reward_base_qlearning.png", save=True)
final_rewards = []

final_rewards.append(np.mean(rewards[-100:]))

# Seek hyper parameter epsilon
epsilons = np.arange(0, 0.5, 0.05)
epsilon_res = []
for eps in epsilons:
    agent = QLearningAgent(
        learning_rate=0.5, epsilon=eps, gamma=0.99, legal_actions=list(range(n_actions))
    )
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent, t_max=200))
    epsilon_res.append(np.mean(rewards[-100:]))
    print(f"Epsilon: {eps:.2f}, mean last rewards: {np.mean(rewards[-100:]):.2f}")

final_epsilon = epsilons[np.argmax(epsilon_res)]

get_graph(epsilons, epsilon_res, "Reward per epsilon value", "Epsilon", "Reward", "figures/part_1/learning_reward_qlearning_eps_search.png", save=True, use_line=True)

# Seek hyper parameter alpha (learning rate)
alphas = np.arange(0.05, 1.05, 0.05)
alpha_res = []
for alph in alphas:
    agent = QLearningAgent(
        learning_rate=alph, epsilon=final_epsilon, gamma=0.99, legal_actions=list(range(n_actions))
    )
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent, t_max=200))
    alpha_res.append(np.mean(rewards[-100:]))
    print(f"Alpha: {alph:.2f}, mean last rewards: {np.mean(rewards[-100:]):.2f}")

final_alpha = alphas[np.argmax(alpha_res)]

get_graph(alphas, alpha_res, "Reward per learning rate", "Learning rate", "Reward", "figures/part_1/learning_reward_qlearning_alpha_search.png", save=True, use_line=True)

agent = QLearningAgent(
        learning_rate=final_alpha, epsilon=final_epsilon, gamma=0.99, legal_actions=list(range(n_actions))
    )

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/qlearning/training/optimized/optimized_qlearning_iteration_{i + 1}.gif", t_max=200, save=True)

assert np.mean(rewards[-100:]) > 0.0

get_animation(env, agent, f"animations/qlearning/optimized_qlearning.gif", t_max=200, save=True)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_1/learning_reward_optimized_qlearning.png", save=True)

eps_qlearning = final_epsilon
alph_qlearning = final_alpha
final_rewards.append(np.mean(rewards[-100:]))

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

print("Qlearning with epsilon scheduling")

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/qlearning_eps_schedule/training/base/base_qlearning_eps_schedule_iteration_{i + 1}.gif", t_max=200, save=True)

assert np.mean(rewards[-100:]) > 0.0

final_rewards.append(np.mean(rewards[-100:]))

# TODO: créer des vidéos de l'agent en action
get_animation(env, agent, f"animations/qlearning_eps_schedule/base_qlearning_eps_schedule.gif", t_max=200, save=True)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_2/learning_reward_base_qlearning_eps_schedule.png", save=True)

epsilons = np.arange(0, 1.05, 0.05)
epsilon_res = []
for eps in epsilons:
    agent = QLearningAgentEpsScheduling(
        learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions)), epsilon_start=eps
    )
    agent.reset()
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent, t_max=200))
    epsilon_res.append(np.mean(rewards[-100:]))
    print(f"Epsilon: {eps:.2f}, mean last rewards: {np.mean(rewards[-100:]):.2f}")

final_epsilon_start = epsilons[np.argmax(epsilon_res)]

get_graph(epsilons, epsilon_res, "Reward per starting epsilon value", "Epsilon", "Reward", "figures/part_2/learning_reward_qlearning_eps_schedule_start_eps_search.png", save=True, use_line=True)

alphas = np.arange(0.05, 1.05, 0.05)
alpha_res = []
for alph in alphas:
    agent = QLearningAgentEpsScheduling(
        learning_rate=alph, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions)), epsilon_start=final_epsilon_start
    )
    agent.reset()
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent, t_max=200))
    alpha_res.append(np.mean(rewards[-100:]))
    print(f"Alpha: {alph:.2f}, mean last rewards: {np.mean(rewards[-100:]):.2f}")

final_alpha = alphas[np.argmax(alpha_res)]

get_graph(alphas, alpha_res, "Reward per learning rate", "Learning rate", "Reward", "figures/part_2/learning_reward_qlearning_esp_schedule_alpha_search.png", save=True, use_line=True)

agent = QLearningAgentEpsScheduling(
        learning_rate=final_alpha, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions)), epsilon_start=final_epsilon_start
    )
agent.reset()

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/qlearning_eps_schedule/training/optimized/optimized_qlearning_iteration_{i + 1}.gif", t_max=200, save=True)

assert np.mean(rewards[-100:]) > 0.0

get_animation(env, agent, f"animations/qlearning_eps_schedule/optimized_qlearning.gif", t_max=200, save=True)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_2/learning_reward_optimized_qlearning_eps_schedule.png", save=True)

eps_qlearning_scheduling = final_epsilon
alph_qlearning_scheduling = final_alpha
final_rewards.append(np.mean(rewards[-100:]))

####################
# 3. Play with SARSA
####################

print("Sarsa")

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/sarsa/training/base/base_sarsa_iteration_{i + 1}.gif", t_max=200, save=True)

assert np.mean(rewards[-100:]) > 0.0
final_rewards.append(np.mean(rewards[-100:]))
get_animation(env, agent, f"animations/sarsa/base_sarsa.gif", t_max=200, save=True)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_3/learning_reward_base_sarsa.png", save=True)

alphas = np.arange(0.05, 1.05, 0.05)
alpha_res = []
for alph in alphas:
    agent = SarsaAgent(
        learning_rate=alph, gamma=0.99, legal_actions=list(range(n_actions))
    )
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent, t_max=200))
    alpha_res.append(np.mean(rewards[-100:]))
    print(f"Alpha: {alph:.2f}, mean last rewards: {np.mean(rewards[-100:]):.2f}")

final_alpha = alphas[np.argmax(alpha_res)]

get_graph(alphas, alpha_res, "Reward per learning rate", "Learning rate", "Reward", "figures/part_3/learning_reward_sarsa_alpha_search.png", save=True, use_line=True)

agent = SarsaAgent(
        learning_rate=final_alpha, gamma=0.99, legal_actions=list(range(n_actions))
    )

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
        get_animation(env, agent, f"animations/sarsa/training/optimized/optimized_sarsa_iteration_{i + 1}.gif", t_max=200, save=True)

assert np.mean(rewards[-100:]) > 0.0

final_rewards.append(np.mean(rewards[-100:]))
get_animation(env, agent, f"animations/sarsa/optimized_sarsa.gif", t_max=200, save=True)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_3/learning_reward_optimized_sarsa.png", save=True)

print(f"QLearning parameters: epsilon = {eps_qlearning}, alpha = {alph_qlearning}")
print(f"QLearning with epsilon scheduling parameters: starting epsilon = {eps_qlearning_scheduling}, alpha = {alph_qlearning_scheduling}")
print(f"Sarsa parameters: alpha = {final_alpha}")
print(f"List of all results: {final_rewards}")