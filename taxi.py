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
from qlearning_eps_reduce import QLearningAgentEpsReduce
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


rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

#assert np.mean(rewards[-100:]) > 0.0
# TODO: créer des vidéos de l'agent en action
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

def get_graph(x, y, title, x_label, y_label, filepath, save=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(x, y, s=2 )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if save:
        fig.savefig(filepath)
    else:
        plt.show()

x = np.arange(0, 1000)
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_1/learning_reward_no_eps_change.png", save=True)
get_animation(env, agent, "figures/part_1/final_game_animation_no_eps_change.gif", t_max=200, save=True)

agent = QLearningAgentEpsReduce(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_1/learning_reward.png", save=True)
get_animation(env, agent, "figures/part_1/final_game_animation.gif", t_max=200, save=True)

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_2/learning_reward_no_reset.png", save=True)
get_animation(env, agent, "figures/part_2/final_game_animation_no_reset.gif", t_max=200, save=True)

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

agent.reset()
rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_2/learning_reward_with_reset.png", save=True)
get_animation(env, agent, "figures/part_2/final_game_animation_with_reset.gif", t_max=200, save=True)

####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0
get_graph(x, rewards, "Reward per learning iteration", "Learning iteration", "Reward", "figures/part_3/learning_reward_no_eps_change.png", save=True)
get_animation(env, agent, "figures/part_3/final_game_animation_no_eps_change.gif", t_max=200, save=True)