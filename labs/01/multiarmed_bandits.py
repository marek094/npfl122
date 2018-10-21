#!/usr/bin/env python3
import numpy as np

class MultiArmedBandits():
    def __init__(self, bandits, episode_length):
        self._bandits = []
        for _ in range(bandits):
            self._bandits.append(np.random.normal(0., 1.))
        self._done = True
        self._episode_length = episode_length
        print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

    def reset(self):
        self._done = False
        self._trials = 0
        return None

    def step(self, action):
        if self._done:
            raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
        self._trials += 1
        self._done = self._trials == self._episode_length
        reward = np.random.normal(self._bandits[action], 1.)
        return None, reward, self._done, {}


class Player():
    def __init__(self, k):
        self.q = np.zeros(k)
        self.n = np.zeros(k)

class GreedyPlayer(Player):
    def __init__(self, k, eps):
        super().__init__(k)
        self.eps = eps

    def choose_action(self):
        if np.random.uniform() < self.eps:
            return np.random.randint(self.q.size)
        return np.argmax(self.q)

    def feedback(self, action, reward):
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]

    def get_param(self):
        return self.eps

class BiasedGreedyPlayer(GreedyPlayer):
    def __init__(self, k, eps, alpha):
        super().__init__(k, eps)
        self.alpha = alpha
    
    def feedback(self, action, reward):
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) * self.alpha

class InitBiasedGreedyPlayer(BiasedGreedyPlayer):
    def __init__(self, k, eps, alpha, initial):
        super().__init__(k, eps, alpha)
        self.q = np.repeat(initial, k)

class UcbPlayer(GreedyPlayer):
    def __init__(self, k, eps, c):
        super().__init__(k, eps)
        self.c = c
        self.n = np.repeat(1/k, k)

    def choose_action(self):
        return np.argmax( self.q + self.c * np.sqrt( np.log(np.sum(self.n)) / self.n) )

    def get_param(self):
        return self.c

class GradientPlayer():
    def __init__(self, k, alpha):
        self.alpha = alpha
        self.h = np.zeros(k)

    def choose_action(self):
        exp = np.exp(self.h)
        self.distr = exp / np.sum(exp)
        action = np.random.choice(self.h.size, 1, p=self.distr)[0]
        return action

    def feedback(self, action, reward):
        ha = self.h[action]
        self.h -= self.alpha * reward * self.distr
        self.h[action] = ha + self.alpha * reward * (1 - self.distr[action])

    def get_param(self):
        return self.alpha


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
    parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
    parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
    parser.add_argument("--c", default=1., type=float, help="Confidence level in UCB.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
    parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.") #Q
    args = parser.parse_args()

    env = MultiArmedBandits(args.bandits, args.episode_length)
    reward_sum = 0
    count = 0
    for episode in range(args.episodes):
        print(f'{episode+1}/{args.episodes}', end='\r')
        env.reset()

        if args.mode == 'greedy':
            if args.alpha == 0:
                player = GreedyPlayer(args.bandits, args.epsilon)
            elif args.initial == 0:
                player = BiasedGreedyPlayer(args.bandits, args.epsilon, args.alpha)
            else:
                player = InitBiasedGreedyPlayer(args.bandits, args.epsilon, args.alpha, args.initial)
        elif args.mode == 'ucb':
            player = UcbPlayer(args.bandits, args.epsilon, args.c)
        elif args.mode == 'gradient':
            player = GradientPlayer(args.bandits, args.alpha)
        else:
            raise Exception('Unknown --mode ' + args.mode)

        done = False
        while not done:
            action = player.choose_action()
            _, reward, done, _ = env.step(action)
            player.feedback(action, reward)
            # print(action, reward)
            reward_sum += reward 
            count += 1
    print()
    print(f'$\t{player.__class__.__name__}\t{player.get_param()}\t{reward_sum/count}\t{vars(args)}')