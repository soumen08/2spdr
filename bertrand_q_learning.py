import numpy as np
import itertools as it
import math
import random
class QLearningAgent:
    def __init__(self, alpha = 0.15, beta = 4e-6, discount=0.95):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = math.exp(-1*self.beta)
        self.discount = discount
        p1 = np.linspace(1, 2.1, 15)
        p2 = np.linspace(1, 2.1, 15)
        self.actions = np.linspace(1, 2.1, 15)

        self.q_table = {}
        states = list(it.product(p1, p2))
        for state in states:
            for action in self.actions:
                self.q_table[(state,action)] = 0
        
        self.p_table = {}
        for state in states:
            self.p_table[(state)] = np.random.choice(self.actions)
        
        self.conv_counter = 0
    
    def get_q_value(self, state, action):
        return self.q_table[(state, action)]
        
    def has_converged(self):
        if self.conv_counter > 100000:
            return True
        return False
    
    def update(self, state, action, next_state, reward):
        q_value = self.get_q_value(state, action)
        next_max_q_value = max([self.get_q_value(next_state, next_action) for next_action in self.actions])
        new_q_value = q_value + self.alpha * (reward + self.discount * next_max_q_value - q_value)
        self.q_table[(state, action)] = new_q_value
        best_action = max([(self.get_q_value(state, action_temp), action_temp) for action_temp in self.actions])[1]
        if best_action == self.p_table[(state)]:
            self.conv_counter = self.conv_counter + 1
        else:
            self.conv_counter = 0
            self.p_table[(state)] = best_action
        self.epsilon = self.epsilon*math.exp(-1*self.beta)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max([(self.get_q_value(state, action), action) for action in self.actions])[1]
    def get_best_action(self, state):
        return max([(self.get_q_value(state, action), action) for action in self.actions])[1]

if __name__ == '__main__':
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()
    
    p1 = np.linspace(1, 2.1, 15)
    p2 = np.linspace(1, 2.1, 15)
    states = list(it.product(p1, p2))
    state = random.choice(states)
    for i in range(1500000):
        p1 = agent1.get_action(state)
        p2 = agent2.get_action(state)
        next_state = (p1,p2)
        a0 = 0
        a1 = 2
        a2 = 2
        c1 = 1
        c2 = 1
        mu = 0.25
        d1 = math.exp((a1-p1)/mu)/(math.exp((a1-p1)/mu) + math.exp((a2-p2)/mu) + math.exp(a0/mu))
        d2 = math.exp((a2-p2)/mu)/(math.exp((a1-p1)/mu) + math.exp((a2-p2)/mu) + math.exp(a0/mu))
        pi1 = d1*(p1-c1)
        pi2 = d2*(p2-c2)
        agent1.update(state,p1,next_state,pi1)
        agent2.update(state,p2,next_state,pi2)
        state = next_state
        if agent1.has_converged() and agent2.has_converged():
            print(i)
            break
    print(agent1.has_converged())
    print(agent2.has_converged())
    out_file = open("prices.txt","w")
    p1 = np.linspace(1, 2.1, 15)
    p2 = np.linspace(1, 2.1, 15)
    state = (p1[10],p2[10])
    for i in range(15):
        p1 = agent1.get_best_action(state)
        p2 = agent2.get_best_action(state)
        state = (p1,p2)
        out_file.write(str(p1) + " " + str(p2) + "\n")
    out_file.close()