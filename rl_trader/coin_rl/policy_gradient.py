'''
APIs for working with differentiable policy function approximators.
'''

import numpy as np
import tensorflow as tf

EPSILON = 1e-10

class PGNetwork:

    def __init__(self, num_assets, layers, discount_gamma=0.999):
        self.num_assets = num_assets
        self.discount_gamma = discount_gamma

        self.X, self.Y_sell, self.Y_buy, self.target_sell, self.target_buy, self.R, self.U, self.train_op = _build_network(num_assets, layers)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def forward(self, state):
        sell, buy = self.sess.run([self.Y_sell, self.Y_buy], feed_dict={self.X: state.get_features()})
        return (sell, buy)

    def take_gradient_step(self, paths):
        states, sell_actions, buy_actions, advantages = self._get_observations(paths)
        num_observations = states.shape[1]
        print("using {} observations".format(num_observations))
        _, score = self.sess.run([self.train_op, self.U],
                feed_dict={self.X: states, self.target_sell: sell_actions, self.target_buy: buy_actions, self.R: advantages})
        return score/num_observations

    def _get_observations(self, paths):
        states = []
        sell_actions = []
        buy_actions = []
        advantages = []
        for path in paths:
            rewards = [r for (_, _, r) in path]
            discounted_rewards = _compute_discounted_rewards(rewards, self.discount_gamma)
            for (i, (s, a, _)) in enumerate(path):
                states.append(s.get_features())
                sell_actions.append(a[0])
                buy_actions.append(a[1])
                advantages.append(discounted_rewards[i])

        m = len(states)
        states_tensor = np.stack(states, axis=1).reshape((-1, m))
        sell_actions_tensor = np.array(sell_actions).reshape((-1, m))
        buy_actions_tensor = np.array(buy_actions).reshape((-1, m))
        advantages_tensor = np.array(advantages).reshape((1, m))
        return states_tensor, sell_actions_tensor, buy_actions_tensor, advantages_tensor

def _compute_discounted_rewards(rewards, discount_gamma):
    num_rewards = len(rewards)
    discounted = [0] * num_rewards
    for (i, r) in enumerate(reversed(rewards)):
        future_rewards = discounted[max(i-1, 0)]
        discounted[i] = r + discount_gamma * future_rewards
    discounted.reverse()
    return discounted

def _build_network(num_assets, layers):
    L = len(layers)

    # input layer
    X_n = layers[0]
    X = tf.placeholder(tf.float32, shape=[X_n, None], name='X')

    # hidden layers
    A = X
    for l in range(1, L):
        W = tf.get_variable('w' + str(l), shape=[layers[l], layers[l-1]], dtype=tf.float32)
        b = tf.get_variable('b' + str(l), shape=[layers[l], 1], dtype=tf.float32)

        Z = tf.matmul(W, A) + b
        A = tf.nn.relu(Z)

    # final layer
    Y_n = 2 * num_assets + 1
    W = tf.get_variable('W' + str(L), shape=[Y_n, layers[-1]], dtype=tf.float32)
    b = tf.get_variable('b' + str(L), shape=[Y_n, 1], dtype=tf.float32)
    Z = tf.matmul(W, A) + b

    # output layer
    Y_sell = tf.nn.sigmoid(Z[0:num_assets, :])
    Y_buy = tf.nn.softmax(Z[num_assets:, :])

    # cost function
    target_sell = tf.placeholder(tf.float32, shape=Y_sell.shape)
    target_buy = tf.placeholder(tf.float32, shape=Y_buy.shape)
    R = tf.placeholder(tf.float32, shape=[1, None])
    U = _compute_log_utility(Y_sell, Y_buy, target_sell, target_buy, R)
    cost = -U
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(cost)

    return X, Y_sell, Y_buy, target_sell, target_buy, R, U, train_op

def _compute_log_utility(Y_sell, Y_buy, target_sell, target_buy, R):
    log_sell_probs = _log_sell_probs(Y_sell, target_sell)
    log_buy_probs = _log_buy_probs(Y_buy, target_buy)
    assert tf.shape(log_sell_probs) == tf.shape(log_buy_probs)

    log_target_probs = log_sell_probs + log_buy_probs
    advantage_weighted_log_probs = tf.multiply(R, log_target_probs)
    log_utility = tf.reduce_mean(tf.reduce_sum(advantage_weighted_log_probs, axis=0))
    return log_utility

def _log_sell_probs(Y, target):
    target_probs = tf.multiply(target, Y) + tf.multiply((1-target), (1-Y))
    log_probs = tf.log(target_probs + EPSILON)
    return log_probs

def _log_buy_probs(Y, target):
    target_probs = tf.reduce_sum(tf.multiply(target, Y), axis=0, keepdims=True)
    log_probs = tf.log(target_probs + EPSILON)
    return log_probs
