'''
APIs for working with differentiable policy function approximators.
'''

import numpy as np
import tensorflow as tf

class PGNetwork:

    def __init__(self, num_assets, layers, discount_gamma=0.999):
        self.num_assets = num_assets
        self.discount_gamma = discount_gamma

        self.X, self.Y, self.params = self._build_network(num_assets, layers)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

    def forward(self, state):
        with tf.Session() as sess:
            output = sess.run([self.Y], feed_dict={self.X: state.get_features()})
        return output

    def take_gradient_step(self, paths, learing_rate=0.001):
        # cost function
        R = tf.placeholder(tf.float32, shape=[1, None])
        cost = PGNetwork._compute_cost(self.Y, R)
        optimizer = tf.train.AdamOptimizer(learning_rate=learing_rate)
        train_op = optimizer.minimize(cost)

        states, actions, advantages = self._get_observations(paths)
        with tf.Session() as sess:
            _, c = sess.run([train_op, cost], feed_dict={self.X: states, self.Y: actions, R: advantages})
        return c

    def _get_observations(self, paths):
        states = []
        actions = []
        advantages = []
        for path in paths:
            rewards = [r for (_, _, r) in path]
            discounted_rewards = self._compute_discounted_rewards(rewards, self.discount_gamma)
            for (i, (s, a, _)) in enumerate(path):
                states.append(s.get_features())
                actions.append(a)
                advantages.append(discounted_rewards[i])
        return np.stack(states, axis=1).squeeze(), np.array(actions).reshape((-1, len(actions))), np.array(advantages)

    @staticmethod
    def _compute_discounted_rewards(rewards, discount_gamma):
        num_rewards = len(rewards)
        discounted = [0] * num_rewards
        for (i, r) in enumerate(reversed(rewards)):
            future_rewards = discounted[max(i-1, 0)]
            discounted[i] = r + discount_gamma * future_rewards
        discounted.reverse()
        return discounted

    @staticmethod
    def _build_network(num_assets, layers):
        L = len(layers)

        # input layer
        X_n = layers[0]
        X = tf.placeholder(tf.float32, shape=[X_n, None], name='X')

        # parameters
        params = {}
        A = X
        for l in range(1, L):
            W_name = 'W' + str(l)
            b_name = 'b' + str(l)
            W = tf.get_variable(W_name, shape=[layers[l], layers[l-1]], dtype=tf.float32)
            b = tf.get_variable(b_name, shape=[layers[l], 1], dtype=tf.float32)
            params[W_name] = W
            params[b_name] = b

            Z = tf.matmul(W, A) + b
            A = tf.nn.relu(Z)

        # final layer
        Y_n = 2 * num_assets + 1
        W = tf.get_variable('W' + str(L), shape=[Y_n, layers[-1]], dtype=tf.float32)
        b = tf.get_variable('b' + str(L), shape=[Y_n, 1], dtype=tf.float32)
        Z = tf.matmul(W, A) + b

        # output layer
        Y = tf.stack([tf.nn.sigmoid(Z[0:num_assets, :]), tf.nn.softmax(Z[num_assets:, :])])

        return X, Y, params

    @staticmethod
    def _compute_cost(Y, R):
        return tf.reduce_sum(tf.multiply(-R, tf.log(Y)))
