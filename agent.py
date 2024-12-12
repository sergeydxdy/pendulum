import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError
from pendulum import Scene
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.01
CLIP_EPSILON = 0.5
BATCH_SIZE = 128
EPOCHS = 10
GAMMA = 0.99


class PPOAgent:
    def __init__(self, path=False, mode='train'):
        if not path:
            self.policy_model = self.create_policy_model()
            self.value_model = self.create_value_model()
        else:
            self.policy_model = tf.keras.models.load_model(path + '_policy')
            self.value_model = tf.keras.models.load_model(path + '_value')

        self.gamma = GAMMA
        self.epsilon = 0.1
        self.memory = []
        self.mode = mode
        self.n_games = 0

    def create_policy_model(self):
        with tf.device('/GPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(input_shape=(4,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def create_value_model(self):
        with tf.device('/GPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(input_shape=(4,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def get_state(self, env):

        return np.array(env.get_state(), dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        discounted_rewards = []
        cumulative = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            if done:
                cumulative = 0
            cumulative = reward + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)

        next_values = self.value_model.predict(next_states, verbose=0).flatten()
        target_values = rewards + (1 - dones) * self.gamma * next_values

        discounted_rewards = np.array(discounted_rewards)

        values = self.value_model.predict(states, verbose=0).flatten()
        advantages = discounted_rewards - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        for _ in range(EPOCHS):
            with tf.GradientTape() as tape:
                old_actions = self.policy_model(states, training=True)

                epsilon = 1e-16
                log_old_actions = tf.math.log(tf.cast(old_actions, tf.float32)) + epsilon
                log_actions = tf.math.log(tf.cast(actions, tf.float32)) + epsilon
                ratio = tf.exp(tf.clip_by_value(log_actions - log_old_actions, -1000, 1000))
                clip_ratio = tf.clip_by_value(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
                loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_ratio * advantages))

            grads = tape.gradient(loss, self.policy_model.trainable_variables)
            grads = [tf.clip_by_value(grad, -10, 10) for grad in grads]
            self.policy_model.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

        self.value_model.fit(states, target_values, epochs=EPOCHS, verbose=0)

        self.memory = []

    def get_action(self, state):
        state = state.reshape((1, -1))
        action = self.policy_model.predict(state, verbose=0)[0][0]

        if np.isnan(action):
            action = 0

        print('Action', round(action, 4))
        return action

    def get_reward(self, env):
        return env.reward()

    def save(self, filename):
        self.policy_model.save(filename + '_policy.h5')
        self.value_model.save(filename + '_value.h5')


def train():
    agent = PPOAgent()
    env = Scene(agent='ai')
    best_score = 0
    score = 0
    time = 0
    time_limit = 100

    while True:

        if time == 0:
            print('Start')
            print('Best_score', best_score)


        state_old = agent.get_state(env)

        action = agent.get_action(state_old)

        env.platform.x_center = max(0, min(env.platform.x_center + action, env.width))

        env.platform.update_position()

        env.next_frame()

        reward = agent.get_reward(env)

        score += reward
        state_new = agent.get_state(env)

        done = score > 100
        agent.remember(state_old, action, reward, state_new, done)
        agent.train()

        if done:
            print('Train')
            agent.train()
            time = 0
            score = 0
            env.reset()

            if score > best_score:
                best_score = score
            agent.save(f'models/ppo_agent_{agent.n_games}.h5')

            print('Best_score', best_score)

            time_limit *= 1.1

        time += 1


if __name__ == '__main__':
    train()
