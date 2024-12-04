import random
import os

from collections import deque
from pickletools import optimize

#from examples.tournament_room import verbose

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError


from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
import numpy
import matplotlib.pyplot as plt

file_name = "trained_vsRandom_F2-5.weights.h5"
file_name_phase = "trained_vsRandom_F2-4.weights.h5"

class DQNAgent(ChefsHatPlayer):
    suffix = "PLAYER"
    training = False

    def __init__(self, name, this_agent_folder: str = "", verbose_console: bool = False, verbose_log: bool = False, log_directory: str = "", continue_training: bool = False, load_network: str = "", use_phase: bool = False):

        super().__init__(self.suffix, name, this_agent_folder, verbose_console, verbose_log, log_directory)

        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.988
        self.learning_rate = 0.00001
        self.action_size = 200
        self.state_size = 28
        self.batchSize = 10
        self.training = continue_training
        self.load_network = load_network
        self.use_phase = use_phase
        self.hiddenLayers = 1
        self.tau = 0.52
        self.history = ''

        if self.training:
            self.epsilon = 0.6
        else:
            self.epsilon = 0.0

        self.reward_score = []
        self.epsilon_history = []
        self.q_values_history = []
        self.loss_history = []

        #modeling DQN
        if not self.use_phase:
            self.model_dqn = self.create_model_dqn()
            self.model_tarqet_network = self.create_model_dqn()
        else:
            loadFrom = os.path.join(os.getcwd(), "Trained", file_name_phase)
            try:
                self.model_dqn = self.create_model_dqn()
                self.model_tarqet_network = self.create_model_dqn()
                self.load_model_dql(loadFrom)
            except Exception as e:
                print(f'The erro is {e}')

            optimizer = optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.model_dqn.compile(optimizer, loss=MeanSquaredError(), metrics=["mse"])
            self.model_tarqet_network.compile(optimizer, loss=MeanSquaredError(), metrics=["mse"])

        print(self.model_dqn.summary())
        self.reward = RewardOnlyWinning()

        if not self.training:
            loadFrom = os.path.join(os.getcwd(), "Trained", file_name)
            try:
                self.load_model_dql(loadFrom)
            except Exception as e:
                print(f'The erro is {e}')

    def create_model_dqn(self):
        layers = tf.keras.layers
        DQN = tf.keras.models.Sequential()
        DQN.add(layers.Dense(256, input_dim=self.state_size, activation='relu', name='State'))
        for i in range(self.hiddenLayers + 1):
            DQN.add(layers.Dense(
                256 * (i + 1),
                activation='relu',
                name=f"Dense{i}"
            ))
        DQN.add(layers.Dense(self.action_size, activation='softmax', name="PossiblesAction"))
        DQN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=MeanSquaredError(), metrics=["mse"])
        return DQN

    def save_exp(self, state, next_state, action, reward, done, possible_action, next_possible_action):
        self.memory.append((state, action, reward, next_state, done, possible_action, next_possible_action))

    def save_model_dql(self):
        dir_path = os.path.join(os.getcwd(), "Trained")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.model_dqn.save_weights(os.path.join(dir_path, file_name))

    def load_model_dql(self, model):
        self.model_dqn = tf.keras.models.load_weights(model)
        self.model_tarqet_network = tf.keras.models.load_weights(model)

    def get_action(self, observations):
        random_int = random.uniform(0,1)

        state_vector = np.expand_dims(np.array(observations[0:28]),0)
        possibleActions = np.array(observations[28:])

        print(possibleActions)

        if random_int > self.epsilon:
            possible_actions_vector = np.expand_dims(possibleActions, 0)
            a = self.model_dqn([state_vector, possible_actions_vector])[0]
        else:
            itemindex = numpy.array(numpy.where(possibleActions == 1))[0].tolist()
            random.shuffle(itemindex)
            aIndex = itemindex[0]
            a = numpy.zeros(200)
            a[aIndex] = 1

        self.log("---------------------------------")
        self.log("        I did an action!")
        self.log("---------------------------------")
        self.log(f"Cards in the board: {observations[0:11]}")
        self.log(f"Cards at hand: {observations[11:28]}")
        self.log(f"Possible Actions: {observations[28:]}")
        self.log(f"Chosen Action: {a}")

        return np.array(a)

    def get_exhanged_cards(self, cards, amount):
        selectedCards = sorted(cards[-amount:])

        self.log("---------------------------------")
        self.log("        I choose cards to exchange!")
        self.log("---------------------------------")
        self.log(f"Cards in my hand: {cards}")
        self.log(f"I need to select: {amount} cards")
        self.log(f"My choice: {selectedCards} ")

        return selectedCards

    def update_exchange_cards(self, cards_sent, cards_received):
        self.log("---------------------------------")
        self.log("        I did a card exchange!")
        self.log("---------------------------------")
        self.log(f"I Received: {cards_received}")
        self.log(f"I sent: {cards_sent} cards")

    def do_special_action(self, info, specialAction):
        return True

    def update_my_action(self, envInfo):
        if self.training:
            obs = envInfo["Observation_Before"]
            next_obs = envInfo["Observation_After"]

            state = np.array(obs[0:28])
            possible_actions = np.array(obs[28:])

            next_state = np.array(next_obs[0:28])
            next_possible_actions = np.array(next_obs[28:])

            action = envInfo["Action_Index"]
            player = envInfo["Author_Index"]
            done_player, done_value = list(envInfo["Finished_Players"].items())[player]
            reward = self.get_reward(envInfo)

            self.save_exp(state, next_state, action, reward, done_value, possible_actions, next_possible_actions)

    def update_action_others(self, envInfo):
        pass

    def update_target_network(self):
        weights_main = self.model_dqn.get_weights()
        weights_target = self.model_tarqet_network.get_weights()

        update_weights = [self.tau * w_main + (1 - self.tau) * w_target
                          for w_main, w_target in zip(weights_main, weights_target)]

        self.model_tarqet_network.set_weights(update_weights)

    def update_end_match(self, envInfo):

        if self.training:

            score_reward = ''
            minibatch = np.array(random.sample(self.memory, self.batchSize), dtype=object)

            if len(self.memory) > self.batchSize:
                states, action, reward, next_states, done, possible_actions, next_possible_actions = zip(*minibatch)
                states = np.array(states)
                next_states = np.array(next_states)

                q_values = self.model_dqn([states, possible_actions]).numpy()
                next_q_values = self.model_dqn([next_states, next_possible_actions]).numpy()

                q_targ_values = self.model_tarqet_network([next_states, next_possible_actions]).numpy()

                for i in range(states.shape[0]):

                    if done[i]:
                        q_values[i, action[i]] = reward[i]
                    else :
                        next_best_action = np.argmax(next_q_values[i, :])
                        q_values[i, action[i]] = reward[i] + self.gamma * q_targ_values[i, next_best_action]

                    score_reward = q_values


                self.q_values_history.append(q_values.mean())
                possible_actions = np.array(possible_actions)

                self.history = self.model_dqn.fit([states, possible_actions], q_values, verbose=False)
                self.loss_history.append(self.history.history['loss'][0])

                self.update_target_network()
                self.save_model_dql()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                self.reward_score.append(score_reward)
                self.epsilon_history.append(self.epsilon)

    def update_start_match(self, cards: list[float], players: list[str], starting_player: int):
        pass

    def get_reward(self, envInfo):
        roles = {"Chef": 0, "Souschef": 1, "Waiter": 2, "Dishwasher": 3}
        this_player = envInfo["Author_Index"]
        this_player_position_key, this_player_position_value = list(envInfo["Match_Score"].items())[this_player]
        this_player_finished = this_player_position_key in envInfo["Finished_Players"]

        this_player_role = envInfo["Current_Roles"][this_player_position_key]

        try:
            this_player_position = roles[this_player_role]
        except:
            # if envInfo["Is_Pizza"] and envInfo["Pizza_Author"] == this_player_position_value:
            #     this_player_position = 0
            # else:
            this_player_position = 3

        reward = self.reward.getReward(this_player_position, True)

        self.log(f"Finishing position: {this_player_role} - Reward: {reward}")

        return reward

    def plot_train(self):

        loss_values = self.history.history['loss']
        val_loss_values = self.history.history.get('val_loss')
        epochs = range(1 , len (self.loss_history) + 1)

        fig, ax1 = plt.subplots(figsize=(10,6))

        ax1.set_xlabel('Matches')
        ax1.set_ylabel('Loss', color='blue')
        ax1.plot(epochs, self.loss_history, label='Loss per Matches', color='blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color='green')
        ax2.plot(epochs, self.epsilon_history, label=f'epsilon decay {self.epsilon_decay}', color='green', linestyle='dashed')
        ax2.tick_params(axis='y', labelcolor='green')


        plt.title("Loss and decay epsilon in train")
        fig.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), "trained", "plot_image", f"plot_mse.png"))
        plt.close()

    def plot_score_and_epsilon(self, matches):

        x = [i+1 for i in range(matches)]

        # plt.plot(x, self.reward_score)
        # plt.xlabel("Matches")
        # plt.ylabel("Rewards score")
        # plt.legend()
        # plt.title("Learning")
        # plt.savefig(os.path.join(os.getcwd(), "trained", "plot_image", "plot_score_rewards.png"))
        # plt.close()

        plt.plot(x, self.epsilon_history, label=f'epsilon decay {self.epsilon_decay}')
        plt.plot()
        plt.xlabel("Matches")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.title("Epsilon Decay")
        plt.grid()
        plt.savefig(os.path.join(os.getcwd(), "trained", "plot_image", "plot_epsilon_decay.png"))
        plt.close()

    def plot_q_value(self):

        epochs = range(1, len(self.q_values_history) + 1)
        plt.plot(epochs, self.q_values_history, label=f'Q-Values', marker='o')
        plt.title("Q-values in training")
        plt.xlabel("Matches")
        plt.ylabel("Q-Value")
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), "trained", "plot_image", f"plot_q_values.png"))
        plt.close()

