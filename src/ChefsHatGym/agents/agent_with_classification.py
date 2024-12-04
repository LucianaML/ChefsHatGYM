import random
import os
import pandas as pd

from collections import deque
from pickletools import optimize

from keras.src.legacy.backend import update

#from examples.tournament_room import verbose

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras import layers, models, optimizers


from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer
from ChefsHatGym.rewards.only_winning import RewardOnlyWinning
import numpy
import matplotlib.pyplot as plt

file_name = "AgentClassification_train_vsRandom.weights.h5"
file_name_phase = "AgentClassification_train_vsRandom.weights.h5"

df = pd.DataFrame({'Player':[], 'Round':[], 'N Cards in Beginning of the Round':[], 'N Cards in End of the Round':[]} | {f'{i}': [] for i in range(0,17)})
cards_discarted_match = []

class AgentClassification(ChefsHatPlayer):
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
            self.epsilon = 1.0
        else:
            self.epsilon = 0.0

        self.reward_score = []
        self.epsilon_history = []
        self.q_values_history = []
        self.loss_history = []

        #entrada da classificação
        self.last_round = 1
        #self.forescast_model = None
        self.forescast_score = [0]*4
        self.forescast_score_last = [0]*4

        #modeling DQN
        if not self.use_phase:
            self.model_dqn = self.create_model_dqn()
            self.model_tarqet_network = self.create_model_dqn()
            self.model_forescast = self.create_model_forecast()
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
        #criando inputs
        state_input = layers.Input(shape=(self.state_size,), name="State_Input")
        forecast_score = layers.Input(shape=(4,), name="forecast_score")

        # camadas do primeiro input
        x = layers.Dense(256, activation='relu')(state_input)
        y = layers.Dense(128, activation='relu')(forecast_score)

        # Concatenando os dois inputs processados
        concatenated = layers.Concatenate(name="Concatenate")([x, y])

        # Saída
        output = layers.Dense(self.action_size, activation='softmax', name="PossiblesAction")(concatenated)

        # Criando o modelo
        DQN = models.Model(inputs=[state_input, forecast_score], outputs=output)

        # Compilando o modelo
        DQN.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=MeanSquaredError(),
                    metrics=["mse"])

        return DQN


    def create_model_forecast(self):

        # Modelo
        forescast = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(21,)),
            layers.Dense(4, activation='softmax')])

        forescast.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return forescast


    def save_exp(self, state, next_state, action, reward, done, possible_action, next_possible_action, new_input, next_new_input):
        self.memory.append((state, action, reward, next_state, done, possible_action, next_possible_action, new_input, next_new_input))

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
        new_input = np.expand_dims(np.array(self.forescast_score), 0)

        if random_int > self.epsilon:
            possible_actions_vector = np.expand_dims(possibleActions, 0)

            #fazendo previsão com as duas entradas
            predicted_q_values = self.model_dqn([state_vector, new_input])[0]

            valid_q_values = predicted_q_values * possible_actions_vector

            # Escolhendo a melhor ação válida
            a = np.argmax(valid_q_values)
            print("=============== action ==============")
            print(a)
            #a = self.model_dqn([state_vector, new_input])[0]

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
            last_score = self.forescast_score_last

            possible_actions = np.array(obs[28:])

            next_state = np.array(next_obs[0:28])
            forescast_next_state = self.forescast_score

            next_possible_actions = np.array(next_obs[28:])

            action = envInfo["Action_Index"]
            player = envInfo["Author_Index"]
            done_player, done_value = list(envInfo["Finished_Players"].items())[player]
            reward = self.get_reward(envInfo)

            self.save_exp(state, next_state, action, reward, done_value, possible_actions, next_possible_actions, last_score, forescast_next_state)
        self.update_foracast(envInfo)

    def update_action_others(self, envInfo):
        self.update_foracast(envInfo)

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
                states, action, reward, next_states, done, possible_actions, next_possible_actions, new_input, next_new_input = zip(*minibatch)
                states = np.array(states)
                new_input = np.array(new_input)
                next_states = np.array(next_states)
                next_new_input = np.array(next_new_input)

                q_values = self.model_dqn([states, new_input]).numpy()
                next_q_values = self.model_dqn([next_states, next_new_input]).numpy()

                q_targ_values = self.model_tarqet_network([next_states, next_new_input]).numpy()

                for i in range(states.shape[0]):

                    if done[i]:
                        q_values[i, action[i]] = reward[i]
                    else :
                        next_best_action = np.argmax(next_q_values[i, :])
                        q_values[i, action[i]] = reward[i] + self.gamma * q_targ_values[i, next_best_action]

                    score_reward = q_values

                self.q_values_history.append(q_values.mean())
                possible_actions = np.array(possible_actions)

                self.history = self.model_dqn.fit([states, new_input], q_values, verbose=False)
                self.loss_history.append(self.history.history['loss'][0])

                self.update_target_network()
                self.save_model_dql()

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                self.reward_score.append(score_reward)
                self.epsilon_history.append(self.epsilon)

    def update_start_match(self, cards: list[float], players: list[str], starting_player: int):

        df = pd.DataFrame({'Player':[], 'Round':[], 'N Cards in Beginning of the Round':[], 'N Cards in End of the Round':[]} | {f'{i}': [] for i in range(0,17)})
        cards_discarted_match = []

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

    def update_foracast(self, envInfo):

        action = envInfo["Action_Decoded"]
        player = envInfo["Author_Index"]
        r = envInfo["Rounds"]

        if action != 'pass':
            aux = action.replace('C', '').replace('Q', '').replace('J', '').split(';')
            cards_action = [int(aux[0])] * int(aux[1])
            if int(aux[2]) == 1:
                cards_action.append(12)
        else:
            cards_action = []

        line = df[(df['Player'] == player) & (df['Round'] == r)]

        if len(line) == 0:
            if r == 1:
                cards_probability = [0] * 17
            else:
                print("--------------- ")
                # Configurando para exibir todas as linhas e colunas
                pd.set_option('display.max_columns', None)  # Mostra todas as colunas

                # Printando o DataFrame completo
                df.to_csv('df.csv')
                print(r)
                line_r = df[(df['Player'] == player) & (df['Round'] == r-1)]
                print("--------------- LineR ----------\n", line_r)
                cards_probability = [float(line_r[f'{i}'].iloc[0]) for i in range(0, 17)]
        else:
            cards_probability = [float(line[f'{i}'].iloc[0]) for i in range(0, 17)]

        n_begin = len([c for c in cards_probability if c == 0])

        print("===========", cards_discarted_match)

        for card in cards_action:
            n_cards_discarted = len([c for c in cards_discarted_match if c == card])
            if card != 12:
                n_cards = card - n_cards_discarted
            else:
                n_cards = 2 - n_cards_discarted

            cards_probability[card - 1] = n_cards / (4 * 17 - len(cards_discarted_match)) if len(cards_discarted_match) < (4*17) else 1
            cards_discarted_match.append(card)


        n_end = n_begin - len(cards_action)
        # 'Player': [], 'Round': [], 'N Cards in Beginning of the Round': [], 'N Cards in End of the Round': []
        if len(line) == 0:
            df.loc[len(df)] = [player, r, n_begin, n_end] + cards_probability
        else:
            df.loc[line.index[0]] = [player, r, n_begin, n_end] + cards_probability

        if self.last_round != r:

            loadFrom = os.path.join(os.getcwd(), "Trained", 'model_2.h5')
            try:
                self.forescast_model = tf.keras.models.load_model(loadFrom)
                print(self.forescast_model.summary())
            except Exception as e:
                print(f'The erro is {e}')


            df_filter = df[df['Round'] == self.last_round].sort_values(by='Player')
            self.forescast_score_last = self.forescast_score
            self.forescast_score = np.argmax(self.model_forescast.predict(df_filter), axis=1)

            self.last_round = r