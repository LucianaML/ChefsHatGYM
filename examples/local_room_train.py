from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.agents.spectator_logger import SpectatorLogger
from ChefsHatGym.agents.dqn_agent import DQNAgent
from ChefsHatGym.agents.agent_with_classification import AgentClassification
from ChefsHatGym.agents.classic.dql import AgentDQL
from ChefsHatGym.agents.classic.ppo import AgentPPO

# Room parameters
room_name = "train_vsRandom"
timeout_player_response = 5


# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 10 #300  matches usados para treinar pela ultima vez!! treino 18
maxRounds = -1

# Logging information
verbose_console = True
verbose_log = True
game_verbose_console = True
game_verbose_log = True
save_dataset = True


# Start the room
room = ChefsHatRoomLocal(
    room_name,
    timeout_player_response=timeout_player_response,
    game_type=game_type,
    stop_criteria=stop_criteria,
    max_rounds=maxRounds,
    verbose_console=verbose_console,
    verbose_log=verbose_log,
    game_verbose_console=game_verbose_console,
    game_verbose_log=game_verbose_log,
    save_dataset=save_dataset,
)


# Create agents config
logDirectory = room.get_log_directory()
agentVerbose = True

# Create players
p1 = DQNAgent(name="01", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
p2 = AgentClassification(name="02", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
p3 = DQNAgent(name="03", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
p4 = DQNAgent(name="04", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
#p2 = AgentClassification(name="02", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
#p3 = AgentClassification(name="03", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
#p4 = AgentClassification(name="04", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=True, use_phase=False)
#p2 = AgentDQL(name="02", log_directory=logDirectory, verbose_log=agentVerbose, agentType="vsEveryone", continueTraining=False)
#p3 = AgentPPO(name="03", log_directory=logDirectory, verbose_log=agentVerbose, agentType="vsEveryone", continueTraining=False, saveFolder="")
#p2 = FirstAgent(name="02", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=False)
#p3 = FirstAgent(name="03", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=False)
#p4 = FirstAgent(name="04", log_directory=logDirectory, verbose_log=agentVerbose, continue_training=False)
#p1 = AgentRandon(name="01", log_directory=logDirectory, verbose_log=agentVerbose)
# p2 = AgentRandon(name="02", log_directory=logDirectory, verbose_log=agentVerbose)
# p3 = AgentRandon(name="03", log_directory=logDirectory, verbose_log=agentVerbose)
# p4 = AgentRandon(name="04", log_directory=logDirectory, verbose_log=agentVerbose)

# Adding players to the room
for p in [p1, p2, p3, p4]:
    room.add_player(p)


# Create spectators
s1 = SpectatorLogger(name="01", log_directory=logDirectory, verbose_log=agentVerbose)
s2 = SpectatorLogger(name="02", log_directory=logDirectory, verbose_log=agentVerbose)

# Adding players to the room
for s in [s1, s2]:
    room.add_spectator(s)


# Start the game

info = room.start_new_game()

print(f"Performance score: {info['Game_Performance_Score']}")
print(f"Scores: {info['Game_Score']}")
