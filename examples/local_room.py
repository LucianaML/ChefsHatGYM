from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.agents.spectator_logger import SpectatorLogger

# Room parameters
room_name = "Testing_2_Local"
timeout_player_response = 5


# Game parameters
game_type = ChefsHatEnv.GAMETYPE["MATCHES"]
stop_criteria = 1
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
p1 = AgentRandon(name="01", log_directory=logDirectory, verbose_log=agentVerbose)
p2 = AgentRandon(name="02", log_directory=logDirectory, verbose_log=agentVerbose)
p3 = AgentRandon(name="03", log_directory=logDirectory, verbose_log=agentVerbose)
p4 = AgentRandon(name="04", log_directory=logDirectory, verbose_log=agentVerbose)

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
