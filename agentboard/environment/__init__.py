# from AgentBoard.agentboard.environment.webshop_env import Webshop
# from AgentBoard.agentboard.environment.babyai_env import BabyAI
# from AgentBoard.agentboard.environment.jericho_env import Jericho
# from AgentBoard.agentboard.environment.pddl_env.pddl_env import PDDL
# from AgentBoard.agentboard.environment.academia_env import AcademiaEnv
# from AgentBoard.agentboard.environment.movie_env import MovieEnv
# from AgentBoard.agentboard.environment.todo_env import TodoEnv
# from AgentBoard.agentboard.environment.weather_env import WeatherEnv
# from AgentBoard.agentboard.environment.sheet_env import SheetEnv
# from AgentBoard.agentboard.environment.scienceworld_env import Scienceworld
# from AgentBoard.agentboard.environment.alfworld.alfworld_env import AlfWorld
# from AgentBoard.agentboard.environment.browser_env import *

from AgentBoard.agentboard.common.registry import registry
import json
import os

# __all__ = [
#     "BabyAI",
#     "AlfWorld",
#     "Scienceworld",
    
#     "PDDL",
#     "Jericho",
    
#     "AcademiaEnv",
#     "MovieEnv",
#     "TodoEnv",
#     "SheetEnv",
#     "WeatherEnv",
    
#     "Webshop",
#     "BrowserEnv",
# ]


def load_environment(name, config):
    
    if name not in registry.list_environments():
        if name == 'babyai': from AgentBoard.agentboard.environment.babyai_env import BabyAI
        if name == "academia": from AgentBoard.agentboard.environment.academia_env import AcademiaEnv
        if name == "todo": from AgentBoard.agentboard.environment.todo_env import TodoEnv
        if name == "jericho": from AgentBoard.agentboard.environment.jericho_env import Jericho
        if name == "webshop": from AgentBoard.agentboard.environment.webshop_env import Webshop
        if name == "alfworld": from AgentBoard.agentboard.environment.alfworld.alfworld_env import AlfWorld
        if name == "scienceworld": from AgentBoard.agentboard.environment.scienceworld_env import Scienceworld
        if name == "movie": from AgentBoard.agentboard.environment.movie_env import MovieEnv
        if name == "weather": from AgentBoard.agentboard.environment.weather_env import WeatherEnv
        if name == "pddl": from AgentBoard.agentboard.environment.pddl_env.pddl_env import PDDL
        if name == "sheet": from AgentBoard.agentboard.environment.sheet_env import SheetEnv
        if name == "BrowserEnv": from AgentBoard.agentboard.environment.browser_env.envs import ScriptBrowserEnv

    
    env = registry.get_environment_class(name).from_config(config)

    return env

