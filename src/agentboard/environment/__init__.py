# from agentboard.environment.webshop_env import Webshop
# from agentboard.environment.babyai_env import BabyAI
# from agentboard.environment.jericho_env import Jericho
# from agentboard.environment.pddl_env.pddl_env import PDDL
# from agentboard.environment.academia_env import AcademiaEnv
# from agentboard.environment.movie_env import MovieEnv
# from agentboard.environment.todo_env import TodoEnv
# from agentboard.environment.weather_env import WeatherEnv
# from agentboard.environment.sheet_env import SheetEnv
# from agentboard.environment.scienceworld_env import Scienceworld
# from agentboard.environment.alfworld.alfworld_env import AlfWorld
# from agentboard.environment.browser_env import *

from agentboard.common.registry import registry
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
        if name == 'babyai': from agentboard.environment.babyai_env import BabyAI
        if name == "academia": from agentboard.environment.academia_env import AcademiaEnv
        if name == "todo": from agentboard.environment.todo_env import TodoEnv
        if name == "jericho": from agentboard.environment.jericho_env import Jericho
        if name == "webshop": from agentboard.environment.webshop_env import Webshop
        if name == "alfworld": from agentboard.environment.alfworld.alfworld_env import AlfWorld
        if name == "scienceworld": from agentboard.environment.scienceworld_env import Scienceworld
        if name == "movie": from agentboard.environment.movie_env import MovieEnv
        if name == "weather": from agentboard.environment.weather_env import WeatherEnv
        if name == "pddl": from agentboard.environment.pddl_env.pddl_env import PDDL
        if name == "sheet": from agentboard.environment.sheet_env import SheetEnv
        if name == "BrowserEnv": from agentboard.environment.browser_env.envs import ScriptBrowserEnv

    
    env = registry.get_environment_class(name).from_config(config)

    return env
