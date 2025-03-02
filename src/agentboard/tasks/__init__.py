# from agentboard.tasks.webshop import EvalWebshop
# from agentboard.tasks.alfworld import Evalalfworld
# from agentboard.tasks.webbrowse import EvalWebBrowse
# from agentboard.tasks.babyai import EvalBabyai
# from agentboard.tasks.pddl import EvalPddl
# from agentboard.tasks.scienceworld import EvalScienceworld
# from agentboard.tasks.jericho import EvalJericho
# from agentboard.tasks.tool import EvalTool

from agentboard.common.registry import registry

# __all__ = [
#     "Evalalfworld",
#     "EvalBabyai",
#     "EvalPddl",
#     "EvalWebBrowse",
#     "EvalWebshop",
#     "EvalJericho",
#     "EvalTool",
#     "EvalWebshop",
#     "EvalScienceworld"
# ]


def load_task(name, run_config, llm_config, agent_config, env_config, llm=None, wandb_run=None):
    from agentboard.tasks.webshop import EvalWebshop
    from agentboard.tasks.alfworld import Evalalfworld
    from agentboard.tasks.webbrowse import EvalWebBrowse
    from agentboard.tasks.babyai import EvalBabyai
    from agentboard.tasks.pddl import EvalPddl
    from agentboard.tasks.scienceworld import EvalScienceworld
    from agentboard.tasks.jericho import EvalJericho
    from agentboard.tasks.tool import EvalTool
    task = registry.get_task_class(name).from_config(run_config, llm_config, agent_config, env_config, llm=llm, wandb_run=wandb_run)

    return task
