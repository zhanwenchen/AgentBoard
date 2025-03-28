import re
import jsonlines
from scienceworld import ScienceWorldEnv
from agentboard.common.registry import registry


@registry.register_environment("scienceworld")
class Scienceworld:
    def __init__(self,
                 serverPath=None,
                 envStepLimit=100,
                 label_path=''
                 ):
        self.env = ScienceWorldEnv("", serverPath, envStepLimit=envStepLimit)
        self.reward = 0.
        self.done = False
        self.label_path = label_path
        self.labels = {}
        self.cur_label = None
        self.modified_goal = ''
        self.selected_obs = ''
        self.finished_sub_goal = []
        with open(self.label_path, 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                task_name = item["additional_info"]["env_name"]
                var = item["additional_info"]["var"]
                self.labels[f"{task_name}_{var}"] = {
                    "task_name": task_name,
                    "var": var,
                    "modified_goal": item["goal"],
                    "subgoals": item['subgoals'],
                    "difficulty": item["difficulty"],
                }


    def load(self, task_name: str, var: str, simplificationStr: str):
        """Load a task from the environment.

        Args:
            task_name: The name of the task to load.
            var: The variant of the task.
            simplificationStr: Simplification string for the environment.

        Returns:
            The loaded environment.

        Raises:
            KeyError: If the task label cannot be found.
        """
        env = self.env.load(task_name, var, simplificationStr=simplificationStr)
        try:
            # Use underscore instead of hyphen to match the key format in __init__
            self.cur_label = self.labels[f"{task_name}_{var}"]
        except KeyError as e:
            # Print available keys for debugging
            print(f"Task key '{task_name}_{var}' not found in available labels.")
            print(f"Available keys: {list(self.labels.keys())}")
            raise e
            # breakpoint()
        self.selected_obs = self.cur_label["subgoals"]
        self.modified_goal = self.cur_label["modified_goal"]
        self.difficulty = self.cur_label["difficulty"]
        self.finished_sub_goal = [0 for i in range(len(self.selected_obs))]
        return env

    def inventory(self):
        return self.env.inventory()

    def parseAction(self, action):
        action = action.strip()
        return action

    def step(self, action):
        action = self.parseAction(action)
        observation = ''
        # reward = self.reward
        if action == "check valid actions":
            valid_actions = ", ".join(self.get_action_space())
            observation = f"Choose an action from these valid actions: {valid_actions}"
            return observation, self.reward, self.done, None
        else:
            observation, _, _, info = self.env.step(action)
            self._check_temperature_string(observation, self.selected_obs)
            self.reward = self.get_reward()
            self.done = self._check_is_done(self.selected_obs)
            return observation, self.reward, self.done, info

    def get_action_space(self, abstract=True):

        # print("Valid action-object combinations:")
        svalid_actions = []
        if abstract:
            for a in self.env.getPossibleActions():
                if "reset" not in a:
                    svalid_actions.append(a)
        else:
            valid_actions = self.env.getValidActionObjectCombinationsWithTemplates()
            forbidden_words = ["teleport",
                               "connect",
                               "dunk",
                               "eat",
                               "flush",
                               "close door",
                               ]
            for valid_action in valid_actions:
                v = valid_action['action']
                for fw in forbidden_words:
                    if fw in v:
                        break
                svalid_actions.append(valid_action['action'])
        if "check valid actions" not in svalid_actions:
            svalid_actions.append("check valid actions")
        return svalid_actions

    def getTaskDescription(self):
        return self.env.getTaskDescription()

    def getGoalProgressStr(self):
        return self.env.getGoalProgressStr()

    def getGoldActionSequence(self):
        return self.env.getGoldActionSequence()

    def reset(self):
        self.reward = 0.
        self.done = False
        return self.env.reset()

    def _check_temperature_string(self, s, selected_obs):
        for i, pattern in enumerate(selected_obs):
            match = re.search(pattern, s)
            if match:
                self.finished_sub_goal[i] = 1.

    def get_reward(self):
        return sum(self.finished_sub_goal) * 1.0 / len(self.finished_sub_goal)

    def _check_is_done(self, selected_obs):
        return sum(self.finished_sub_goal) >= len(selected_obs)

    def should_continue_iteration(self) -> bool:
        """Determines if there are remaining subgoals to complete in the current task.

        This method checks the progress of subgoal completion and returns whether
        the agent should continue iterating on the current task.

        Returns:
            bool: True if there are remaining subgoals to complete, False if all
                  subgoals have been completed.
        """
        if not self.selected_obs or len(self.selected_obs) == 0:
            return False

        # Check if there are any subgoals that haven't been completed yet
        incomplete_subgoals = sum(1 for completed in self.finished_sub_goal if completed == 0)
        return incomplete_subgoals > 0

    @classmethod
    def from_config(cls, cfg):
        serverPath = cfg.get("serverPath", None)
        envStepLimit = cfg.get("envStepLimit", 50)
        label_path = cfg.get("label_path", '')
        env = cls(serverPath=serverPath,
                  envStepLimit=envStepLimit,
                  label_path=label_path
                   )
        return env
