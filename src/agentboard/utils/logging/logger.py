from typing import Dict, List, Any, Union, Optional, Tuple, Set
from pathlib import Path
from json import loads as json_loads, dumps as json_dumps
import re
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from plotly.io import to_html
from plotly.graph_objects import Bar, Layout, Figure
from plotly.express import line, line_polar
from wandb import Table, Html, Plotly, Image
import jsonlines


METRICS = ["success_rate", "progress_rate", "grounding_acc", "success_rate_hard", "success_rate_easy", "progress_rate_hard", "progress_rate_easy"]


class SummaryLogger:
    """Logger for summarizing metrics across multiple tasks and comparing with baselines."""

    def __init__(self, wandb_run, log_path: Union[str, Path], baseline_dir: Union[str, Path] = "data/baseline_results"):
        """Initialize the SummaryLogger.

        Args:
            wandb_run: Weights & Biases run object for logging
            log_path: Path to store log files
            baseline_dir: Directory containing baseline results for comparison
        """
        self.dimension_scoring = {
            "Memory": {"alfworld": 1, "scienceworld": 2, "babyai": 1, "jericho": 1, "pddl": 2, "webshop": 1, "webarena": 3, "tool-query": 2, "tool-operation": 3},
            "Planning": {"alfworld": 1, "scienceworld": 2, "babyai": 2, "jericho": 3, "pddl": 3, "webshop": 2, "webarena": 3, "tool-query": 2, "tool-operation": 2},
            "World Modeling": {"alfworld": 3, "scienceworld": 3, "babyai": 2, "jericho": 3, "pddl": 1, "webshop": 1, "webarena": 3, "tool-query": 1, "tool-operation": 1},
            "Self-reflection": {"alfworld": 3, "scienceworld": 2, "babyai": 2, "jericho": 1, "pddl": 3, "webshop": 2, "webarena": 2, "tool-query": 1, "tool-operation": 1},
            "Grounding": {"alfworld": 2, "scienceworld": 3, "babyai": 2, "jericho": 1, "pddl": 3, "webshop": 3, "webarena": 3, "tool-query": 3, "tool-operation": 3},
            "Spatial Navigation": {"alfworld": 2, "scienceworld": 2, "babyai": 2, "jericho": 2, "pddl": 1, "webshop": 1, "webarena": 2, "tool-query": 1, "tool-operation": 1}
        }

        self.baseline_dir = Path(baseline_dir)
        self.current_run_metrics = []
        self.log_path = Path(log_path) / "all_results.txt"
        self.log_dimension_path = Path(log_path) / "dimension.txt"
        self.wandb_log = wandb_run.log


    def check_metric_item_is_logged(self, metric_type: str, file_path: Union[str, Path]) -> bool:
        """Check if a metric has already been logged in the specified file.

        Args:
            metric_type: The type of metric to check for
            file_path: Path to the log file

        Returns:
            True if the metric is found in the file, False otherwise
        """
        with open(file_path) as f:
            return any(metric_type in line for line in f)

    def log_run_result(self, task_name: str, success_rate: float, reward_score: float,
                       grounding_acc: float, hard_sr: float, easy_sr: float,
                       hard_pr: float, easy_pr: float) -> None:
        """Log the result of a task run.

        Args:
            task_name: Name of the task
            success_rate: Overall success rate
            reward_score: Progress/reward score
            grounding_acc: Grounding accuracy
            hard_sr: Success rate on hard problems
            easy_sr: Success rate on easy problems
            hard_pr: Progress rate on hard problems
            easy_pr: Progress rate on easy problems
        """
        result = {
            "task_name": task_name,
            "success_rate": success_rate,
            "progress_rate": reward_score,
            "grounding_acc": grounding_acc,
            "success_rate_hard": hard_sr,
            "success_rate_easy": easy_sr,
            "progress_rate_hard": hard_pr,
            "progress_rate_easy": easy_pr
        }

        self.current_run_metrics.append(result)

        if not self.check_metric_item_is_logged(task_name, self.log_path):
            with open(self.log_path, "a+") as f:
                f.write(json_dumps(result) + "\n")

    def load_baseline_results(self, task_name: str, baseline_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load baseline results for a specific task.

        Args:
            task_name: Name of the task to load results for
            baseline_dir: Directory containing baseline results (defaults to self.baseline_dir)

        Returns:
            Dictionary mapping model names to their baseline results
        """
        baseline_dir = Path(baseline_dir) if baseline_dir else self.baseline_dir
        baseline_results = {}

        for model_dir in baseline_dir.iterdir():
            if not model_dir.is_dir():
                continue

            results_file = model_dir / "all_results.txt"
            if not results_file.exists():
                continue

            with open(results_file, "r") as f:
                for line_f in f:
                    try:
                        result = json_loads(line_f.strip())
                        if result.get("task_name") == task_name:
                            baseline_results[model_dir.name] = result
                    except Exception:
                        continue

        return baseline_results

    def log_summary_metric(self) -> None:
        """Calculate and log summary metrics across all task types."""
        tasks_type = {
            "embodied": ["alfworld", "scienceworld", "babyai"],
            "game": ["jericho", "pddl"],
            "web": ["webshop", "webarena"],
            "tool": ["tool-query", "tool-operation"],
            "all": ["alfworld", "scienceworld", "babyai", "jericho", "pddl", "webshop", "webarena", "tool-query", "tool-operation"]
        }

        metrics_table = Table(columns=["Metric Name", "Metric Value (%)"])
        metrics_dict = {}

        for type_ in tasks_type:
            tasks = tasks_type[type_]
            results = [r for r in self.current_run_metrics if r["task_name"] in tasks]

            if len(results) == len(tasks):
                self._process_type_metrics(type_, results, metrics_table, metrics_dict, tasks_type)

        self.wandb_log({"summary/metrics": metrics_table})

        if "all" in metrics_dict:
            self._visualize_baseline_comparison(metrics_dict, tasks_type)

    def _process_type_metrics(self, type_: str, results: List[Dict[str, Any]],
                             metrics_table: Table, metrics_dict: Dict[str, Dict[str, float]],
                             tasks_type: Dict[str, List[str]]) -> None:
        """Process metrics for a specific task type.

        Args:
            type_: The task type (e.g., "embodied", "game", etc.)
            results: List of result dictionaries for tasks of this type
            metrics_table: The W&B table to add metrics to
            metrics_dict: Dictionary to store computed metrics
            tasks_type: Dictionary mapping task types to lists of task names
        """
        task_name = f"{type_}_summary"
        all_metrics = {'task_name': task_name}

        for metric in METRICS:
            mean_metric = np.mean([task_result[metric] for task_result in results])
            all_metrics[metric] = mean_metric
            metric_name = " ".join([word.capitalize() for word in metric.split("_")])
            metrics_table.add_data(f"Average {type_.capitalize()} {metric_name}", mean_metric)

        if not self.check_metric_item_is_logged(task_name, self.log_path):
            with open(self.log_path, "a+") as f:
                f.write(json_dumps(all_metrics) + "\n")

        success_rate = all_metrics["success_rate"]
        progress_rate = all_metrics["progress_rate"]

        metrics_table.add_data(f"Average {type_.capitalize()} Progress Rate", progress_rate)
        metrics_table.add_data(f"Average {type_.capitalize()} Success Rate", success_rate)

        metrics_dict[type_] = {"success_rate": success_rate, "progress_rate": progress_rate}

        if type_ == "all":
            self._calculate_dimension_metrics(results, tasks_type["all"])

    def _calculate_dimension_metrics(self, results: List[Dict[str, Any]], all_tasks: List[str]) -> None:
        """Calculate dimension-specific metrics.

        Args:
            results: List of result dictionaries
            all_tasks: List of all task names
        """
        dimension_metrics = {}
        for dimension in self.dimension_scoring:
            weights = self.dimension_scoring[dimension]
            weights_sum = sum(weights[task_name] for task_name in all_tasks)
            score = 0

            for task_result in results:
                task_name = task_result["task_name"]
                score += weights[task_name] * task_result["success_rate"] * 100

            score /= weights_sum
            dimension_metrics[dimension] = score

        with open(self.log_dimension_path, "w") as f:
            f.write(json_dumps(dimension_metrics))

    def _visualize_baseline_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                                      tasks_type: Dict[str, List[str]]) -> None:
        """Create visualization comparing the current run to baseline models.

        Args:
            metrics_dict: Dictionary of metrics by task type
            tasks_type: Dictionary mapping task types to lists of task names
        """
        # Calculate average performance for each baseline model
        avg_baseline_results = self._calculate_avg_baseline_results(tasks_type["all"])

        baseline_models = list(avg_baseline_results.keys())
        metrics = metrics_dict["all"]
        models = ["Current Run"] + [model_name.capitalize() for model_name in baseline_models]

        accuracies = [metrics["success_rate"]] + [avg_baseline_results[m]["success_rate"] for m in baseline_models]
        rewards = [metrics["progress_rate"]] + [avg_baseline_results[m]["progress_rate"] for m in baseline_models]

        marker_color_acc = ['rgba(0,128, 255, 1)'] + ['rgba(0,128, 255, 0.6)'] * len(baseline_models)
        marker_color_reward = ['rgba(51, 255,153, 1)'] + ['rgba(51, 255,153, 0.6)'] * len(baseline_models)

        data = [
            Bar(name='Progress Rate (%)', x=models, y=rewards, marker_color=marker_color_reward),
            Bar(name='Success Rate (%)', x=models, y=accuracies, marker_color=marker_color_acc)
        ]

        layout = Layout(
            width=800,
            height=400,
            xaxis={'categoryorder': 'total descending'},
            title='Average Metrics for All Tasks Compared to Baseline Models',
        )

        fig = Figure(data=data, layout=layout)
        self.wandb_log({"summary/avg_metrics_comparison": Plotly(fig)})

    def _calculate_avg_baseline_results(self, all_tasks: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate average baseline results across all tasks.

        Args:
            all_tasks: List of all task names

        Returns:
            Dictionary mapping model names to their average metrics
        """
        avg_baseline_results = {}

        for task_name in all_tasks:
            baseline_results = self.load_baseline_results(task_name)
            baseline_models = list(baseline_results.keys())

            for model_name in baseline_models:
                if model_name not in avg_baseline_results:
                    avg_baseline_results[model_name] = {"success_rate": [], "progress_rate": []}
                avg_baseline_results[model_name]["success_rate"].append(baseline_results[model_name]["success_rate"])
                avg_baseline_results[model_name]["progress_rate"].append(baseline_results[model_name]["progress_rate"])

        for model_name in list(avg_baseline_results.keys()):
            if len(avg_baseline_results[model_name]["success_rate"]) == len(all_tasks):
                avg_baseline_results[model_name]["success_rate"] = np.mean(avg_baseline_results[model_name]["success_rate"])
                avg_baseline_results[model_name]["progress_rate"] = np.mean(avg_baseline_results[model_name]["progress_rate"])
            else:
                del avg_baseline_results[model_name]

        return avg_baseline_results

    def log_summary(self) -> None:
        """Generate and log summary visualizations for all metrics."""
        # Log average metrics for task types
        self.log_summary_metric()

        # Get results for all tasks
        all_results = self._gather_all_results()

        # Draw task-specific radar chart
        self._create_task_radar_chart(all_results)

        # Draw dimension-based radar chart
        self._create_dimension_radar_chart(all_results)

    def _gather_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Gather results for all tasks.

        Returns:
            Dictionary mapping task names to dictionaries of model results
        """
        all_results = {}

        for task_result in self.current_run_metrics:
            task_name = task_result["task_name"]
            results_dict = {"Current Run": task_result}

            # Add baseline results
            results_dict.update(self.load_baseline_results(task_name))
            all_results[task_name] = results_dict

        # Find valid baseline models (present for all tasks)
        valid_baseline_models = self._find_valid_baseline_models(all_results)

        # Filter out models not in valid_baseline_models
        for task_name, task_value in all_results.items():
            for model_name in list(task_value.keys()):
                if model_name not in valid_baseline_models:
                    del task_value[model_name]

        return all_results

    def _find_valid_baseline_models(self, all_results: Dict[str, Dict[str, Any]]) -> Set[str]:
        """Find baseline models that have results for all tasks.

        Args:
            all_results: Dictionary mapping task names to dictionaries of model results

        Returns:
            Set of valid baseline model names
        """
        if not all_results:
            return set()

        first_task = next(iter(all_results))
        valid_models = set(all_results[first_task].keys())

        for task_name in all_results:
            valid_models &= set(all_results[task_name].keys())

        return valid_models

    def _create_task_radar_chart(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create a radar chart showing success rates for all tasks.

        Args:
            all_results: Dictionary mapping task names to dictionaries of model results
        """
        if not all_results:
            return

        task_categories = list(all_results.keys())
        result_df = DataFrame(columns=["model_name", "task_name", "success_rate", "baseline"])

        for task_name in task_categories:
            for model_name, metrics in all_results[task_name].items():
                dash = model_name != "Current Run"
                result_df.loc[len(result_df)] = {
                    "model_name": model_name,
                    "task_name": task_name,
                    "success_rate": 100 * metrics["success_rate"],
                    "baseline": dash
                }

        radar_results = line_polar(
            result_df,
            r='success_rate',
            theta='task_name',
            line_close=True,
            category_orders={"category": task_categories},
            color='model_name',
            markers=True,
            labels={'success_rate': 'Success Rate (%)', 'task_name': 'Task Name', 'model_name': 'Model Name'},
            line_dash="baseline"
        )

        radar_results.update_layout(
            width=700,
            height=400,
            title='Success Rate (%) w.r.t Tasks for All Models',
            title_x=0.1,
            legend_title_text='',
        )

        self.wandb_log({"summary/all_results": Html(to_html(radar_results))})

    def _create_dimension_radar_chart(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """Create a radar chart showing scores along different dimensions.

        Args:
            all_results: Dictionary mapping task names to dictionaries of model results
        """
        if not all_results:
            return

        task_categories = list(all_results.keys())
        dimension_categories = list(self.dimension_scoring.keys())
        dimension_df = DataFrame(columns=["model_name", "dimension", "score", "baseline"])

        for dimension in dimension_categories:
            weights = self.dimension_scoring[dimension]
            dimension_weights = [weights[task_name] for task_name in task_categories]
            weights_sum = sum(dimension_weights)

            for model_name in all_results[task_categories[0]]:
                scores = []
                for task_name in task_categories:
                    success_rate = all_results[task_name][model_name]["success_rate"]
                    dimension_weight = self.dimension_scoring[dimension][task_name]
                    scores.append((100 * success_rate, dimension_weight))

                weighted_score = sum(score * weight for score, weight in scores) / weights_sum
                dash = model_name != "Current Run"
                dimension_df.loc[len(dimension_df)] = {
                    "model_name": model_name,
                    "dimension": dimension,
                    "score": weighted_score,
                    "baseline": dash
                }

        radar_dimension = line_polar(
            dimension_df,
            r='score',
            theta='dimension',
            line_close=True,
            category_orders={"category": dimension_categories},
            color='model_name',
            markers=True,
            line_dash="baseline",
            labels={'score': 'Score', 'dimension': 'Dimension', 'model_name': 'Model Name'},
        )

        radar_dimension.update_layout(
            width=700,
            height=400,
            title='Agent Ability Dimension Score w.r.t Models',
            title_x=0.1,
            legend_title_text='',
        )

        self.wandb_log({"summary/agent_abilities": Html(to_html(radar_dimension))})


class TaskLogger:
    """Logger for individual task execution details and visualization."""

    def __init__(self, wandb_run, task_name: str, log_path: Union[str, Path],
                max_num_steps: int = 30, baseline_dir: Union[str, Path] = "data/baseline_results"):
        """Initialize the TaskLogger.

        Args:
            wandb_run: Weights & Biases run object for logging
            task_name: Name of the task being logged
            log_path: Path to store log files
            max_num_steps: Maximum number of steps in task execution
            baseline_dir: Directory containing baseline results for comparison
        """
        self.wandb_log = wandb_run.log
        self.task_name = task_name
        self.max_num_steps = max_num_steps
        self.baseline_dir = Path(baseline_dir)

        # Setup logging paths
        log_path = Path(log_path)
        self.log_path = log_path / "logs" / f"{task_name}.jsonl"
        self.log_summary_path = log_path / f"{task_name}.txt"

        # Ensure directories exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create empty log files
        self.log_path.write_text("")
        self.log_summary_path.write_text("")

        # Setup table for wandb
        columns = ["id", "is_done", "env", "reward", "grounding_accuracy", "reward_wrt_step", "trajectory"]
        self.table = Table(columns=columns)
        self.columns = columns

        # Load baseline metrics
        self.baseline_metrics, self.baseline_reward_wrt_step = self.load_baseline_results()

    def extract_variables(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract variables from a log line.

        Args:
            line: The log line to parse

        Returns:
            Dictionary of extracted variables or None if parsing failed
        """
        pattern = r"\[EXP\] (\d+): \[success_rate\]: (.*), \[progress_rate\]: (.*), \[grounding_acc\]: (.*), \[score_state\]: (.*)"
        match = re.match(pattern, line)
        if not match:
            return None

        exp_id = int(match.group(1))

        # Process success rate
        sr_temp = match.group(2)
        if sr_temp == "True":
            sr_temp = 1
        elif sr_temp == "False":
            sr_temp = 0
        sr = float(sr_temp)

        # Extract other metrics
        score = float(match.group(3))
        grounding_acc = float(match.group(4))

        # Parse score state
        score_state_str = match.group(5)
        score_state = eval(score_state_str)
        score_state = [(int(step), float(score)) for step, score in score_state]

        return {
            "EXP": exp_id,
            "success_rate": sr,
            "progress_rate": score,
            "grounding_acc": grounding_acc,
            "score_state": score_state
        }

    def complete_score_state(self, score_state: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Complete the score state by filling in missing steps.

        Args:
            score_state: List of (step, score) tuples

        Returns:
            Completed list of (step, score) tuples for all steps
        """
        score_state = score_state.copy()  # Copy to avoid modifying the original
        complete_state = []
        current_score = 0

        for step in range(self.max_num_steps):
            if score_state and step == score_state[0][0]:
                current_score = score_state.pop(0)[1]
            complete_state.append((step, current_score))

        return complete_state

    def load_baseline_results(self) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
        """Load baseline results for the task.

        Returns:
            Tuple of (baseline metrics, baseline reward progression)
        """
        # Load baseline metrics
        baseline_metrics = self._load_baseline_metrics()

        # Load baseline reward progression
        baseline_reward_progression = self._load_baseline_reward_progression()

        return baseline_metrics, baseline_reward_progression

    def _load_baseline_metrics(self) -> Dict[str, Any]:
        """Load baseline metrics from all_results.txt files.

        Returns:
            Dictionary mapping model names to their metrics
        """
        baseline_metrics = {}
        task_name = self.task_name

        for model_dir in self.baseline_dir.iterdir():
            if not model_dir.is_dir():
                continue

            results_file = model_dir / "all_results.txt"
            if not results_file.exists():
                continue

            with open(results_file, "r") as f:
                for line in f:
                    try:
                        result = json_loads(line.strip())
                        if result.get("task_name") == task_name:
                            baseline_metrics[model_dir.name] = result
                    except Exception:
                        continue

        return baseline_metrics

    def _load_baseline_reward_progression(self) -> Dict[str, List[float]]:
        """Load baseline reward progression from task-specific log files.

        Returns:
            Dictionary mapping model names to reward progression lists
        """
        baseline_reward_progression = {}
        task_name = self.task_name

        for model_dir in self.baseline_dir.iterdir():
            if not model_dir.is_dir():
                continue

            log_file = model_dir / f"{task_name}.txt"
            if not log_file.exists():
                continue

            results = []
            with open(log_file, "r") as f:
                for line in f:
                    result = self.extract_variables(line)
                    if result:
                        result['score_state'] = self.complete_score_state(result['score_state'])
                        results.append(result)

            if not results:
                continue

            # Calculate average reward progression
            reward_score_list = [0] * self.max_num_steps
            for result in results:
                for step, score in result['score_state']:
                    reward_score_list[step] += score

            # Normalize
            for i in range(self.max_num_steps):
                reward_score_list[i] /= len(results)

            # Convert to percentage
            reward_score_list = [score * 100 for score in reward_score_list]

            # Add 0 at step 0
            reward_score_list.insert(0, 0)

            baseline_reward_progression[model_dir.name] = reward_score_list

        return baseline_reward_progression

    def log_example_data(self, id: int, is_done: Union[bool, str], reward: float,
                        grounding_accuracy: float, score_change_record: List[Tuple[int, float]],
                        env_details: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> None:
        """Log example data to the W&B table.

        Args:
            id: The example ID
            is_done: Whether the task was completed successfully
            reward: The reward/progress score
            grounding_accuracy: The grounding accuracy
            score_change_record: List of (step, score) tuples showing progress
            env_details: Environment details
            trajectory: List of trajectory steps
        """
        # Convert types
        if self.task_name not in ["webarena"]:
            is_done = bool(is_done)
        reward = float(reward)
        grounding_accuracy = float(grounding_accuracy)

        # Create reward progression chart
        reward_chart = self._create_reward_chart(score_change_record)

        # Create trajectory HTML visualization
        trajectory_html = self._create_trajectory_html(trajectory)

        # Add to W&B table
        self.table.add_data(
            id, is_done, env_details, reward, grounding_accuracy,
            Image(reward_chart),
            Html(trajectory_html)
        )

    def _create_reward_chart(self, score_change_record: List[Tuple[int, float]]) -> plt.Figure:
        """Create a reward progression chart.

        Args:
            score_change_record: List of (step, score) tuples showing progress

        Returns:
            Matplotlib figure with the reward chart
        """
        # Calculate reward at each step
        reward_wrt_steps = [0] * self.max_num_steps
        for step, reward in score_change_record:
            for i in range(int(step), self.max_num_steps):
                reward_wrt_steps[i] = reward

        # Create figure
        fig = plt.figure(figsize=(4, 4))
        plt.plot(
            range(self.max_num_steps),
            reward_wrt_steps,
            color='blue',
            marker='o',
            linestyle='solid',
            linewidth=1,
            markersize=2
        )

        # Remove top and right spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim(0, 1)

        return fig

    def _create_trajectory_html(self, trajectory: List[Dict[str, Any]]) -> str:
        """Create HTML visualization of a trajectory.

        Args:
            trajectory: List of trajectory steps

        Returns:
            HTML string representing the trajectory
        """
        html_head = '''
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                .goal { color: black; font-size: 10px; }
                .observation { color: grey; font-size: 10px; }
                .progress { color: brown; font-size: 10px; }
                .action { color: green; font-size: 10px; }
            </style>
            </head>
            <body>
        '''

        html_body = ""
        for item in trajectory:
            type_ = list(item.keys())[0]
            type_name = type_
            step_id = item["id"]
            content = item[type_]
            if isinstance(content, str) and len(content_split := content.split('\n')) > 5:
                content = "\n".join(content_split[:5]) + "\n   ..."

            if type_ == "Progress Rate":
                type_ = "progress"
            html_body += f'''
            <p class="{type_.lower()}"><b>Step {step_id} </b><b>{type_name}: </b>{content}</p>
            '''

        html_tail = '''
            </body>
            </html>
        '''
        return html_head + html_body + html_tail

    def update(self) -> None:
        """Update the W&B table with the latest data."""
        new_table = Table(columns=self.table.columns, data=self.table.data)
        self.wandb_log({f'{self.task_name}/predictions': new_table})

    def save_sample_data_to_file_detailed(self, id: int, is_done: Union[bool, str], reward: float,
                                        grounding_accuracy: float, score_change_record: List[Tuple[int, float]],
                                        env_details: Dict[str, Any], trajectory: List[Dict[str, Any]],
                                        example_prompt: Optional[str] = None) -> None:
        """Save detailed sample data to a JSONL file.

        Args:
            id: The sample ID
            is_done: Whether the task was completed successfully
            reward: The reward/progress rate value
            grounding_accuracy: The grounding accuracy value
            score_change_record: List of score changes at different steps
            env_details: Environment details dictionary
            trajectory: List of trajectory items
            example_prompt: Optional prompt used for the example

        Returns:
            None

        """
        if self.task_name not in ["webarena"]:
            is_done = bool(is_done)
        reward = float(reward)
        grounding_accuracy = float(grounding_accuracy)

        sample_result = {'id': id} | env_details | {
            'is_done': is_done,
            'progress_rate': reward,
            'grounding_acc': grounding_accuracy,
            'score_change_record': score_change_record,
            'trajectory': {},
        }

        for item in trajectory:
            type_ = list(item.keys())[0]
            step_id = int(item["id"])
            content = item[type_]

            step_name = f"Interaction Turn {step_id}"

            if step_name not in sample_result['trajectory']:
                sample_result['trajectory'][step_name] = {}
            sample_result['trajectory'][step_name][type_] = content

        if example_prompt is not None:
            sample_result["example_prompt"] = example_prompt

        # Use jsonlines to properly append a JSON object as a single line
        with jsonlines.open(self.log_path, mode='a') as writer:
            writer.write(sample_result)

    def save_sample_data_to_file_overview(self, id_: int, is_done: Union[bool, str], reward: float,
                                          grounding_accuracy: float, score_change_record: List[Tuple[int, float]],
                                          env_details: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> None:
        """Save overview sample data to a text file.

        Args:
            id_: The sample ID
            is_done: Whether the task was completed successfully
            reward: The reward/progress rate value
            grounding_accuracy: The grounding accuracy value
            score_change_record: List of score changes at different steps
            env_details: Environment details dictionary
            trajectory: List of trajectory items

        Returns:
            None

        """
        with open(self.log_summary_path, "a+") as f:
            f.write(f"[EXP] {id_}: [success_rate]: {is_done}, [progress_rate]: {reward}, [grounding_acc]: {grounding_accuracy}, [score_state]: {score_change_record} \n")

    def log_example(self, id_: int, is_done: Union[bool, str], reward: float,
                    grounding_accuracy: float, score_change_record: List[Tuple[int, float]],
                    env_details: Dict[str, Any], trajectory: List[Dict[str, Any]],
                    example_prompt: Optional[str] = None) -> None:
        """Log an example to both file and W&B.

        Args:
            id_: The example ID
            is_done: Whether the task was completed successfully
            reward: The reward/progress rate value
            grounding_accuracy: The grounding accuracy value
            score_change_record: List of score changes at different steps
            env_details: Environment details dictionary
            trajectory: List of trajectory items
            example_prompt: Optional prompt used for the example

        Returns:
            None

        """
        self.save_sample_data_to_file_detailed(id_, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory, example_prompt)  # log to file
        self.save_sample_data_to_file_overview(id_, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory)
        self.log_example_data(id_, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory)  # log to wandb table
        self.update()

    def log_summary(self, success_rate: float, reward_score: float, grounding_acc: float,
                    score_steps: List[List[Tuple[int, float]]], hard_sr: Optional[float] = None,
                    hard_rs: Optional[float] = None, easy_sr: Optional[float] = None,
                    easy_rs: Optional[float] = None) -> None:
        """Log summary metrics and visualizations for the task.

        Args:
            success_rate: Overall success rate
            reward_score: Progress/reward score
            grounding_acc: Grounding accuracy
            score_steps: List of score change records for different examples
            hard_sr: Success rate on hard problems (optional)
            hard_rs: Progress rate on hard problems (optional)
            easy_sr: Success rate on easy problems (optional)
            easy_rs: Progress rate on easy problems (optional)

        Returns:
            None

        """
        # Log metrics to W&B table
        metrics_table = Table(columns=["Metric Name", "Metric Value (%)"])
        metrics_table.add_data("Progress Rate", reward_score)
        metrics_table.add_data("Success Rate", success_rate)
        metrics_table.add_data("Grounding Accuracy", grounding_acc)
        self.wandb_log({f'{self.task_name}/metrics': metrics_table})

        # Log comparison with baseline models
        self._log_baseline_comparison(success_rate, reward_score, grounding_acc)

        # Log reward progression
        self._log_reward_progression(score_steps)

        # Log success and progress rates by difficulty (if available)
        if hard_sr is not None:
            self._log_difficulty_metrics(hard_sr, hard_rs, easy_sr, easy_rs)

    def _log_baseline_comparison(self, success_rate: float, reward_score: float, grounding_acc: float) -> None:
        """Log comparison of current run with baseline models.

        Args:
            success_rate: Overall success rate
            reward_score: Progress/reward score
            grounding_acc: Grounding accuracy

        Returns:
            None

        """
        baseline_models = list(self.baseline_metrics.keys())
        models = ["Current Run"] + baseline_models

        accuracies = [success_rate] + [self.baseline_metrics[model]["success_rate"] for model in baseline_models]
        rewards = [reward_score] + [self.baseline_metrics[model]["progress_rate"] for model in baseline_models]
        grounding_accs = [grounding_acc] + [self.baseline_metrics[model]["grounding_acc"] for model in baseline_models]

        marker_color_acc = ['rgba(0,128, 255, 1)'] + ['rgba(0,128, 255, 0.6)'] * len(baseline_models)
        marker_color_reward = ['rgba(0, 204,102, 1)'] + ['rgba(0, 204,102, 0.6)'] * len(baseline_models)
        marker_color_grounding = ['rgba(248,173,30, 1)'] + ['rgba(248,173,30, 0.6)'] * len(baseline_models)

        data = [
            Bar(name='Progress Rate (%)', x=models, y=rewards, marker_color=marker_color_reward),
            Bar(name='Success Rate (%)', x=models, y=accuracies, marker_color=marker_color_acc),
            Bar(name='Grounding Accuracy (%)', x=models, y=grounding_accs, marker_color=marker_color_grounding)
        ]

        layout = Layout(
            width=800,
            height=400,
            xaxis={'categoryorder': 'total descending'},
            title=f'{self.task_name.capitalize()} Metrics Compared to Baseline Models',
        )

        fig = Figure(data=data, layout=layout)
        self.wandb_log({f'{self.task_name}/metrics_comparison': Plotly(fig)})

    def _log_reward_progression(self, score_steps: List[List[Tuple[int, float]]]) -> None:
        """Log reward progression over steps.

        Args:
            score_steps: List of score change records for different examples

        Returns:
            None

        """
        df = DataFrame(columns=["models", "steps", "score", "baseline"])

        # Add current run
        reward_score_list = [0] * self.max_num_steps
        for score_step_example in score_steps:
            score_step_example = self.complete_score_state(score_step_example)
            for step, score in score_step_example:
                reward_score_list[step] += score

        for i in range(self.max_num_steps):
            reward_score_list[i] /= len(score_steps)

        reward_score_list = [score * 100 for score in reward_score_list]
        reward_score_list.insert(0, 0)

        for step, score in enumerate(reward_score_list):
            df.loc[len(df)] = {"models": "Current Run", "steps": step, "score": score, "baseline": False}

        # Add baseline runs
        for model in self.baseline_metrics:
            for step, score in enumerate(self.baseline_reward_wrt_step[model]):
                df.loc[len(df)] = {"models": model, "steps": step, "score": score, "baseline": True}

        line_fig = line(df, x="steps", y="score", color='models', title=f"Average Progress Rate (%) w.r.t Steps for {self.task_name} Tasks", width=800, height=400, line_dash="baseline",
                        labels={"models": "Model Name", "baseline": "Is Baseline"})

        self.wandb_log({f'{self.task_name}/task_reward_w.r.t_steps': Plotly(line_fig)})

    def _log_difficulty_metrics(self, hard_sr: float, hard_rs: float, easy_sr: float, easy_rs: float) -> None:
        """Log success and progress rates by difficulty.

        Args:
            hard_sr: Success rate on hard problems
            hard_rs: Progress rate on hard problems
            easy_sr: Success rate on easy problems
            easy_rs: Progress rate on easy problems

        Returns:
            None

        """
        baseline_models = list(self.baseline_metrics.keys())
        models = ["Current Run"] + baseline_models

        easy_sr_data = [easy_sr] + [self.baseline_metrics[model]["success_rate_easy"] for model in baseline_models]
        hard_sr_data = [hard_sr] + [self.baseline_metrics[model]["success_rate_hard"] for model in baseline_models]

        difficulty_sr_data = [
            Bar(name='Success Rate For Easy Examples(%)', y=models, x=easy_sr_data, marker_color=['rgba(102, 255, 255, 1)'] + ['rgba(102, 255,255, 0.6)'] * len(baseline_models), orientation='h'),
            Bar(name='Success Rate For Hard Examples(%)', y=models, x=hard_sr_data, marker_color=['rgba(0, 128, 255, 0.4)'] + ['rgba(0, 128,255, 0.4)'] * len(baseline_models), orientation='h')
        ]
        layout_sr_difficulty = Layout(
            width=800,
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            barmode='overlay',
            title=f'{self.task_name.capitalize()} Success Rate w.r.t Difficulty',
        )

        fig_sr_difficulty = Figure(data=difficulty_sr_data, layout=layout_sr_difficulty)

        easy_rs_data = [easy_rs] + [self.baseline_metrics[model]["progress_rate_easy"] for model in baseline_models]
        hard_rs_data = [hard_rs] + [self.baseline_metrics[model]["progress_rate_hard"] for model in baseline_models]

        difficulty_rs_data = [
            Bar(name='Progress Rate For Easy Examples(%)', y=models, x=easy_rs_data, marker_color=['rgba(0,255,128, 1)'] + ['rgba(0,255,128, 0.6)'] * len(baseline_models), orientation='h'),
            Bar(name='Progress Rate For Hard Examples(%)', y=models, x=hard_rs_data, marker_color=['rgba(0, 153,76, 0.6)'] + ['rgba(0, 153,76, 0.6)'] * len(baseline_models), orientation='h')
        ]

        layout_rs_difficulty = Layout(
            width=800,
            height=400,
            barmode='overlay',
            yaxis={'categoryorder': 'total ascending'},
            title=f'{self.task_name.capitalize()} Progress Rate w.r.t Difficulty',
        )
        fig_rs_difficulty = Figure(data=difficulty_rs_data, layout=layout_rs_difficulty)
        fig_rs_difficulty.for_each_trace(lambda t: t.update(name='<b>' + t.name + '</b>') if t.name in "Current Run" else ())

        self.wandb_log({
            f'{self.task_name}/success_rate_w.r.t_difficulty': Plotly(fig_sr_difficulty),
            f'{self.task_name}/progress_score_w.r.t_difficulty': Plotly(fig_rs_difficulty),
        })
