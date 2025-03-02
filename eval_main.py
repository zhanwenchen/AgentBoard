from argparse import ArgumentParser
from os import getenv, putenv
from re import compile as re_compile
from json import loads as json_loads
from pathlib import Path  # Import Path from pathlib
from typing import Dict, Union, Tuple, List, Any, Optional  # Added type annotations
# import contextlib
# from gc import collect
# from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
# from torch.cuda import empty_cache
# from torch.distributed import destroy_process_group
from dotenv import load_dotenv
from yaml import add_implicit_resolver, add_constructor, load as yaml_load, FullLoader
from wandb import init as wandb_init, Api
from agentboard.tasks import load_task
from agentboard.llm import load_llm
from agentboard.utils.logging.agent_logger import AgentLogger
from agentboard.utils.logging.logger import SummaryLogger


# warnings.filterwarnings('ignore')

PATH_MATCHER = re_compile(r'\$\{([^}^{]+)\}')
TASKS = ['alfworld', 'jericho', 'pddl', 'webshop', 'webarena', 'tool-query', 'tool-operation', 'babyai', 'scienceworld']
WANDB_TEAM = 'dri-ice'
WANDB_PROJECT = 'finetuners'


def parse_args():
    parser = ArgumentParser(description='Testing')
    parser.add_argument('--ab-cfg-path', required=True, help='path to configuration file.')
    parser.add_argument('--ab-tasks', required=True, type=str, nargs='+',help='specify the tasks')
    parser.add_argument('--ab-model', required=True ,help='specify the models, available models are stated in the configuration file')
    parser.add_argument('--ab-wandb', action='store_true', help='specify whether the wandb board is needed')
    parser.add_argument('--ab-log_path', required=False, default='', help='specify the place to store the resuls')
    parser.add_argument('--ab-project_name', required=False, default='', help='specify the project name for wandb')
    parser.add_argument('--ab-baseline_dir', required=False, default='', help='specify the baseline loggings for wandb baseline comparison visualization')
    parser.add_argument('--ab-max_num_steps', required=True, type=int, help='specify the maximum number of steps used to finish the problems')
    return parser.parse_args()


def path_constructor(_, node):
    '''Extract the matched value, expand env variable, and replace the match.

    Args:
        _: Unused parameter.
        node: The YAML node containing the path.

    Returns:
        The constructed path with environment variables expanded.
    '''
    value = node.value
    match = PATH_MATCHER.match(value)
    env_var = match.group()[2:-1]
    return getenv(env_var) + value[match.end():]


def load_config(cfg_path: Union[str, Path], ab_log_path, ab_project_name, ab_baseline_dir, ab_wandb, ab_max_num_steps) -> Tuple[Dict, Dict, Dict, Dict]:
    '''Load configuration from YAML file.

    Args:
        cfg_path: Path to the configuration file.
        args: Command line arguments.

    Returns:
        Tuple containing llm_config, agent_config, env_config, and run_config.
    '''
    add_implicit_resolver('!path', PATH_MATCHER)
    add_constructor('!path', path_constructor)
    with open(cfg_path, 'r') as f:
        config = yaml_load(f, Loader=FullLoader)
    llm_config = config['llm']
    agent_config = config['agent']
    env_config = config['env']
    run_config = config['run']

    if ab_log_path != '':
        run_config['log_path'] = ab_log_path
    if ab_project_name != '':
        run_config['project_name'] = ab_project_name
    if ab_baseline_dir != '':
        run_config['baseline_dir'] = ab_baseline_dir

    run_config['wandb'] = ab_wandb
    run_config['max_num_steps'] = ab_max_num_steps

    return llm_config, agent_config, env_config, run_config


def make_log_paths(log_dir: Union[str, Path], baseline_dir: Union[str, Path]) -> None:
    '''Ensure that all required directories and files exist, creating them if necessary.

    Args:
        log_dir: Path to the log directory.
        baseline_dir: Path to the baseline directory.

    Returns:
        None
    '''
    # Convert to Path objects
    log_path = Path(log_dir)
    logs_path = log_path / 'logs'
    baseline_path = Path(baseline_dir)
    all_results_path = log_path / 'all_results.txt'

    # Create directories if they don't exist
    logs_path.mkdir(parents=True, exist_ok=True)
    baseline_path.mkdir(parents=True, exist_ok=True)

    # Create results file if it doesn't exist
    if not all_results_path.exists():
        all_results_path.touch()


def get_wandb_run(wandb_run_id: str, logger):
    '''Get a Weights & Biases run by ID.

    Args:
        wandb_run_id: The ID of the run to retrieve.
        logger: The logger to use.

    Returns:
        The wandb run object.

    Raises:
        RuntimeError: If the run could not be found.
    '''
    try:
        logger.info(f'train.get_wandb_run: trying to load wandb_run_id={wandb_run_id}')
        wandb_run_current = Api().from_path(f'{WANDB_TEAM}/{WANDB_PROJECT}/runs/{wandb_run_id}')
    except Exception as e:
        logger.info(f'train.get_wandb_run: wandb_run_id={wandb_run_id} not found')
        raise RuntimeError(f'train.get_wandb_run: wandb_run_id={wandb_run_id} not found') from e
    return wandb_run_current


def load_history(log_dir: Union[str, Path], logger) -> Dict[str, Any]:
    '''Load task history from the all_results.txt file.

    Args:
        log_dir: Path to the log directory.
        logger: The logger to use.

    Returns:
        Dictionary mapping task names to their results.
    '''
    log_history = {}
    results_path = Path(log_dir) / 'all_results.txt'

    with results_path.open('r') as f:
        for line in f:
            logger.info(line)
            if '_summary' not in line:
                line_value = json_loads(line.strip())
                log_history[line_value['task_name']] = line_value
    return log_history


def evaluate_task(log_history: Dict[str, Any], task_name: str, llm, llm_config, agent_config: Dict[str, Any], env_config: Dict[str, Any], run_config: Dict[str, Any], agentboard, logger, wandb_run) -> None:
    '''Evaluate a specific task.

    Args:
        log_history: Dictionary of previously evaluated tasks.
        task_name: Name of the task to evaluate.
        llm: Language model to use for evaluation.
        agent_config: Agent configuration.
        env_config: Environment configuration.
        run_config: Run configuration.
        agentboard: SummaryLogger instance for logging results.
        logger: Logger for tracking progress.

    Returns:
        None
    '''
    # If the results of the task is already available at {log_path}/all_results.txt, skip the evaluation of this task to avoid rerunning.
    # If you wish to rerun a task, make sure to remove the line recording previous task results from {log_path}/all_results.txt
    results_path = Path(run_config['log_path']) / 'all_results.txt'

    if task_name in log_history:
        raise RuntimeError(f'task_name: {task_name} already exists in the log_history = {log_history}, fpath_all_results={results_path}')
        logger.info(f"Task {task_name} has been evaluated, skip")


        agentboard.log_run_result(task_name, log_history[task_name]["success_rate"], log_history[task_name]["progress_rate"], log_history[task_name]["grounding_acc"],
                                    log_history[task_name]["success_rate_hard"], log_history[task_name]["success_rate_hard"], log_history[task_name]["progress_rate_hard"],
                                    log_history[task_name]["progress_rate_easy"])

        return

    logger.info(f"Start evaluating task {task_name}")

    agent_task_config = agent_config.copy()
    for key in env_config[task_name]:
        if key in ["check_actions", "check_inventory", "init_prompt_path"]:
            agent_task_config[key] = env_config[task_name][key]

    task = load_task('tool' if 'tool' in task_name else task_name, run_config, llm_config, agent_task_config, env_config[task_name], llm=llm, wandb_run=wandb_run)
    logger.info(f"Loaded task {task_name}")

    success_rates, progress_rates, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr = task.evaluate()

    success_rate = sum(success_rates) * 1.0 / len(success_rates)
    progress_rate = sum(progress_rates) * 1.0 / len(progress_rates)
    grounding_acc = sum(grounding_accs) * 1.0 / len(grounding_accs)

    logger.finish(f'Task {task_name} | Success Rate: {success_rate}, Progress Rate: {progress_rate}, Easy SR: {easy_sr}. Hard SR: {hard_sr}, Easy PR: {easy_pr}, Hard PR: {hard_pr}, Grounding Accuracy: {grounding_acc}')
    agentboard.log_run_result(task_name, success_rate, progress_rate, grounding_acc, hard_sr, easy_sr, hard_pr, easy_pr)


def main(args, llm_config_eval: dict, wandb_run_id: str):
    '''Main function to run evaluation.

    Args:
        args: Command line arguments.
        llm_config_eval: Optional LLM configuration overrides.
        wandb_run_id: Optional Weights & Biases run ID.
    '''
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...

    # args = parse_args()
    llm_config, agent_config, env_config, run_config = load_config(args.ab_cfg_path, args.ab_log_path, args.ab_project_name, args.ab_baseline_dir, args.ab_wandb, args.ab_max_num_steps)
    if llm_config_eval:
        llm_config.update(llm_config_eval)
        llm_config = llm_config[args.ab_model]
    logger = AgentLogger(__name__)
    logger.info(f'llm_config={llm_config}')
    # with open() as f:
    #     yaml.dump(llm_config, f)
    logger.info(f'Got args.ab_model={args.ab_model}')

    #---------------------------------------------- load llm -----------------------------------------------------
    logger.info('Start loading language model')

    llm = load_llm(llm_config['name'], llm_config)

    logger.info('Finished loading language model')

    #------------------------------------------------ initialize agentboard ------------------------------------
    if not run_config.get('wandb', False):
        logger.info('Wandb is disabled')
        putenv('WANDB_MODE', 'disabled')
        wandb_run = wandb_init(mode='disabled')
    elif wandb_run_id is None:
        logger.info("Wandb is not enabled because wandb_run_id is None")
        wandb_run = wandb_init(
            project=run_config["project_name"],
            name=f'{llm_config["name"]}_{llm_config["engine"]}',
            notes="An evaluation of LLM agent",
            config={
                'llm_config': llm_config,
                'agent_config': agent_config,
                'env_config': env_config,
                'run_config': run_config
            },
        )
    else:
        logger.info(f'Using existing wandb_run_id={wandb_run_id}')
        wandb_run_api = get_wandb_run(wandb_run_id, logger)
        wandb_run = wandb_init(project=wandb_run_api.project, name=wandb_run_api.name, id=wandb_run_api.id)

    log_dir = run_config.get("log_path", None)

    baseline_path = run_config.get('baseline_dir', 'data/baseline_results_details')

    make_log_paths(log_dir, baseline_path)

    # agentboard is the main launcher of visualizations and metrics calculation,
    agentboard = SummaryLogger(wandb_run, baseline_dir=baseline_path, log_path=log_dir)

    ab_tasks = args.ab_tasks
    logger.info(f'ab_tasks={ab_tasks}')
    task_names = TASKS if ab_tasks == ['all'] else ab_tasks

    log_history = load_history(log_dir, logger)
    logger.info("Tested tasks: " + " ".join(log_history))

    #------------------------------------------------- start evaluation -------------------------------------------
    task = None
    for task_name in task_names:
        evaluate_task(log_history, task_name, llm, llm_config, agent_config, env_config, run_config, agentboard, logger, wandb_run)

    logger.info('Finish evaluating all tasks')

    agentboard.log_summary()
    # destroy_model_parallel()
    # destroy_distributed_environment()
    del task, agentboard, llm
    # # collect()
    # empty_cache()
    # with contextlib.suppress(AssertionError):
    #     destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
