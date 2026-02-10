import os
import time
import re
import argparse
import shutil
import asyncio
import json
import random
import yaml
from beartype.typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
WEBARENA_ROOT = os.path.join(PROJECT_ROOT, "webarena")
if WEBARENA_ROOT not in sys.path:
    sys.path.append(WEBARENA_ROOT)
AGENTOCCAM_ROOT = os.path.join(PROJECT_ROOT, "AgentOccam")
if AGENTOCCAM_ROOT not in sys.path:
    sys.path.append(AGENTOCCAM_ROOT)

from AgentOccam.env import WebArenaEnvironmentWrapper

from webagents_step.utils.data_prep import *
from webagents_step.agents.step_agent import StepAgent

from AgentOccam.prompts import AgentOccam_prompt
from webagents_step.prompts.webarena import step_fewshot_template_adapted, step_fewshot_template

from AgentOccam.utils import EVALUATOR_DIR

from memory_retrieving.memory_graph import MemoryGraph
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant
from utils import save_episodic

import AgentOccamWithMemory_prompt
from plugmem_agent import AgentOccamWithMemory, print_memory_graph_stats


TRAJECTORY_DIR_DEFAULT = os.path.join(
    CURRENT_DIR, "AgentOccam-Trajectories-demo", "AgentOccam-debug"
)
def _load_trajectory(task_id: int, trajectory_dir: str) -> List[Dict]:
    trajectory_file = os.path.join(trajectory_dir, f"{task_id}.json")
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found: {trajectory_file}.")
        return []
    try:
        with open(trajectory_file, "r") as f:
            trajectory_json = json.load(f)
            trajectory_data = trajectory_json.get("trajectory", [])
            if trajectory_data:
                print(f"Loaded trajectory with {len(trajectory_data)} steps from {trajectory_file}")
            else:
                print(f"Trajectory file {trajectory_file} exists but has no trajectory data.")
            return trajectory_data
    except Exception as e:
        print(f"Failed to load trajectory from {trajectory_file}: {e}.")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    parser.add_argument(
        "--replay-trajectory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replay trajectory before normal evaluation (default: False)"
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=TRAJECTORY_DIR_DEFAULT,
        help="trajectory directory for replay"
    )
    parser.add_argument(
        "--load_memory_graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="load persisted memory graph from disk (default: False)"
    )
    parser.add_argument(
        "--refresh-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="refresh memory embeddings on load (default: False)"
    )
    parser.add_argument(
        "--read-only-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="read from memory graph but do not insert new memories (default: False)"
    )
    parser.add_argument(
        "--disable-memory-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable all memory graph related operations (default: False)"
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))
    
    dstdir = None
    if config.logging:
        if config.logname:
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    random.seed(42)
    
    config_file_list = []
    
    task_ids = config.env.task_ids
    if hasattr(config.env, "relative_task_dir"):
        relative_task_dir = config.env.relative_task_dir
    else:
        relative_task_dir = "tasks"
    if task_ids == "all" or task_ids == ["all"]:
        task_ids = [filename[:-len(".json")] for filename in os.listdir(f"{AGENTOCCAM_ROOT}/config_files/{relative_task_dir}") if filename.endswith(".json")]
    for task_id in task_ids:
        config_file_list.append(f"{AGENTOCCAM_ROOT}/config_files/{relative_task_dir}/{task_id}.json")

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    # memory init
    mg = None
    if not args.disable_memory_graph:
        mg = MemoryGraph(
            tag_equal=TagEqual(),
            tag_relevant=TagRelevant(),
            semantic_equal=SemanticEqual(),
            semantic_relevant=SemanticRelevant(),
            subgoal_equal=SubgoalEqual(),
            subgoal_relevant=SubgoalRelevant(),
            procedural_equal=ProceduralEqual(),
            procedural_relevant=ProceduralRelevant(),
            load_from_disk=args.load_memory_graph,
            refresh_embeddings=args.refresh_embeddings
        )
        # Print memory graph stats after initialization
        print_memory_graph_stats(mg, context="After Initialization")
    else:
        print("Memory graph disabled; skipping initialization and related operations.")
    
    # agent init factory
    def agent_init(trajectory_data=None, replay_mode=False):
        if config.agent.type == "AgentOccam":
            return AgentOccamWithMemory(
                prompt_dict = {k: v for k, v in AgentOccamWithMemory_prompt.__dict__.items() if isinstance(v, dict)},
                config = config.agent,
                memory_graph = mg,
                trajectory_data = trajectory_data,
                replay_mode = replay_mode,
                read_only_memory = args.read_only_memory
            )
        raise NotImplementedError(f"{config.agent.type} not implemented")
    
    """
    EVALUATION LOOP
    """
    # Track results
    results = []
    total_tasks = 0
    successful_tasks = 0
    
    for config_file in config_file_list:
        with open(config_file, "r") as f:
            task_config = json.load(f)
            task_id = task_config['task_id']
            print(f"Task {task_id}.")
        if dstdir and os.path.exists(os.path.join(dstdir, f"{task_id}.json")):
            print(f"Skip {task_id}.")
            continue
        if task_id in list(range(600, 650))+list(range(681, 689)):
            print("Reddit post task. Sleep 30 mins.")
            time.sleep(1800)

        if args.replay_trajectory:
            trajectory_data = _load_trajectory(task_id=task_id, trajectory_dir=args.trajectory_dir)
            if trajectory_data:
                replay_agent = agent_init(trajectory_data=trajectory_data, replay_mode=True)
                replay_objective = trajectory_data[0].get("objective", task_config.get("goal", ""))
                print(f"Replay objective: {replay_objective}")
                replay_agent.act(objective=replay_objective, env=None, memory_graph=mg, task_id=task_id)
            else:
                print(f"Skipping replay for task {task_id} (missing trajectory).")
        
        env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                        max_browser_rows=config.env.max_browser_rows, 
                                        max_steps=config.max_steps, 
                                        slow_mo=1, 
                                        observation_type="accessibility_tree", 
                                        current_viewport_only=current_viewport_only, 
                                        viewport_size={"width": 1920, "height": 1080}, 
                                        headless=config.env.headless,
                                        global_config=config)
        agent = agent_init()
        objective = env.get_objective()

        print(f"Objective: {objective}")
        # Call act with memory_graph integration
        status = agent.act(objective=objective, env=env, memory_graph=mg, task_id=task_id)
        
        # Check task success
        if status is None:
            # Get status from environment if agent.act didn't return it
            status = env.status()
        
        # Extract success information
        if isinstance(status, dict):
            reward = status.get('reward', 0.0)
            success = status.get('success', 0.0)
            num_actions = status.get('num_actions', 0)
            is_done = status.get('done', False)
        else:
            # Fallback: get status from environment
            status = env.status()
            reward = status.get('reward', 0.0)
            success = status.get('success', 0.0)
            num_actions = status.get('num_actions', 0)
            is_done = status.get('done', False)
        
        # Determine success status
        is_success = reward > 0 or success > 0
        success_str = "SUCCESS" if is_success else "FAIL"
        
        # Update counters
        total_tasks += 1
        if is_success:
            successful_tasks += 1
        
        # Print result
        print(f"Task {task_id}: {success_str} (Reward: {reward}, Steps: {num_actions})")
        
        # Store result
        result = {
            'task_id': task_id,
            'objective': objective,
            'success': is_success,
            'reward': reward,
            'num_actions': num_actions,
            'done': is_done,
            'status': success_str
        }
        results.append(result)
        
        # Save result to file if logging is enabled
        if dstdir:
            result_file = os.path.join(dstdir, f"{task_id}_result.json")
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
        
        env.close()

        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "trajectory": agent.get_trajectory(),
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.actor.model if hasattr(config.agent, "actor") else config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
            }
            if status:
                summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {successful_tasks}")
    print(f"Failed tasks: {total_tasks - successful_tasks}")
    if total_tasks > 0:
        success_rate = (successful_tasks / total_tasks) * 100
        print(f"Success rate: {success_rate:.2f}%")
    print("="*80)
    
    # Save summary if logging is enabled
    if dstdir:
        summary_file = os.path.join(dstdir, "evaluation_summary.json")
        summary = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'success_rate': (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0,
            'results': results
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")
    
    
if __name__ == "__main__":
    main()
