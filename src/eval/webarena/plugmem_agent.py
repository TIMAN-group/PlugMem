import json
import re
from beartype.typing import Any, Dict, List

from AgentOccam.AgentOccam import AgentOccam, PlanTreeNode, Actor
from webagents_step.utils.data_prep import DotDict

from memory_structuring.memory import Memory
from memory_retrieving.memory_graph import MemoryGraph
from utils import call_gpt


def print_memory_graph_stats(mg: MemoryGraph, context: str = ""):
    """
    Print statistics about the memory graph, showing number of nodes of each memory type.
    """
    stats = {
        "episodic_nodes": len(mg.episodic_nodes),
        "semantic_nodes": len(mg.semantic_nodes),
        "tag_nodes": len(mg.tag_nodes),
        "subgoal_nodes": len(mg.subgoal_nodes),
        "procedural_nodes": len(mg.procedural_nodes),
    }
    total = sum(stats.values())
    print("\n" + "=" * 60)
    if context:
        print(f"Memory Graph Stats - {context}")
    else:
        print("Memory Graph Stats")
    print("=" * 60)
    print(f"Episodic nodes:  {stats['episodic_nodes']}")
    print(f"Semantic nodes:  {stats['semantic_nodes']}")
    print(f"Tag nodes:       {stats['tag_nodes']}")
    print(f"Subgoal nodes:   {stats['subgoal_nodes']}")
    print(f"Procedural nodes: {stats['procedural_nodes']}")
    print(f"Total nodes:     {total}")
    print("=" * 60 + "\n")


class ActorWithMemory(Actor):
    """
    Modified Actor class that takes into consideration retrieved_memory in predict_action()
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize ActorWithMemory by calling parent Actor __init__
        """
        super().__init__(*args, **kwargs)
        # Initialize retrieved_memory attribute
        self.retrieved_memory = ""
    
    def get_retrieved_memory(self):
        return self.retrieved_memory
    
    def get_tabs_info(self):
        return self.online_interaction.get("tabs", "N/A")

    def get_url_info(self):
        return "CURRENT URL: " + self.online_interaction.get("url", "N/A")

    def get_online_input(self, criticism_elements):
        input_template = self.prompt_template["input_template"]
        input_prefix, input_suffix = input_template.split("{input}")
        INPUT_TYPE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "interaction history": self.get_interaction_history(),
            "current observation": self.get_observation_text(),
            "current visual observation": self.get_observation_image(),
            "url": self.get_url_info(),
            "tabs": self.get_tabs_info(),
            "retrieved memory": self.get_retrieved_memory()
        }
        input_list = []
        for input_type in self.config.input:
            input_content = None
            if input_type == "current visual observation":
                continue
            elif input_type in INPUT_TYPE_TO_CONTENT_MAP.keys():
                input_content = INPUT_TYPE_TO_CONTENT_MAP[input_type]
            elif input_type.startswith("critic: ") and criticism_elements and input_type[len("critic: "):] in criticism_elements.keys() and criticism_elements[input_type[len("critic: "):]]:
                input_type = input_type[len("critic: "):]
                input_content = criticism_elements[input_type]
                input_type = "FROM USER: " + input_type
            if input_content and isinstance(input_content, str):
                input_list.append(("text", f"{input_type.upper()}:\n{input_content}\n"))
            elif input_content and isinstance(input_content, list):
                input_list.append(("text", f"{input_type.upper()}:\n"))
                input_list += input_content if len(input_content) > 0 else ["N/A"]

        if "image" in self.config.current_observation.type:
            input_type = "current visual observation"
            input_list.append(("text", f"{input_type.upper()}:\n"))
            input_list.append(("image", INPUT_TYPE_TO_CONTENT_MAP["current visual observation"]))

        return self.prune_message_list(message_list=[("text", input_prefix)] + input_list + [("text", input_suffix)])

    def verbose(self, instruction, online_input, model_response_list, action_element_list):
        action_element_keys = [k for k in self.config.play if k in action_element_list[0].keys()]
        other_play_keys = [k for k in self.config.play if k not in action_element_list[0].keys()]

        VERBOSE_TO_CONTENT_MAP = {
            "step": self.get_step(),
            "objective": self.objective,
            "previous plans": self.get_previous_plans(verbose=True),
            "url": self.online_interaction["url"],
            "observation": self.get_observation_text(),
            "retrieved memory": self.get_retrieved_memory(),
            "response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join([f"|\tAgent {i}:\n{model_response}" for i, model_response in enumerate(model_response_list[:self.config.number])]) if self.config.number > 1 else model_response_list[0],
            "instruction": instruction,
            "online input": "\n".join([i[1] for i in online_input if i[0]=="text"]),
            "alter ego response": "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n".join(["|\tAgent {}:\n{}".format(identity.config.name, response) for identity, response in zip(self.identities, model_response_list[self.config.number:])])
        }

        if self.config.others.verbose > 0 and self.config.verbose > 0:
            with open(self.output_trash_path, "a") as af:
                af.write("-"*32+"ACTOR"+"-"*32+"\n")
            for t in self.config.trash:
                content = VERBOSE_TO_CONTENT_MAP.get(t, "")
                with open(self.output_trash_path, "a") as af:
                    af.write(f"{t.upper()}:\n{content}\n\n")
            with open(self.output_play_path, "w") as _:
                pass
            for p in other_play_keys:
                content = VERBOSE_TO_CONTENT_MAP.get(p, "")
                with open(self.output_play_path, "a") as af:
                    af.write(f"{p.upper()}:\n{content}\n\n")
            for i, action_elements in enumerate(action_element_list):
                if len(action_element_list) > 1:
                    with open(self.output_play_path, "a") as af:
                        af.write("-"*32+f"AGENT {i}"+"-"*32+"\n")
                for action_element_key in action_element_keys:
                    content = action_elements.get(action_element_key, "N/A")
                    with open(self.output_play_path, "a") as af:
                        af.write(f"{action_element_key.upper()}:\n{content}\n\n")


class AgentOccamWithMemory(AgentOccam):
    """
    AgentOccam with memory graph integration and optional trajectory replay.
    """
    def __init__(self, 
                config: DotDict = None,
                prompt_dict: Dict[str, Any] = None, 
                memory_graph: MemoryGraph = None,
                trajectory_data: List[Dict] = None,
                replay_mode: bool = False,
                read_only_memory: bool = False):
        self.memory_graph = memory_graph
        self.memory = None  # Will be initialized in act method
        self.trajectory_data = trajectory_data  # Loaded trajectory data
        self.trajectory_step = 0  # Current step index in trajectory
        self.current_task_id = None
        self.read_only_memory = read_only_memory  # If True, don't insert memories into graph
        self.replay_mode = replay_mode
        self.tabs_info = None
        super().__init__(config=config, prompt_dict=prompt_dict)
    
    def init_actor(self):
        self.config.actor.others = self.config.others
        if len(self.sites) > 1:
            self.config.actor.navigation_command += ["go_home"]
        self.actor = ActorWithMemory(
            config=self.config.actor,
            objective=self.objective,
            prompt_template=self.prompt_dict["actor"],
            plan_tree_node=PlanTreeNode(id=0, type="branch", text=f"Find the solution to \"{self.objective}\"", level=0, url=self.online_url, step=0)
        )
        self.actor.retrieved_memory = ""
        with open(self.actor.output_trash_path, "w") as _:
            pass

    def predict_action(self, memory_graph: MemoryGraph = None):
        """
        Predict action from model or load from trajectory if replay_mode is enabled.
        """
        if self.replay_mode:
            if self.trajectory_data is not None and self.trajectory_step < len(self.trajectory_data):
                trajectory_step_data = self.trajectory_data[self.trajectory_step]
                
                # Extract action elements from trajectory
                # Note: trajectory uses underscores, but code expects spaces in keys
                action_elements = {
                    "action": trajectory_step_data.get("action", ""),
                    "reason": trajectory_step_data.get("reason", ""),
                    "observation description": trajectory_step_data.get("observation_description", ""),
                    "interaction history summary": trajectory_step_data.get("interaction_history_summary", ""),
                    "observation highlight": trajectory_step_data.get("observation_highlight", ""),
                    "retained element ids": trajectory_step_data.get("retained_element_ids", []),
                }
                
                # Handle navigation action if present (check both formats)
                if "navigation_action" in trajectory_step_data:
                    action_elements["navigation action"] = trajectory_step_data["navigation_action"]
                elif "navigation action" in trajectory_step_data:
                    action_elements["navigation action"] = trajectory_step_data["navigation action"]
                
                # Format as action_element_list (list with single element for single agent)
                action_element_list = [action_elements]
                
                print(f"Loaded action from trajectory step {self.trajectory_step}: {action_elements['action']}")
                self.trajectory_step += 1
                
                return action_elements, action_element_list
            else:
                raise RuntimeError(
                    f"Trajectory exhausted or not available (step {self.trajectory_step}, trajectory length: {len(self.trajectory_data) if self.trajectory_data else 0})."
                )
        
        mg = memory_graph if memory_graph is not None else self.memory_graph
        
        # Retrieve relevant memory if available
        retrieved_memory_str = ""
        if mg is not None and self.memory is not None and self.objective is not None:
            try:
                # Retrieve memory based on current state
                goal = self.objective
                observation_text = self.actor.get_observation_text()
                url_info_str = self.actor.get_url_info()
                tabs_info_str = self.actor.get_tabs_info()
                observation_collection_str = f"MAIN_GOAL:\n{goal}\nOBSERVATION:\n{observation_text}\nURL:\n{url_info_str}\nTABS:\n{tabs_info_str}"
                state = self.memory.state_t0
                
                messages, variables, sel_type = mg.retrieve_memory(
                    goal=goal,
                    observation=observation_collection_str,
                    state=state,
                    task_type="web navigation task",
                    task_id=self.current_task_id,
                    step_idx=self.get_step()
                )
                
                retrieved_memory_str = call_gpt(messages=messages, model_id="gpt-4o")
                self.actor.retrieved_memory = retrieved_memory_str
                print(f"Sel Type: {sel_type}, Retrieved memory: {retrieved_memory_str}")
                
            except Exception as e:
                print(f"Warning: Failed to retrieve memory: {e}")
                retrieved_memory_str = ""
        
        return super().predict_action()

    def get_action_context(self, action_elements: dict, observation: dict):
        # given the action and the observation, return the context for the action (+- 5 lines from the observation)
        # locate the element id in the observation, and return the +- 5 lines from the element
        action_context = ""
        action_str = action_elements.get("action", "") or action_elements.get("navigation action", "")
        if not action_str:
            return action_context

        element_id_match = re.search(r"\[(\d+)\]", action_str)
        if not element_id_match:
            return action_context
        element_id = element_id_match.group(1)

        obs_text = ""
        if isinstance(observation, dict):
            obs_text = observation.get("text", "")
        elif isinstance(observation, str):
            obs_text = observation
        if not obs_text:
            return action_context

        obs_lines = obs_text.splitlines()
        if not obs_lines:
            return action_context

        line_idx = None
        exact_pattern = re.compile(rf"\[{re.escape(element_id)}\]")
        for idx, line in enumerate(obs_lines):
            if exact_pattern.search(line):
                line_idx = idx
                break

        if line_idx is None:
            fallback_pattern = re.compile(rf"\b{re.escape(element_id)}\b")
            for idx, line in enumerate(obs_lines):
                if fallback_pattern.search(line):
                    line_idx = idx
                    break

        if line_idx is None:
            return action_context

        start_idx = max(0, line_idx - 5)
        end_idx = min(len(obs_lines), line_idx + 6)
        action_context = "\n".join(obs_lines[start_idx:end_idx])
        return action_context
    
    def act(self, objective, env, memory_graph: MemoryGraph = None, task_id=None):
        if self.replay_mode:
            return self._act_replay(objective=objective, env=env, memory_graph=memory_graph, task_id=task_id)
        return self._act_online(objective=objective, env=env, memory_graph=memory_graph, task_id=task_id)
    
    def _act_online(self, objective, env, memory_graph: MemoryGraph = None, task_id=None):
        """
        Act with memory graph integration.
        Creates Memory object, updates it during execution, and inserts into memory_graph at the end.
        This method overrides the parent act to integrate Memory updates after each step.
        """
        mg = memory_graph if memory_graph is not None else self.memory_graph
        self.current_task_id = task_id
        
        # Initialize parent state
        self.objective = objective
        self.sites = env.get_sites()
        self.tabs_info = env.get_tabs_info()
        observation = env.observation()
        url = env.get_url()
        self.update_online_state(url=url, observation=observation)

        observation_text = observation["text"] if isinstance(observation, dict) else observation
        observation_init_collection = f"MAIN_GOAL:\n{self.objective}\nOBSERVATION:\n{observation_text}\nURL:\n{url}\nTABS:\n{self.tabs_info}"
        if mg is not None:
            self.memory = Memory(goal=self.objective, observation=observation_init_collection)  # time is not used for webarena
        else:
            self.memory = None
        
        # Initialize actor, critic, and judge
        self.init_actor()
        self.init_critic()
        self.init_judge()
        
        # Main action loop with Memory updates
        last_action_str = None
        last_action_description_str = None
        while not env.done():
            observation = env.observation()
            url = env.get_url()
            self.tabs_info = env.get_tabs_info()
            self.update_online_state(url=url, observation=observation)
            self.actor.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            self.critic.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            self.judge.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            
            # Predict action (this will retrieve memory if available)
            action_elements, action_element_list = self.predict_action()
            action = action_elements["action"]
            navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")

            # Prepare for Memory update
            observation_text_before = self.actor.get_observation_text()
            url_info_before = self.actor.get_url_info()
            tabs_info_before = self.actor.get_tabs_info()
            observation_before_collection_str = f"OBSERVATION:\n{observation_text_before}\nURL:\n{url_info_before}\nTABS:\n{tabs_info_before}"

            if mg is not None:
                try:
                    self.memory.append(
                        action_t0=last_action_description_str,
                        observation_t1=observation_before_collection_str
                    )
                except Exception as e:
                    print(f"Warning: Failed to update Memory: {e}")
            
            # Store action string for Memory
            last_action_str = action
            if self.memory is not None:
                observation_before_collection_str = f"MAIN_GOAL:\n{self.objective}\nOBSERVATION:\n{observation_text_before}\nURL:\n{url_info_before}\nTABS:\n{tabs_info_before}"
                last_action_description_str = last_action_str
                try:
                    prompt = (
                        "You translate a low-level browser action into concise natural language. "
                        "Given the raw action string and the page observations before the action, "
                        "describe what the agent attempted with enough detail for future retrieval. "
                        "Include element targets (text/aria/ids) and intent. "
                        "Return one short sentence. Follow these action notes:\n"
                        "- branch [parent_plan_id] [new_subplan_intent]: create new subplan linked to parent plan ID (use previous plans). Example: branch [12] [Navigate to the \"Issue\" page to check all the issues.]\n"
                        "- prune [resume_plan_id] [reason]: return to previous plan state by ID when current plan impractical. Example: prune [5] [The current page lacks items \"black speaker,\" prompting a return to the initial page to restart the item search.]\n"
                        "- click [id]: click element by numeric ID; if no transition, element may be non-interactive—try another relevant element. Example: click [7]\n"
                        "- type [id] [content] [press_enter_after=0|1]: type into field by ID; Enter pressed unless flag is 0. Consider refining keywords if first attempt fails. Example: type [15] [Example University] [1]\n"
                        "- stop [answer]: finish interaction with answer; if no textual answer, use N/A with reasons and gathered info. Example: stop [5h 47min]\n"
                        "- note [content]: record important info for the task. Example: note [Spent $10 on 4/1/2024]\n"
                        "- go_back: return to the previously viewed page.\n"
                        "- goto [url] [open_in_new_tab=0|1]: To navigate directly to a full URL. Default is 1 to open in a new tab; set to 0 to stay in the current tab. Always include the complete URL (e.g., https://example.com/page). E.g., `goto [https://example.com/login] [1]`\n"
                        "- hover [id]: To hover over an element with its numerical ID (e.g., to reveal tooltips or dropdowns) without clicking. E.g., `hover [7]`\n"
                        "- tab_focus [tab_index]: To switch browser focus to a specific tab by its index (0-based unless otherwise specified). E.g., `tab_focus [1]`\n\n"
                        f"Raw action: {last_action_str}\n\n"
                        f"Observation before action:\n{observation_before_collection_str}\n\n"
                    )
                    last_action_description_str = f"Raw action: {last_action_str}\nAction description: {call_gpt(prompt=prompt)}"
                except Exception as e:
                    print(f"Warning: Failed to translate action: {e}")
                    last_action_description_str = last_action_str

            # Execute action
            status = env.step(navigation_action)
            if navigation_action and self.is_navigation(action=navigation_action) and status == False: # means invalid action
                flaw_node = self.actor.active_node
                flaw_node.note.append(f"STEP {self.get_step()}: You generate action \"{action}\", which has INVALID syntax. Strictly follow the action specifications.")          
            
            # Update actor history (parent class behavior)
            DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP = {
                "observation": observation,
                "action": action,
                "url": url,
                "plan": self.get_actor_active_plan(),
                "reason": action_elements.get("reason", ""),
                "observation highlight": action_elements.get("observation highlight", ""),
                "retained element ids": action_elements.get("retained element ids", []),
                "observation summary": action_elements.get("observation description", "")                  
            }
            self.actor.update_history(**DOCUMENTED_INTERACTION_ELEMENT_KEY_TO_CONTENT_MAP)
            self.actor.del_observation_node()
            assert self.actor.equal_history_length()

            # Log step if configured
            if len(action_element_list) > 1:
                if self.config.others.logging:
                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **action_elements,
                        **{f"actor {i}:{k}": _action_elements[k] for i, _action_elements in enumerate(action_element_list) for k in _action_elements.keys() if k != "input" and k != "instruction"}
                    )
            else:
                if self.config.others.logging:
                    self.log_step(
                        status=status if "status" in locals() and isinstance(status, dict) else env.status(),
                        plan=self.get_actor_active_plan(),
                        **action_elements,
                    )

        # After task completion, finalize memory and insert into memory_graph
        final_status = status if "status" in locals() and isinstance(status, dict) else env.status()
        
        if self.memory is not None:
            # Close memory to finalize it
            try:
                self.memory.close()

                # Insert memory into memory_graph if available and not in read-only mode
                if mg is not None and not self.read_only_memory:
                    try:
                        mg.insert(self.memory)
                        print("Successfully inserted memory into memory_graph")
                        # Print memory graph stats after insertion
                        print_memory_graph_stats(mg, context="After Insert")
                    except Exception as e:
                        print(f"Warning: Failed to insert memory into memory_graph: {e}")
                elif mg is not None and self.read_only_memory:
                    print("Read-only mode: Skipping memory insertion into memory_graph")
            except Exception as e:
                print(f"Warning: Failed to close Memory: {e}")
        
        return final_status

    def _act_replay(self, objective, env=None, memory_graph: MemoryGraph = None, task_id=None):
        """
        Replay a stored trajectory to populate memory, then insert into memory_graph.
        """
        mg = memory_graph if memory_graph is not None else self.memory_graph
        self.current_task_id = task_id

        # Reset trajectory step counter
        self.trajectory_step = 0

        if not self.trajectory_data or len(self.trajectory_data) == 0:
            raise RuntimeError("Trajectory data is required for replay but was not provided.")

        first_step = self.trajectory_data[0]
        initial_observation = first_step.get("observation", "")
        initial_url = first_step.get("url", "")

        # Initialize parent state using trajectory data instead of a live environment
        self.objective = objective if objective is not None else first_step.get("objective")
        self.sites = env.get_sites() if env is not None else []
        self.tabs_info = first_step.get("tabs", "N/A")

        # Initialize Memory object for this task using saved observation text
        observation_text = initial_observation["text"] if isinstance(initial_observation, dict) else initial_observation
        observation_init_collection = f"OBSERVATION:\n{observation_text}\nURL:\nCURRENT URL: {initial_url}\nTABS:\n{self.tabs_info}"
        self.memory = Memory(goal=self.objective, observation=observation_init_collection)  # time is not used for webarena
        
        # Initialize actor, critic, and judge
        self.init_actor()
        self.init_critic()
        self.init_judge()
        
        # Main action loop using stored trajectory steps (no live environment)
        final_status = {"reward": 0.0, "success": 0.0, "done": False, "num_actions": 0}
        last_action_str = None
        last_action_description_str = None
        while self.trajectory_step < len(self.trajectory_data):
            step_idx = self.trajectory_step
            step_data = self.trajectory_data[step_idx]

            current_observation = step_data.get("observation", "")
            current_url = step_data.get("url", initial_url)
            current_tabs = step_data.get("tabs", self.tabs_info)
            current_observation_collection = f"OBSERVATION:\n{current_observation}\nURL:\nCURRENT URL: {current_url}\nTABS:\n{current_tabs}"

            # Update Memory with t observation and t-1 action 
            try:
                self.memory.append(
                    action_t0=last_action_description_str,
                    observation_t1=current_observation_collection
                )
            except Exception as e:
                print(f"Warning: Failed to update Memory: {e}")

            # Predict action from trajectory
            action_elements, action_element_list = self.predict_action(memory_graph=mg)
            action = action_elements["action"]
            
            # Store action string for Memory
            last_action_str = action
            observation_before_collection_str = current_observation_collection
            last_action_description_str = last_action_str
            try:
                prompt = (
                    "You translate a low-level browser action into concise natural language. "
                    "Given the raw action string and the page observations before the action, "
                    "describe what the agent attempted with enough detail for future retrieval. "
                    "Include element targets (text/aria/ids) and intent. "
                    "Return one short sentence. Follow these action notes:\n"
                    "- branch [parent_plan_id] [new_subplan_intent]: create new subplan linked to parent plan ID (use previous plans). Example: branch [12] [Navigate to the \"Issue\" page to check all the issues.]\n"
                    "- prune [resume_plan_id] [reason]: return to previous plan state by ID when current plan impractical. Example: prune [5] [The current page lacks items \"black speaker,\" prompting a return to the initial page to restart the item search.]\n"
                    "- click [id]: click element by numeric ID; if no transition, element may be non-interactive—try another relevant element. Example: click [7]\n"
                    "- type [id] [content] [press_enter_after=0|1]: type into field by ID; Enter pressed unless flag is 0. Consider refining keywords if first attempt fails. Example: type [15] [Example University] [1]\n"
                    "- stop [answer]: finish interaction with answer; if no textual answer, use N/A with reasons and gathered info. Example: stop [5h 47min]\n"
                    "- note [content]: record important info for the task. Example: note [Spent $10 on 4/1/2024]\n"
                    "- go_back: return to the previously viewed page.\n"
                    "- goto [url] [open_in_new_tab=0|1]: To navigate directly to a full URL. Default is 1 to open in a new tab; set to 0 to stay in the current tab. Always include the complete URL (e.g., https://example.com/page). E.g., `goto [https://example.com/login] [1]`\n"
                    "- hover [id]: To hover over an element with its numerical ID (e.g., to reveal tooltips or dropdowns) without clicking. E.g., `hover [7]`\n"
                    "- tab_focus [tab_index]: To switch browser focus to a specific tab by its index (0-based unless otherwise specified). E.g., `tab_focus [1]`\n\n"
                    f"Raw action: {last_action_str}\n\n"
                    f"Observation before action:\n{observation_before_collection_str}\n\n"
                )
                last_action_description_str = f"Raw action: {last_action_str}\nAction description: {call_gpt(prompt=prompt)}"
            except Exception as e:
                print(f"Warning: Failed to translate action: {e}")
                last_action_description_str = last_action_str
            print(f"Action description: {last_action_description_str}")
            
            # Compose status from stored trajectory metadata
            status = {
                "done": step_data.get("done", self.trajectory_step >= len(self.trajectory_data)),
                "reward": step_data.get("reward", 0.0),
                "success": step_data.get("success", 0.0),
                "num_actions": step_data.get("num_actions", self.trajectory_step)
            }
            final_status = status
        
        # After task completion, finalize memory and insert into memory_graph
        if self.memory is not None:
            # Close memory to finalize it
            try:
                self.memory.close()
                
                # Insert memory into memory_graph if available and not in read-only mode
                if mg is not None and not self.read_only_memory:
                    try:
                        mg.insert(self.memory)
                        print("Successfully inserted memory into memory_graph")
                        print_memory_graph_stats(mg, context="After Insert")
                    except Exception as e:
                        print(f"Warning: Failed to insert memory into memory_graph: {e}")
                elif mg is not None and self.read_only_memory:
                    print("Read-only mode: Skipping memory insertion into memory_graph")
            except Exception as e:
                print(f"Warning: Failed to close Memory: {e}")
        
        return final_status

    def log_step(self, status, **kwargs):
        def serialize_message_list(message_list):
            if not isinstance(message_list, list):
                return message_list
            return "".join([m[1] for m in message_list if m[0]=="text"])
        data_to_log = {}
        data_to_log['objective'] = self.objective
        data_to_log['url'] = self.online_url
        data_to_log['observation'] = self.actor.get_observation_text()
        data_to_log['retrieved memory'] = self.actor.get_retrieved_memory()
        data_to_log['tabs'] = self.actor.get_tabs_info()
        for (k, v) in status.items():
            data_to_log[k] = v
        for k in kwargs.keys():
            try:
                json.dumps(kwargs[k])
                data_to_log[k.replace(" ", "_")] = kwargs[k] if not "input" in k else serialize_message_list(kwargs[k])
            except:
                pass
        self.trajectory.append(data_to_log)
