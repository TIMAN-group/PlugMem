import os
import json
import re
from beartype.typing import Any, Dict, List

from AgentOccam.AgentOccam import AgentOccam, PlanTreeNode, Actor
from webagents_step.utils.data_prep import DotDict

from memory_retrieving.retrieving_inference import get_plan
from memory_reasoning.prompt_reasoning_webarena import WebarenaProceduralPrompt
from utils import call_gpt, get_embedding, get_similarity


def print_rag_memory_stats(store, context: str = ""):
    if store is None:
        return
    total_entries = len(store.entries)
    print("\n" + "=" * 60)
    if context:
        print(f"RAG Memory Stats - {context}")
    else:
        print("RAG Memory Stats")
    print("=" * 60)
    print(f"Total entries: {total_entries}")
    print("=" * 60 + "\n")


class ProceduralRAGStore:
    """
    Conventional dense retrieval store keyed by objective (NV-Embed-v2).
    Stores concatenated OBJECTIVE/ACTIONS/OBSERVATIONS entries only.
    """
    def __init__(self, load_from_disk: bool = False, refresh_embeddings: bool = False):
        self.entries = []
        self.storage_path = None
        dir_path = os.environ.get("DIR_PATH", None)
        if dir_path:
            self.storage_path = os.path.join(dir_path, "no_structure_rag_memory.jsonl")
        if load_from_disk:
            self._load_from_disk(refresh_embeddings=refresh_embeddings)

    def _load_from_disk(self, refresh_embeddings: bool = False):
        if not self.storage_path or not os.path.exists(self.storage_path):
            print("RAG memory load skipped: storage file not found.")
            return
        try:
            with open(self.storage_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if "embedding" in entry and "objective" in entry and "entry_text" in entry:
                        self.entries.append(entry)
            if refresh_embeddings and self.entries:
                updated_entries = []
                for entry in self.entries:
                    embedding = get_embedding(entry["objective"])
                    if embedding is None:
                        embedding = entry.get("embedding")
                    updated_entries.append({
                        "objective": entry["objective"],
                        "entry_text": entry["entry_text"],
                        "embedding": embedding
                    })
                self.entries = updated_entries
                with open(self.storage_path, "w") as f:
                    for entry in self.entries:
                        f.write(json.dumps(entry) + "\n")
            print(f"Loaded {len(self.entries)} RAG memory entries.")
        except Exception as e:
            print(f"Warning: Failed to load RAG memory entries: {e}")

    def _persist_entry(self, entry: dict):
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to persist RAG memory entry: {e}")

    def add_entry(self, objective: str, entry_text: str):
        embedding = get_embedding(objective)
        if embedding is None:
            print("Warning: Failed to embed objective; skipping memory entry.")
            return
        entry = {
            "objective": objective,
            "entry_text": entry_text,
            "embedding": embedding
        }
        self.entries.append(entry)
        self._persist_entry(entry)

    def retrieve_by_objective(self, objective: str):
        if not self.entries:
            return None, None
        query_embedding = get_embedding(objective)
        if query_embedding is None:
            print("Warning: Failed to embed query objective for retrieval.")
            return None, None
        best_entry = None
        best_score = -1.0
        for entry in self.entries:
            score = get_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry, best_score

    @staticmethod
    def build_memory_entry(objective: str, actions: List[str], observations: List[str]) -> str:
        lines = [f"OBJECTIVE: {objective}"]
        max_len = max(len(actions), len(observations))
        for i in range(max_len):
            obs = observations[i] if i < len(observations) else "N/A"
            act = actions[i] if i < len(actions) else "N/A"
            lines.append(f"OBSERVATION_{i+1}: {obs}")
            lines.append(f"ACTION_{i+1}: {act}")
        return "\n".join(lines)


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
    AgentOccam with procedural-only RAG memory (no graph structure).
    """
    def __init__(self, 
                config: DotDict = None,
                prompt_dict: Dict[str, Any] = None, 
                rag_store: ProceduralRAGStore = None,
                read_only_memory: bool = False):
        self.rag_store = rag_store
        self.read_only_memory = read_only_memory  # If True, don't insert memories
        self.current_task_id = None
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

    def predict_action(self, rag_store: ProceduralRAGStore = None):
        """
        Predict action with procedural-only RAG retrieval.
        """
        store = rag_store if rag_store is not None else self.rag_store
        
        # Retrieve relevant memory if available
        retrieved_memory_str = ""
        if store is not None and self.objective is not None:
            try:
                goal = self.objective
                observation_text = self.actor.get_observation_text()
                url_info_str = self.actor.get_url_info()
                tabs_info_str = self.actor.get_tabs_info()
                observation_collection_str = (
                    f"MAIN_GOAL:\n{goal}\nOBSERVATION:\n{observation_text}\nURL:\n{url_info_str}\nTABS:\n{tabs_info_str}"
                )

                entry, score = store.retrieve_by_objective(goal)
                if entry is not None:
                    next_subgoal, _ = get_plan(
                        goal=goal,
                        subgoal="",
                        state="",
                        observation=observation_collection_str
                    )
                    prompt_obj = WebarenaProceduralPrompt()
                    variables = {
                        "goal": goal,
                        "subgoal": next_subgoal,
                        "state": "",
                        "observation": observation_collection_str,
                        "procedural_memory": entry["entry_text"],
                        "semantic_memory": "",
                        "episodic_memory_semantic": "",
                        "episodic_memory_procedural": "",
                        "time": ""
                    }
                    messages = prompt_obj.render(variables)
                    messages = [{"role": m.role, "content": m.content} for m in messages]
                    retrieved_memory_str = call_gpt(messages=messages, model_id="gpt-4o")
                    self.actor.retrieved_memory = retrieved_memory_str
                    print(f"RAG score: {score}, Retrieved memory: {retrieved_memory_str}")
                else:
                    self.actor.retrieved_memory = ""
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
    
    def act(self, objective, env, rag_store: ProceduralRAGStore = None, task_id=None):
        """
        Act with procedural-only RAG memory.
        After each task, store concatenated OBJECTIVE/ACTIONS/OBSERVATIONS.
        """
        store = rag_store if rag_store is not None else self.rag_store
        self.current_task_id = task_id
        
        # Initialize parent state
        self.objective = objective
        self.sites = env.get_sites()
        self.tabs_info = env.get_tabs_info()
        observation = env.observation()
        url = env.get_url()
        self.update_online_state(url=url, observation=observation)

        action_history = []
        observation_history = []
        
        # Initialize actor, critic, and judge
        self.init_actor()
        self.init_critic()
        self.init_judge()
        
        # Main action loop with Memory updates
        last_action_str = None
        while not env.done():
            observation = env.observation()
            url = env.get_url()
            self.tabs_info = env.get_tabs_info()
            self.update_online_state(url=url, observation=observation)
            self.actor.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            self.critic.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            self.judge.update_online_state(url=url, observation=observation, tabs=self.tabs_info)
            # observation_text_before = observation["text"] if isinstance(observation, dict) else observation
            
            # Predict action (this will retrieve memory if available)
            action_elements, action_element_list = self.predict_action()
            action = action_elements["action"]
            navigation_action = action_elements["action"] if not action_elements.get("navigation action", "") else action_elements.get("navigation action", "")

            # Prepare for RAG memory entry pieces
            observation_text_before = self.actor.get_observation_text()
            url_info_before = self.actor.get_url_info()
            tabs_info_before = self.actor.get_tabs_info()
            observation_before_collection_str = (
                f"OBSERVATION:\n{observation_text_before}\nURL:\n{url_info_before}\nTABS:\n{tabs_info_before}"
            )
            observation_history.append(observation_before_collection_str)
            
            # Store action string for Memory
            last_action_str = action
            observation_before_collection_str = (
                f"MAIN_GOAL:\n{self.objective}\nOBSERVATION:\n{observation_text_before}\nURL:\n{url_info_before}\nTABS:\n{tabs_info_before}"
            )
            action_history.append(last_action_str)

            
            # Execute action
            status = env.step(navigation_action)
            if navigation_action and self.is_navigation(action=navigation_action) and status == False: # means invalid action
                flaw_node = self.actor.active_node
                flaw_node.note.append(f"STEP {self.get_step()}: You generate action \"{action}\", which has INVALID syntax. Strictly follow the action specifications.")          
            
            # Get new observation after action
            # new_observation = env.observation()
            # observation_text = new_observation["text"] if isinstance(new_observation, dict) else new_observation
            # observation_text_after = self.actor.get_observation_text()
            # url_info_after = self.actor.get_url_info()
            # tabs_info_after = self.actor.get_tabs_info()
            
            
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

        # After task completion, persist RAG memory entry
        if store is not None and not self.read_only_memory:
            entry_text = store.build_memory_entry(self.objective, action_history, observation_history)
            store.add_entry(self.objective, entry_text)
            print("Successfully inserted RAG memory entry")
            print_rag_memory_stats(store, context="After Insert")
        elif store is not None and self.read_only_memory:
            print("Read-only mode: Skipping RAG memory insertion")
        
        return status if "status" in locals() else env.status()

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
