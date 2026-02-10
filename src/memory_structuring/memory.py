import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_structuring.structuring_inference import get_subgoal, get_reward, get_state, get_procedural, get_semantic
from utils import call_gpt, call_qwen, get_embedding, get_similarity
import re

class Memory:
    
    def __init__(self, goal, observation, time: int = 0):

        self.time = time
        self.memory = {}
        self.memory["goal"] = goal
        self.memory["episodic"] = []
        self.memory["procedural"] = []
        self.memory["semantic"] = []
        self.memory_embedding = {}
        self.memory_embedding["procedural"] = []
        self.memory_embedding["semantic"] = []

        self.observation_t0 = observation
        self.goal = goal
        self.trajectory = []
        self.step = {}
        self.state_t0 = ""

    def append(self, action_t0, observation_t1):

        subgoal = get_subgoal(
            goal = self.goal, 
            state_t0 = self.state_t0,
            observation_t0 = self.observation_t0,
            action_t0 = action_t0
        )

        reward = get_reward(
            goal = subgoal,
            state_t0 = self.state_t0,
            action_t0 = action_t0,
            observation_t1 = observation_t1
        )

        similarity_subgoal = -1

        if not len(self.trajectory) == 0:
            similarity_subgoal = get_similarity(get_embedding(self.trajectory[-1]["subgoal"]), get_embedding(subgoal))
            if similarity_subgoal < 0.75:
                self.memory["episodic"].append(self.trajectory)
                self.trajectory = []

        self.trajectory.append({
            "subgoal": subgoal,
            "state": self.state_t0,
            "observation": self.observation_t0,
            "action": action_t0,
            "reward": reward,
            "similarity_subgoal": similarity_subgoal,
            "time": self.time
        })

        self.state_t0 = get_state(
            goal = self.goal,
            state_t0 = self.state_t0,
            action_t0 = action_t0,
            observation_t1 = observation_t1
        )
        self.observation_t0 = observation_t1
        
    def close(self):
        
        self.memory["episodic"].append(self.trajectory)
        self.trajectory = []
        for j, trajectory in enumerate(self.memory["episodic"]):
            
            trajectory_str = ""
            for i, step in  enumerate(trajectory):
                trajectory_str += f"Step {i}:\n-State: {step['state']}\n-Action: {step['action']}\n-Reward: {step['reward']}\n"
                new_semantic = get_semantic(step, j, i, self.time)
                self.memory["semantic"] += new_semantic
                for semantic_memory in new_semantic:
                    self.memory_embedding["semantic"].append({
                        "semantic_memory": get_embedding(semantic_memory["semantic_memory"]),
                        "tags": [get_embedding(tag) for tag in semantic_memory["tags"]]}
                    )
             
            procedural_memory, goal, _return = get_procedural(trajectory=trajectory_str)

            self.memory["procedural"].append({
                "subgoal": goal,
                "procedural_memory": procedural_memory,
                "trajectory_num": j,
                "time": self.time,
                "return": _return
            })
            self.memory_embedding["procedural"].append({
                "subgoal": get_embedding(goal)
            })