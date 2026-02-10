import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, Union, final
import sys
import re

class EpisdoicNode:
    def __init__(self, episodic_id: int):
        self.episodic_id = episodic_id

    def get_episodic_memory_semantic(self):
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/episodic_memory/episodic_memory_{self.episodic_id}.json","r",) as input:
            _json = json.load(input)
            episodic_memory = _json['observation'] + _json['time']
        return episodic_memory
    
    def get_episodic_memory_procedural(self):
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/episodic_memory/episodic_memory_{self.episodic_id}.json","r",) as input:
            _json = json.load(input)
            episodic_memory = _json['action'] + _json['time']
        return episodic_memory
    
    def get_episodic_memory(self):
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/episodic_memory/episodic_memory_{self.episodic_id}.json","r",) as input:
            _json = json.load(input)
            if "observation" in _json and "action" in _json and "time" in _json:
                episodic_memory = _json['observation'] + '\n' + _json['action'] + '\n' +  _json['time']
            elif "episodic_memory" in _json:
                episodic_memory = _json['episodic_memory']
            else:
                raise ValueError(f"Invalid episodic memory")
        return episodic_memory

class SemanticNode:
    def __init__(self, semantic_memory:dict, semantic_memory_embedding, semantic_id: int, semantic_memory_str:str="", time: int=None, son=[], cache_embedding_gate: bool = True, is_active: bool = True):
        self.semantic_memory_str = semantic_memory_str
        self.embedding = semantic_memory_embedding
        self.tags = []
        self.tag_nodes = []
        self.semantic_id = semantic_id
        self.time = time
        self.episodic_nodes = []
        self.bro_semantic_nodes = []
        self.Credibility = 10
        self.updated = False
        self.is_active = is_active
        self.son_semantic = son
        
        # embedding lazy-load state
        self._embedding = semantic_memory_embedding
        self._cache_embedding_gate = cache_embedding_gate

    def _load_embedding_from_json(self) -> np.ndarray:
        dir_path = os.environ.get("DIR_PATH", None)
        if not dir_path:
            raise RuntimeError("Missing DIR_PATH")
        path = os.path.join(dir_path, "semantic_memory", f"semantic_embedding_{self.semantic_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        emb_list = obj["semantic_embedding"]
        return np.asarray(emb_list, dtype=np.float32)

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            emb = self._load_embedding_from_json()
            if self._cache_embedding_gate:
                self._embedding = emb
            return emb

        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Union[list[float], np.ndarray]) -> None:
        self._embedding = value

    def unload_embedding(self) -> None:
        self._embedding = None
    
    def get_semantic_memory(self):
        if self.semantic_memory_str != "":
            return self.semantic_memory_str
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/semantic_memory/semantic_memory_{self.semantic_id}.json","r",) as input:
            _json = json.load(input)
            if _json['time'] == None:
                _json['time'] = ""
            if not isinstance(_json['time'], str):
                _json['time'] = str(_json['time'])
            semantic_memory = _json['semantic_memory']
        return semantic_memory

class TagNode:
    def __init__(self, tag: str, tag_embedding, tag_id: int, time: int, cache_embedding_gate: bool = True):
        self.tag = tag
        self.tag_id = tag_id
        self.embedding = tag_embedding
        self.semantic_nodes = []
        self.importance = 1
        self.time = time
        
        # embedding lazy-load state
        self._embedding = tag_embedding
        self._cache_embedding_gate = cache_embedding_gate

    def _load_embedding_from_json(self) -> np.ndarray:
        dir_path = os.environ.get("DIR_PATH", None)
        if not dir_path:
            raise RuntimeError("Missing DIR_PATH")
        path = os.path.join(dir_path, "tag", f"tag_embedding_{self.tag_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        emb_list = obj["tag_embedding"]
        return np.asarray(emb_list, dtype=np.float32)

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            emb = self._load_embedding_from_json()
            if self._cache_embedding_gate:
                self._embedding = emb
            return emb

        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Union[list[float], np.ndarray]) -> None:
        self._embedding = value

    def unload_embedding(self) -> None:
        self._embedding = None
    
class ProceduralNode:
    def __init__(self, procedural_memory: Dict, procedural_memory_embedding, procedural_id: int, time: int, cache_embedding_gate: bool = True):
        # self.embedding = procedural_memory_embedding
        self.procedural_id = procedural_id
        self.subgoals = []
        self.subgoal_nodes = []
        self.time = time
        self.episodic_nodes = []
        self.Return = procedural_memory['return']
        
        # embedding lazy-load state
        self._embedding = procedural_memory_embedding
        self._cache_embedding_gate = cache_embedding_gate
    
    def _load_embedding_from_json(self) -> np.ndarray:
        dir_path = os.environ.get("DIR_PATH", None)
        if not dir_path:
            raise RuntimeError("Missing DIR_PATH")
        path = os.path.join(dir_path, "procedural_memory", f"procedural_embedding_{self.procedural_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        emb_list = obj["procedural_embedding"]
        return np.asarray(emb_list, dtype=np.float32)

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            emb = self._load_embedding_from_json()
            if self._cache_embedding_gate:
                self._embedding = emb
            return emb

        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Union[list[float], np.ndarray]) -> None:
        self._embedding = value

    def unload_embedding(self) -> None:
        self._embedding = None
    
    def get_procedural_memory(self):
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/procedural_memory/procedural_memory_{self.procedural_id}.json","r",) as input:
            _json = json.load(input)
            procedural_memory = _json['procedural_memory'] + _json['time']
        return procedural_memory

class SubgoalNode:
    def __init__(self, subgoal: str, subgoal_embedding, subgoal_id: int, time: int, subgoal_nodes: List[any] = [], cache_embedding_gate: bool = True):
        self.subgoal = subgoal
        self.subgoal_id = subgoal_id
        # self.embedding = subgoal_embedding
        self.child_subgoal = []
        self.procedural_nodes = []
        for subgoal_node in subgoal_nodes:
            self.child_subgoal.append(subgoal_node)
        self.activate = False
        self.importance = 1
        self.edge = []
        self.time = time
        
        # embedding lazy-load state
        self._embedding = subgoal_embedding
        self._cache_embedding_gate = cache_embedding_gate

    def _load_embedding_from_json(self) -> np.ndarray:
        dir_path = os.environ.get("DIR_PATH", None)
        if not dir_path:
            raise RuntimeError("Missing DIR_PATH")
        path = os.path.join(dir_path, "subgoal", f"subgoal_{self.subgoal_id}.json")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        emb_list = obj["subgoal_embedding"]
        return np.asarray(emb_list, dtype=np.float32)

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            emb = self._load_embedding_from_json()
            if self._cache_embedding:
                self._embedding = emb
            return emb

        if isinstance(self._embedding, list):
            self._embedding = np.asarray(self._embedding, dtype=np.float32)
        return self._embedding

    @embedding.setter
    def embedding(self, value: Union[list[float], np.ndarray]) -> None:
        self._embedding = value

    def unload_embedding(self) -> None:
        self._embedding = None
    
    def activation(self, procedural_nodes: List[ProceduralNode] = []):
        for procedural_node in procedural_nodes:
            self.procedural_nodes.append(procedural_node)
        self.activate = True
    
    def get_subgoal(self):
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+f"/subgoal/subgoal_{self.subgoal_id}.json","r",) as input:
            _json = json.load(input)
            subgoal = _json['subgoal']
        return subgoal