import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional, final, Any, Callable
import sys
import random
import heapq

import re
new_sys_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if new_sys_dir not in sys.path:
    sys.path.append(new_sys_dir)
from utils import  save_episodic_hpqa_ver, save_semantic_hpqa_ver, save_procedural_hpqa_ver, \
        save_subgoal_hpqa_ver, save_tag_hpqa_ver, update_semantic_hpqa_ver, update_tag_hpqa_ver,\
        save_episodic_longmem_ver, save_semantic_longmem_ver, save_procedural_longmem_ver, \
        save_subgoal_longmem_ver, save_semantic_webarena_ver, save_subgoal_webarena_ver, \
        save_procedural_webarena_ver, get_embedding, get_similarity, call_gpt, call_qwen,  set_logger
        
from memory_retrieving.graph_node import SemanticNode, ProceduralNode, TagNode, SubgoalNode, EpisdoicNode
from memory_structuring.memory import Memory
from memory_retrieving.retrieving_inference import get_plan, get_new_semantic, get_new_subgoal, get_mode
from memory_retrieving.value_longmemeval import TagEqual, TagRelevant, SemanticEqual, SemanticRelevant, SubgoalEqual, SubgoalRelevant, ProceduralEqual, ProceduralRelevant, SemanticRelevant4Episodic
from memory_retrieving.value_base import ValueBase
from prompt_base import PromptBase
from memory_reasoning.prompt_reasoning import DefaultEpisodicPrompt, DefaultSemanticPrompt, DefaultProceduralPrompt

import logging
from tqdm import tqdm

class MemoryGraph:
    def __init__(self, 
                 tag_equal: ValueBase=TagEqual(), 
                 tag_relevant: ValueBase=TagRelevant(), 
                 semantic_equal: ValueBase=SemanticEqual(), 
                 semantic_relevant: ValueBase=SemanticRelevant(), 
                 subgoal_equal: ValueBase=SubgoalEqual(), 
                 subgoal_relevant: ValueBase=SubgoalRelevant(), 
                 procedural_equal: ValueBase=ProceduralEqual(), 
                 procedural_relevant: ValueBase=ProceduralRelevant(),
                 log_file: str=None
                 ):

        self.tag_equal = tag_equal
        self.tag_relevant = tag_relevant
        self.semantic_equal = semantic_equal
        self.semantic_relevant = semantic_relevant
        self.semantic_relevant4episodic = SemanticRelevant4Episodic()
        self.subgoal_equal = subgoal_equal
        self.subgoal_relevant = subgoal_relevant
        self.procedural_equal = procedural_equal
        self.procedural_relevant = procedural_relevant
        self.procedural_nodes = []
        self.semantic_nodes = []
        self.tag_nodes = []
        self.subgoal_nodes = []
        self.episodic_nodes = []
        self.semantic_time = 0
        self.procedural_time = 0
        self.log_file = log_file
        self.logger = set_logger(log_file)
        self.session_ids = []
        
        self.tag2node: Dict[str, TagNode] = {}
        self.subgoal2node: Dict[str, SubgoalNode] = {}
        self.episodic_id2node: Dict[int, EpisdoicNode] = {}
        self.semantic_id2node: Dict[int, SemanticNode] = {}
        self.procedural_id2node: Dict[int, ProceduralNode] = {}
        self.subgoal_id2node: Dict[int, SubgoalNode] = {}
        self.tag_id2node: Dict[int, TagNode] = {}
        
    
    def return_logger(self,) -> logging.Logger:
        return self.logger
    
    # ====== LongMemEval specific ============
    def build_mem_from_disk_lme_ver(self, file_path: str):
        #print(file_path)
        if not os.path.exists(file_path):
            raise ValueError("file_path not exists.")
        with open(file_path, "r") as f:
            graph_data = json.load(f)
        
        self.session_ids = graph_data["session_ids"]

        sem_items: list[dict[str, Any]] = graph_data['semantic_nodes']
        tag_items: list[dict[str, Any]] = graph_data['tag_nodes']
        subgoal_items: list[dict[str, Any]] = graph_data['subgoal_nodes']
        procedural_items: list[dict[str, Any]] = graph_data['procedural_nodes']

        embedding_texts: List[str] = []
        embedding_texts.extend([x.get("semantic_memory", "") for x in sem_items])
        embedding_texts.extend([x.get("tag", "") for x in tag_items])
        embedding_texts.extend([x.get("subgoal", "") for x in subgoal_items])
        embedding_texts.extend([x.get("procedural_memory", "") for x in procedural_items])
        emb_cache = self._parallel_get_embeddings(embedding_texts)
        
        # -------------------------
        # 2) Load episodic nodes
        # -------------------------
        has_epis = True
        epis_id2node: dict[int, EpisdoicNode] = {}
        epis_items: list[dict[str, Any]] = graph_data['episodic_nodes']
        for epis_item in epis_items:
            episodic_node = EpisdoicNode(episodic_id=epis_item["episodic_id"])
            self.episodic_nodes.append(episodic_node)
            episodic_node.observation = epis_item.get("observation", "")
            episodic_node.action = epis_item.get("action", "")
            episodic_node.time = epis_item.get("time", "")
            episodic_node.session_id = epis_item.get("session_id", None)

        epis_id2node = {node.episodic_id: node for node in self.episodic_nodes}

        # -------------------------
        # 2) Load semantic nodes
        # -------------------------
        for sem_item in sem_items:
            semantic_id = sem_item["semantic_id"]
            episodic_ids = sem_item.get("episodic_nodes", None)
            #episodic_id = sem_item.get("episodic_id", None)
            semantic_memory_str = sem_item["semantic_memory"]
            semantic_embedding = emb_cache.get(semantic_memory_str)
            if semantic_embedding is None and semantic_memory_str:
                semantic_embedding = get_embedding(semantic_memory_str)
            #bro_semantic_ids = sem_item.get("bro_semantic_ids", [])
            tags = sem_item.get("tags", [])
            _time = sem_item.get("time", 0)
            date = sem_item.get("date", "")

            if not isinstance(_time, int):
                _time = 0

            semantic_node = SemanticNode(semantic_memory=sem_item,
                                         semantic_memory_embedding=semantic_embedding,
                                         semantic_id=semantic_id,
                                         semantic_memory_str=semantic_memory_str,
                                         time=_time,
                                         date=date)
            semantic_node.session_id = sem_item.get("session_id", None)
            # Only link episodic if episodic nodes were loaded and the id exists
            if has_epis:
                for episodic_id_item in episodic_ids:
                    epis_node = epis_id2node.get(episodic_id_item)
                    if epis_node is not None:
                        semantic_node.episodic_nodes.append(epis_node)
                        epis_node.semantic_nodes.append(semantic_node)
            # store ids temporarily, then replace ids with true nodes
            #semantic_node.bro_semantic_nodes = bro_semantic_ids if isinstance(bro_semantic_ids, list) else []
            self.semantic_nodes.append(semantic_node)
            # keep tags on semantic node
            for tag in tags:
                semantic_node.tags.append(tag)
            self.semantic_time = max(self.semantic_time, _time + 1)

        sem_id2node: dict[int, SemanticNode] = {node.semantic_id: node for node in self.semantic_nodes}
        
        #self.logger.info("load all semantic nodes")

        # -------------------------
        # 3) Load tag nodes
        # -------------------------
        for tag_item in tag_items:
            tag_id = tag_item["tag_id"]
            tag = tag_item["tag"]
            tag_embedding = emb_cache.get(tag)
            if tag_embedding is None and tag:
                tag_embedding = get_embedding(tag)
            semantic_ids = tag_item.get("semantic_nodes", [])
            _time = tag_item.get("time", 0)
            importance = tag_item.get("importance", 1)

            tag_node = TagNode(tag=tag,tag_embedding=tag_embedding,tag_id=tag_id,time=_time,)
            tag_node.importance = importance

            for sid in semantic_ids:
                semantic_node = sem_id2node.get(sid)
                if semantic_node is None:
                    continue
                # TagNode -> SemanticNode
                tag_node.semantic_nodes.append(semantic_node)
                # SemanticNode -> TagNode
                semantic_node.tag_nodes.append(tag_node)
                # ensure semantic_node.tags contains this tag
                if hasattr(semantic_node, "tags") and tag_node.tag not in semantic_node.tags:
                    semantic_node.tags.append(tag_node.tag)
            self.tag_nodes.append(tag_node)

        #self.logger.info("load all tag nodes")
        
        # -------------------------
        # 4) Load subgoal nodes
        # -------------------------
        for i, subgoal_item in enumerate(subgoal_items):
            subgoal_id = i
            subgoal_str = subgoal_item["subgoal"]
            subgoal_embedding = emb_cache.get(subgoal_str)
            if subgoal_embedding is None and subgoal_str:
                subgoal_embedding = get_embedding(subgoal_str)
            subgoal_time = subgoal_item.get("time", 0)
            subgoal_node = SubgoalNode(subgoal_str, subgoal_embedding, subgoal_id, subgoal_time)
            self.subgoal_nodes.append(subgoal_node)
        
        subgoal_id2node: dict[int, SubgoalNode] = {node.subgoal_id: node for node in self.subgoal_nodes}
        #self.logger.info(f"load all subgoal nodes")
        
        # -------------------------
        # 5) Load procedural nodes
        # -------------------------
        for procedural_item in procedural_items:
            procedural_id = procedural_item["procedural_id"]
            episodic_ids = procedural_item.get("episodic_nodes", None)
            procedural_memory_str = procedural_item["procedural_memory"]
            procedural_memory_embedding = emb_cache.get(procedural_memory_str)
            if procedural_memory_embedding is None and procedural_memory_str:
                procedural_memory_embedding = get_embedding(procedural_memory_str)
            procedural_node = ProceduralNode(procedural_memory = procedural_item,
                                        procedural_memory_embedding = procedural_memory_embedding,
                                        procedural_id = procedural_id, 
                                        time = procedural_item["time"])
            # Only link episodic if episodic nodes were loaded and the id exists
            if has_epis:
                if isinstance(episodic_ids, list):
                    for episodic_id_item in episodic_ids:
                        epis_node = epis_id2node.get(episodic_id_item)
                        if epis_node is not None:
                            procedural_node.episodic_nodes.append(epis_node)
                elif episodic_ids is None:
                    pass
                else:
                    raise ValueError("episodic_ids must be a list when provided in procedural_memory_xx.json")
                    
            subgoal_id = procedural_item.get("subgoal_node", None)
            if subgoal_id is not None:
                subgoal_node = subgoal_id2node.get(subgoal_id,None)
                if subgoal_node is not None:
                    procedural_node.subgoal_nodes.append(subgoal_node)
                    subgoal_node.procedural_nodes.append(procedural_node)
                else:
                    raise ValueError(f"subgoal_id {subgoal_id} in procedural_memory_{procedural_id}.json does not exist in loaded subgoal nodes.")
            
            self.procedural_nodes.append(procedural_node)
        #self.logger.info(f"load all procedural nodes")

        # -------------------------
        # 6) Build id2node or str2node dicts
        # -------------------------
        self.tag2node = {x.tag: x for x in self.tag_nodes}
        self.subgoal2node = {x.subgoal: x for x in self.subgoal_nodes}
        self.episodic_id2node = {x.episodic_id: x for x in self.episodic_nodes}
        self.semantic_id2node = {x.semantic_id: x for x in self.semantic_nodes}
        self.procedural_id2node = {x.procedural_id: x for x in self.procedural_nodes}
        self.subgoal_id2node = {x.subgoal_id: x for x in self.subgoal_nodes}
        self.tag_id2node = {x.tag_id: x for x in self.tag_nodes}
        
        node_num_stat = {
            "semantic_nodes": len(self.semantic_nodes),
            "tag_nodes": len(self.tag_nodes),
            "episodic_nodes": len(self.episodic_nodes),
            "procedural_nodes": len(self.procedural_nodes),
            "subgoal_nodes": len(self.subgoal_nodes),
            }
        #self.logger.info(f"Memory Graph loaded. Node Num Statistics: \n{node_num_stat}")
        return node_num_stat

    # ====== HPQA specific ============
    def build_mem_from_disk_hpqa_ver(self, dir_path: str):
        if not os.path.exists(dir_path):
            raise ValueError("dir_path not exists.")

        # -------------------------
        # 1) Optional episodic load
        # -------------------------
        epis_dir = os.path.join(dir_path, "episodic_memory")
        has_epis = False
        epis_id2node: dict[int, EpisdoicNode] = {}

        if os.path.isdir(epis_dir):
            epis_files = [f for f in os.listdir(epis_dir) if f.endswith(".json")]
            if len(epis_files) > 0:
                has_epis = True
                epis_items: list[dict[str, Any]] = []
                for file in tqdm(epis_files,desc="loading episodic nodes"):
                    with open(os.path.join(epis_dir, file), "r", encoding="utf-8") as f:
                        epis_items.append(json.load(f))
                for epis_item in epis_items:
                    self.episodic_nodes.append(EpisdoicNode(episodic_id=epis_item["episodic_id"]))
                self.logger.info("load all episodic nodes")

                epis_id2node = {node.episodic_id: node for node in self.episodic_nodes}
            else:
                self.logger.info("episodic_memory folder is empty; skip loading episodic nodes")
        else:
            self.logger.info("episodic_memory folder not found; skip loading episodic nodes")

        # -------------------------
        # 2) Load semantic nodes
        # -------------------------
        sem_dir = os.path.join(dir_path, "semantic_memory")
        sem_items: list[dict[str, Any]] = []
        for file in tqdm(os.listdir(sem_dir),desc="loading semantic nodes"):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(sem_dir, file), "r", encoding="utf-8") as f:
                sem_items.append(json.load(f))

        for sem_item in sem_items:
            semantic_id = sem_item["semantic_id"]
            episodic_ids = sem_item.get("episodic_ids", None)
            episodic_id = sem_item.get("episodic_id", None)
            semantic_memory_str = sem_item["semantic_memory"]
            semantic_embedding = sem_item["semantic_embedding"]
            bro_semantic_ids = sem_item.get("bro_semantic_ids", [])
            tags = sem_item.get("tags", [])
            _time = sem_item.get("time", 0)

            if not isinstance(_time, int):
                _time = 0

            semantic_node = SemanticNode(semantic_memory=sem_item,
                                         semantic_memory_embedding=semantic_embedding,
                                         semantic_id=semantic_id,
                                         semantic_memory_str=semantic_memory_str,
                                         time=_time,)
            # Only link episodic if episodic nodes were loaded and the id exists
            if has_epis:
                if episodic_ids is None and episodic_id is not None:
                    episodic_ids = [episodic_id]
                if isinstance(episodic_ids, list):
                    for episodic_id_item in episodic_ids:
                        epis_node = epis_id2node.get(episodic_id_item)
                        if epis_node is not None:
                            semantic_node.episodic_nodes.append(epis_node)
                    print("current semantic node's episodic ids: ", [x.episodic_id for x in semantic_node.episodic_nodes])
                elif episodic_ids is None:
                    pass
                else:
                    raise ValueError("episodic_ids must be a list when provided in semantic_memory_xx.json")
            # store ids temporarily, then replace ids with true nodes
            semantic_node.bro_semantic_nodes = bro_semantic_ids if isinstance(bro_semantic_ids, list) else []
            self.semantic_nodes.append(semantic_node)
            # keep tags on semantic node
            for tag in tags:
                semantic_node.tags.append(tag)
            self.semantic_time = max(self.semantic_time, _time + 1)

        sem_id2node: dict[int, SemanticNode] = {node.semantic_id: node for node in self.semantic_nodes}
        
        # Resolve bro_semantic_nodes ids -> nodes (skip missing ids safely)
        for sem_node in self.semantic_nodes:
            bro_ids = sem_node.bro_semantic_nodes if isinstance(sem_node.bro_semantic_nodes, list) else []
            sem_node.bro_semantic_nodes = [sem_id2node[bid] for bid in bro_ids if bid in sem_id2node]
        self.logger.info("load all semantic nodes")

        # -------------------------
        # 3) Load tag nodes
        # -------------------------
        tag_dir = os.path.join(dir_path, "tag")
        tag_items: list[dict[str, Any]] = []
        for file in tqdm(os.listdir(tag_dir),desc="loading tag nodes"):
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(tag_dir, file)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    tag_items.append(json.load(f))
            except Exception as e:
                print(f"error loading tag file {json_path}: {e}")
                continue
        for tag_item in tag_items:
            tag_id = tag_item["tag_id"]
            tag = tag_item["tag"]
            tag_embedding = tag_item["tag_embedding"]
            semantic_ids = tag_item.get("semantic_ids", [])
            _time = tag_item.get("time", 0)
            importance = tag_item.get("importance", 1)

            tag_node = TagNode(tag=tag,tag_embedding=tag_embedding,tag_id=tag_id,time=_time,)
            tag_node.importance = importance

            for sid in semantic_ids:
                semantic_node = sem_id2node.get(sid)
                if semantic_node is None:
                    continue
                # TagNode -> SemanticNode
                tag_node.semantic_nodes.append(semantic_node)
                # SemanticNode -> TagNode
                semantic_node.tag_nodes.append(tag_node)
                # ensure semantic_node.tags contains this tag
                if hasattr(semantic_node, "tags") and tag_node.tag not in semantic_node.tags:
                    semantic_node.tags.append(tag_node.tag)
            self.tag_nodes.append(tag_node)

        self.logger.info("load all tag nodes")
        
        # -------------------------
        # 4) Load subgoal nodes
        # -------------------------
        subgoal_dir = os.path.join(dir_path, "subgoal")
        subgoal_items: list[dict[str, Any]] = []
        for file in tqdm(os.listdir(subgoal_dir),desc="loading subgoal nodes"):
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(subgoal_dir, file)
            with open(json_path, "r", encoding="utf-8") as f:
                subgoal_items.append(json.load(f))
        for subgoal_item in subgoal_items:
            subgoal_id = subgoal_item["subgoal_id"]
            subgoal_str = subgoal_item["subgoal"]
            subgoal_embedding = subgoal_item["subgoal_embedding"]
            subgoal_time = subgoal_item.get("time", 0)
            if not isinstance(subgoal_time, int):
                subgoal_time = 0
            subgoal_node = SubgoalNode(subgoal_str, subgoal_embedding, subgoal_id, subgoal_time)
            self.subgoal_nodes.append(subgoal_node)
        
        subgoal_id2node: dict[int, SubgoalNode] = {node.subgoal_id: node for node in self.subgoal_nodes}
        self.logger.info(f"load all subgoal nodes")
        
        # -------------------------
        # 5) Load procedural nodes
        # -------------------------
        procedural_dir = os.path.join(dir_path, "procedural_memory")
        procedural_items: list[dict[str, Any]] = []
        for file in tqdm(os.listdir(procedural_dir),desc="loading procedural nodes"):
            if not file.endswith(".json"):
                continue
            json_path = os.path.join(procedural_dir, file)
            with open(json_path, "r", encoding="utf-8") as f:
                procedural_items.append(json.load(f))
        for procedural_item in procedural_items:
            procedural_id = procedural_item["procedural_id"]
            episodic_ids = procedural_item.get("episodic_ids", None)
            episodic_id = procedural_item.get("episodic_id", None)
            procedural_memory_str = procedural_item["procedural_memory"]
            procedural_memory_embedding = procedural_item["procedural_embedding"]
            procedural_node = ProceduralNode(procedural_memory = procedural_item,
                                        procedural_memory_embedding = procedural_memory_embedding,
                                        procedural_id = procedural_id, 
                                        time = procedural_item["time"])
            # Only link episodic if episodic nodes were loaded and the id exists
            if has_epis:
                if episodic_ids is None and episodic_id is not None:
                    episodic_ids = [episodic_id]
                if isinstance(episodic_ids, list):
                    for episodic_id_item in episodic_ids:
                        epis_node = epis_id2node.get(episodic_id_item)
                        if epis_node is not None:
                            procedural_node.episodic_nodes.append(epis_node)
                elif episodic_ids is None:
                    pass
                else:
                    raise ValueError("episodic_ids must be a list when provided in procedural_memory_xx.json")
                    
            subgoal_id = procedural_item.get("subgoal_id", None)
            if subgoal_id is not None:
                subgoal_node = subgoal_id2node.get(subgoal_id,None)
                if subgoal_node is not None:
                    procedural_node.subgoal_nodes.append(subgoal_node)
            
            self.procedural_nodes.append(procedural_node)
        self.logger.info(f"load all procedural nodes")

        # -------------------------
        # 6) Build id2node or str2node dicts
        # -------------------------
        self.tag2node = {x.tag: x for x in self.tag_nodes}
        self.subgoal2node = {x.subgoal: x for x in self.subgoal_nodes}
        self.episodic_id2node = {x.episodic_id: x for x in self.episodic_nodes}
        self.semantic_id2node = {x.semantic_id: x for x in self.semantic_nodes}
        self.procedural_id2node = {x.procedural_id: x for x in self.procedural_nodes}
        self.subgoal_id2node = {x.subgoal_id: x for x in self.subgoal_nodes}
        self.tag_id2node = {x.tag_id: x for x in self.tag_nodes}
        
        node_num_stat = {
            "semantic_nodes": len(self.semantic_nodes),
            "tag_nodes": len(self.tag_nodes),
            "episodic_nodes": len(self.episodic_nodes),
            "procedural_nodes": len(self.procedural_nodes),
            "subgoal_nodes": len(self.subgoal_nodes),
            }
        self.logger.info(f"Memory Graph loaded. Node Num Statistics: \n{node_num_stat}")
        return node_num_stat

    # ====== WebArena specific ============
    def build_mem_from_disk_webarena_ver(self, dir_path: str, refresh_embeddings: bool=False):
        if not os.path.exists(dir_path):
            raise ValueError("dir_path not exists.")

        def _list_json(path: str, prefix: str) -> List[str]:
            if not os.path.isdir(path):
                return []
            files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(".json")]
            files.sort(key=lambda f: int(re.findall(r"(\d+)\.json", f)[0]))
            return [os.path.join(path, f) for f in files]

        # -------------------------
        # 1) Load episodic nodes
        # -------------------------
        epis_dir = os.path.join(dir_path, "episodic_memory")
        epis_files = _list_json(epis_dir, "episodic_memory_")
        for filepath in tqdm(epis_files, desc="loading episodic nodes"):
            match = re.search(r"episodic_memory_(\d+)\.json", filepath)
            if not match:
                continue
            episodic_id = int(match.group(1))
            while len(self.episodic_nodes) <= episodic_id:
                self.episodic_nodes.append(None)
            self.episodic_nodes[episodic_id] = EpisdoicNode(episodic_id=episodic_id)
        self.logger.info(f"load all episodic nodes: {len([n for n in self.episodic_nodes if n is not None])}")

        # -------------------------
        # 2) Load subgoal nodes
        # -------------------------
        subgoal_dir = os.path.join(dir_path, "subgoal")
        subgoal_files = _list_json(subgoal_dir, "subgoal_")
        subgoal_strings: Dict[int, str] = {}
        for filepath in tqdm(subgoal_files, desc="loading subgoal nodes"):
            match = re.search(r"subgoal_(\d+)\.json", filepath)
            if not match:
                continue
            subgoal_id = int(match.group(1))
            with open(filepath, "r", encoding="utf-8") as f:
                subgoal_json = json.load(f)
            subgoal_str = subgoal_json.get("subgoal", "")
            subgoal_embedding = None if refresh_embeddings else subgoal_json.get("embedding")
            if subgoal_embedding is None and subgoal_str:
                subgoal_embedding = get_embedding(subgoal_str)
            if subgoal_embedding is None:
                continue
            subgoal_time = subgoal_json.get("time", 0)
            while len(self.subgoal_nodes) <= subgoal_id:
                self.subgoal_nodes.append(None)
            subgoal_node = SubgoalNode(subgoal_str, subgoal_embedding, subgoal_id, subgoal_time)
            self.subgoal_nodes[subgoal_id] = subgoal_node
            # cache for exact match
            subgoal_node.cached_subgoal = subgoal_str
            subgoal_strings[subgoal_id] = subgoal_str
            if refresh_embeddings:
                save_subgoal_webarena_ver(
                    subgoal=subgoal_str,
                    subgoal_id=subgoal_id,
                    procedural_ids=subgoal_json.get("procedural_ids", []),
                    subgoal_embedding=subgoal_embedding,
                )
        self.logger.info(f"load all subgoal nodes: {len([n for n in self.subgoal_nodes if n is not None])}")

        # -------------------------
        # 3) Load semantic nodes and tags
        # -------------------------
        tag_map: Dict[str, TagNode] = {}
        pending_bro_links: List[Tuple[int, List[int]]] = []
        sem_dir = os.path.join(dir_path, "semantic_memory")
        sem_files = _list_json(sem_dir, "semantic_memory_")
        for filepath in tqdm(sem_files, desc="loading semantic nodes"):
            match = re.search(r"semantic_memory_(\d+)\.json", filepath)
            if not match:
                continue
            semantic_id = int(match.group(1))
            with open(filepath, "r", encoding="utf-8") as f:
                sem_item = json.load(f)
            semantic_memory_str = sem_item.get("semantic_memory", "")
            tags = sem_item.get("tags", [])
            episodic_ids = sem_item.get("episodic_ids", None)
            episodic_id = sem_item.get("episodic_id", None)
            if episodic_ids is None and episodic_id is not None:
                episodic_ids = [episodic_id]
            if episodic_ids is None:
                episodic_ids = []
            bro_semantic_ids = sem_item.get("bro_semantic_ids", [])
            _time = sem_item.get("time", semantic_id)
            if not isinstance(_time, int):
                _time = semantic_id
            semantic_embedding = None if refresh_embeddings else sem_item.get("embedding")
            if semantic_embedding is None and semantic_memory_str:
                semantic_embedding = get_embedding(semantic_memory_str)
            if semantic_embedding is None:
                continue
            tag_embeddings = {} if refresh_embeddings else sem_item.get("tag_embeddings", {})

            semantic_node = SemanticNode(
                semantic_memory=sem_item,
                semantic_memory_embedding=semantic_embedding,
                semantic_id=semantic_id,
                semantic_memory_str=semantic_memory_str,
                time=_time,
            )
            while len(self.semantic_nodes) <= semantic_id:
                self.semantic_nodes.append(None)
            self.semantic_nodes[semantic_id] = semantic_node
            self.semantic_time = max(self.semantic_time, _time + 1)

            for tag in tags:
                if tag in tag_map:
                    tag_node = tag_map[tag]
                else:
                    tag_embedding = tag_embeddings.get(tag)
                    if tag_embedding is None:
                        tag_embedding = get_embedding(tag)
                    if tag_embedding is None:
                        continue
                    tag_node = TagNode(tag=tag, tag_embedding=tag_embedding, tag_id=len(self.tag_nodes), time=_time)
                    tag_map[tag] = tag_node
                    self.tag_nodes.append(tag_node)
                tag_node.semantic_nodes.append(semantic_node)
                tag_node.importance = max(tag_node.importance, len(tag_node.semantic_nodes))
                semantic_node.tag_nodes.append(tag_node)
                semantic_node.tags.append(tag_node.tag)

            for epi_id in episodic_ids:
                if epi_id < len(self.episodic_nodes) and self.episodic_nodes[epi_id] is not None:
                    semantic_node.episodic_nodes.append(self.episodic_nodes[epi_id])
            if bro_semantic_ids:
                pending_bro_links.append((semantic_id, bro_semantic_ids))

            if refresh_embeddings:
                save_semantic_webarena_ver(
                    semantic_memory=semantic_memory_str,
                    tags=tags,
                    semantic_id=semantic_id,
                    time=_time,
                    episodic_ids=episodic_ids,
                    bro_semantic_ids=bro_semantic_ids,
                    semantic_embedding=semantic_embedding,
                    tag_embeddings={tag_node.tag: tag_node.embedding for tag_node in semantic_node.tag_nodes},
                )
        self.logger.info(f"load all semantic nodes: {len([n for n in self.semantic_nodes if n is not None])}")
        self.logger.info(f"load all tag nodes: {len(self.tag_nodes)}")

        # Resolve bro_semantic_nodes ids -> nodes (skip missing ids safely)
        for sem_id, bro_ids in pending_bro_links:
            if sem_id >= len(self.semantic_nodes):
                continue
            semantic_node = self.semantic_nodes[sem_id]
            if semantic_node is None:
                continue
            for bro_id in bro_ids:
                if bro_id < len(self.semantic_nodes) and self.semantic_nodes[bro_id] is not None:
                    bro_node = self.semantic_nodes[bro_id]
                    if bro_node not in semantic_node.bro_semantic_nodes:
                        semantic_node.bro_semantic_nodes.append(bro_node)

        # -------------------------
        # 4) Load procedural nodes
        # -------------------------
        procedural_dir = os.path.join(dir_path, "procedural_memory")
        procedural_files = _list_json(procedural_dir, "procedural_memory_")
        for filepath in tqdm(procedural_files, desc="loading procedural nodes"):
            match = re.search(r"procedural_memory_(\d+)\.json", filepath)
            if not match:
                continue
            procedural_id = int(match.group(1))
            with open(filepath, "r", encoding="utf-8") as f:
                procedural_item = json.load(f)
            subgoal_str = procedural_item.get("subgoal", "")
            episodic_ids = procedural_item.get("episodic_ids", None)
            episodic_id = procedural_item.get("episodic_id", None)
            if episodic_ids is None and episodic_id is not None:
                episodic_ids = [episodic_id]
            if episodic_ids is None:
                episodic_ids = []
            subgoal_id = procedural_item.get("subgoal_id", None)
            subgoal_embedding = None if refresh_embeddings else procedural_item.get("subgoal_embedding")
            if subgoal_embedding is None and subgoal_str:
                subgoal_embedding = get_embedding(subgoal_str)
            if subgoal_embedding is None:
                continue

            subgoal_node = None
            if subgoal_id is not None and subgoal_id < len(self.subgoal_nodes):
                subgoal_node = self.subgoal_nodes[subgoal_id]
            if subgoal_node is None:
                for sid, existing_str in subgoal_strings.items():
                    if existing_str == subgoal_str and sid < len(self.subgoal_nodes):
                        subgoal_node = self.subgoal_nodes[sid]
                        break
            if subgoal_node is None:
                subgoal_node = self.retrieve_subgoal_nodes(
                    subgoal=subgoal_str,
                    subgoal_embedding=subgoal_embedding,
                    value_func=self.subgoal_equal,
                )
            if subgoal_node is None:
                subgoal_node = SubgoalNode(subgoal=subgoal_str, subgoal_embedding=subgoal_embedding, subgoal_id=len(self.subgoal_nodes), time=self.procedural_time)
                self.subgoal_nodes.append(subgoal_node)
                subgoal_strings[subgoal_node.subgoal_id] = subgoal_str

            procedural_node = ProceduralNode(
                procedural_memory=procedural_item,
                procedural_memory_embedding=subgoal_embedding,
                procedural_id=procedural_id,
                time=self.procedural_time,
            )
            subgoal_node.procedural_nodes.append(procedural_node)
            subgoal_node.importance = max(subgoal_node.importance, len(subgoal_node.procedural_nodes))

            while len(self.procedural_nodes) <= procedural_id:
                self.procedural_nodes.append(None)
            self.procedural_nodes[procedural_id] = procedural_node
            for epi_id in episodic_ids:
                if epi_id < len(self.episodic_nodes) and self.episodic_nodes[epi_id] is not None:
                    procedural_node.episodic_nodes.append(self.episodic_nodes[epi_id])
            self.procedural_time = max(self.procedural_time, procedural_id + 1)

            if refresh_embeddings:
                save_procedural_webarena_ver(
                    procedural_memory=procedural_item,
                    procedural_id=procedural_id,
                    subgoal_id=subgoal_id,
                    episodic_ids=episodic_ids,
                    subgoal_embedding=subgoal_embedding,
                )

        # -------------------------
        # 5) Build id2node or str2node dicts
        # -------------------------
        self.tag2node = {x.tag: x for x in self.tag_nodes}
        self.subgoal2node = {x.subgoal: x for x in self.subgoal_nodes if x is not None}
        self.episodic_id2node = {x.episodic_id: x for x in self.episodic_nodes if x is not None}
        self.semantic_id2node = {x.semantic_id: x for x in self.semantic_nodes if x is not None}
        self.procedural_id2node = {x.procedural_id: x for x in self.procedural_nodes if x is not None}
        self.subgoal_id2node = {x.subgoal_id: x for x in self.subgoal_nodes if x is not None}
        self.tag_id2node = {x.tag_id: x for x in self.tag_nodes}

        node_num_stat = {
            "semantic_nodes": len([n for n in self.semantic_nodes if n is not None]),
            "tag_nodes": len(self.tag_nodes),
            "episodic_nodes": len([n for n in self.episodic_nodes if n is not None]),
            "procedural_nodes": len([n for n in self.procedural_nodes if n is not None]),
            "subgoal_nodes": len([n for n in self.subgoal_nodes if n is not None]),
        }
        self.logger.info(f"Memory Graph loaded. Node Num Statistics: \n{node_num_stat}")
        return node_num_stat
        
    # ====== Default ============
    def insert(self, memory: Memory):
        episodic_nodes = []
        for i, trajectory in enumerate(memory.memory["episodic"]):
            episodic_nodes.append([])
            for episodic_memory in trajectory:
                episodic_node = EpisdoicNode(len(self.episodic_nodes))
                self.episodic_nodes.append(episodic_node)
                episodic_nodes[i].append(episodic_node)
                save_episodic_longmem_ver(episodic_memory, episodic_node.episodic_id)

        for semantic_memory, semantic_memory_embedding in zip(
            memory.memory["semantic"], memory.memory_embedding["semantic"]
        ):
            semantic_memory_str = semantic_memory["semantic_memory"]
            semantic_node = SemanticNode(
                semantic_memory=semantic_memory,
                semantic_memory_embedding=semantic_memory_embedding["semantic_memory"],
                semantic_id=len(self.semantic_nodes),
                semantic_memory_str=semantic_memory_str,
                time=self.semantic_time,
            )
            self.semantic_nodes.append(semantic_node)

            for tag, tag_embedding in zip(
                semantic_memory["tags"], semantic_memory_embedding["tags"]
            ):
                tag_nodes = self.retrieve_tag_nodes(
                    tag=tag,
                    tag_embedding=tag_embedding,
                    value_func=self.tag_equal,
                    make_tag_nodes=True,
                )
                tag_node = tag_nodes[0]
                tag_node.semantic_nodes.append(semantic_node)
                tag_node.semantic_nodes = list(set(tag_node.semantic_nodes))
                semantic_node.tag_nodes.append(tag_node)
                semantic_node.tags.append(tag_node.tag)

            semantic_node.tags = list(set(semantic_node.tags))
            semantic_node.episodic_nodes.append(
                episodic_nodes[semantic_memory["trajectory_num"]][semantic_memory["turn_num"]]
            )
            self.semantic_time += 1
            save_semantic_longmem_ver(
                semantic_memory_str,
                semantic_node.tags,
                semantic_node.semantic_id,
                semantic_memory["time"],
            )

        for procedural_memory, procedral_memory_embedding in zip(
            memory.memory["procedural"], memory.memory_embedding["procedural"]
        ):
            subgoal_node = self.retrieve_subgoal_nodes(
                subgoal=procedural_memory["subgoal"],
                subgoal_embedding=procedral_memory_embedding["subgoal"],
                value_func=self.subgoal_equal,
            )
            subgoal_str = procedural_memory["subgoal"]
            procedural_embedding = get_embedding(procedural_memory["procedural_memory"])

            if subgoal_node is not None:
                procedural_node = ProceduralNode(
                    procedural_memory=procedural_memory,
                    procedural_memory_embedding=procedural_embedding,
                    procedural_id=len(self.procedural_nodes),
                    time=self.procedural_time,
                )
                subgoal_str = get_new_subgoal(subgoal_node.get_subgoal(), subgoal_str)
                subgoal_node.embedding = get_embedding(subgoal_str)
                subgoal_node.time = self.procedural_time
                subgoal_node.procedural_nodes.append(procedural_node)
            else:
                subgoal_node = SubgoalNode(
                    procedural_memory["subgoal"],
                    procedral_memory_embedding["subgoal"],
                    len(self.subgoal_nodes),
                    self.procedural_time,
                )
                procedural_node = ProceduralNode(
                    procedural_memory=procedural_memory,
                    procedural_memory_embedding=procedural_embedding,
                    procedural_id=len(self.procedural_nodes),
                    time=self.procedural_time,
                )
                subgoal_node.activation([procedural_node])
                self.subgoal_nodes.append(subgoal_node)

            procedural_node.episodic_nodes += episodic_nodes[
                procedural_memory["trajectory_num"]
            ]
            self.procedural_nodes.append(procedural_node)
            self.procedural_time += 1
            save_subgoal_longmem_ver(subgoal_str, subgoal_node.subgoal_id)
            save_procedural_longmem_ver(procedural_memory, procedural_node.procedural_id)

        self.tag2node = {x.tag: x for x in self.tag_nodes}
        self.subgoal2node = {x.subgoal: x for x in self.subgoal_nodes}
        self.episodic_id2node = {x.episodic_id: x for x in self.episodic_nodes}
        self.semantic_id2node = {x.semantic_id: x for x in self.semantic_nodes}
        self.procedural_id2node = {x.procedural_id: x for x in self.procedural_nodes}
        self.subgoal_id2node = {x.subgoal_id: x for x in self.subgoal_nodes}
        self.tag_id2node = {x.tag_id: x for x in self.tag_nodes}

    # ====== HPQA specific ============
    def insert_hpqa_ver(self, memory: Memory):
        # === save episodic nodes ===
        episodic_nodes: List[EpisdoicNode] = []
        for epis_mem_item in memory.memory["episodic"]:
            epis_id = len(self.episodic_nodes)
            if isinstance(epis_mem_item, dict):
                episodic_memory_str = epis_mem_item.get("observation", epis_mem_item.get("episodic_memory", ""))
            else:
                episodic_memory_str = str(epis_mem_item)
            epis_node = EpisdoicNode(episodic_id=epis_id)
            self.episodic_nodes.append(epis_node)
            episodic_nodes.append(epis_node)
            save_episodic_hpqa_ver(episodic_memory_str=episodic_memory_str, episodic_id=epis_id)
        self.logger.info(
            f"current saved episodic node ids: {[node.episodic_id for node in episodic_nodes]}"
        )
        episodic_ids = [node.episodic_id for node in episodic_nodes]
        
        # === save semantic nodes ===
        curr_sem_nodes:List[SemanticNode] = []
        curr_sem_strs:List[str] = []
        
        for sem_mem_item, sem_mem_emb_item in zip(memory.memory["semantic"], memory.memory_embedding["semantic"]):
            semantic_memory_str = sem_mem_item["semantic_memory"]
            if semantic_memory_str is None or semantic_memory_str == "":
                self.logger.info(f"skip saving semantic node because semantic_memory_str is None or empty")
                continue
            
            sem_node = SemanticNode(semantic_memory = sem_mem_item, 
                                    semantic_memory_embedding = sem_mem_emb_item["semantic_memory"], 
                                    semantic_id = len(self.semantic_nodes), 
                                    time = self.semantic_time,
                                         )
            sem_node.episodic_nodes = list(episodic_nodes)
            curr_sem_nodes.append(sem_node)
            curr_sem_strs.append(semantic_memory_str)
            
            self.semantic_nodes.append(sem_node)
            # self.semantic_time += 1
            
            # === save tag nodes ===
            for tag, tag_embedding in zip(sem_mem_item["tags"], sem_mem_emb_item["tags"]):
                tag_is_new = None
                tag_node = self.tag2node.get(tag)
                if tag_node is None:
                    tag_node = TagNode(tag=tag, tag_embedding=tag_embedding, tag_id=len(self.tag_nodes), time=self.semantic_time)
                    self.tag_nodes.append(tag_node)
                    self.tag2node[tag] = tag_node
                    tag_is_new = True
                else:
                    tag_is_new = False
                
                sem_node.tag_nodes.append(tag_node)
                sem_node.tags.append(tag_node.tag)
                sem_node.tags = list(set(sem_node.tags))
                
                tag_node.semantic_nodes.append(sem_node)
                tag_node.semantic_nodes=list(set(tag_node.semantic_nodes))
                self.logger.info(f"current tag_node-{tag_node.tag_id} {tag_node.tag}, its sem nodes: {[x.semantic_id for x in tag_node.semantic_nodes]}")
                
                if tag_is_new:
                    save_tag_hpqa_ver(tag=tag_node.tag, 
                            tag_id=tag_node.tag_id, 
                            semantic_ids=[x.semantic_id for x in tag_node.semantic_nodes], 
                            time=tag_node.time, 
                            tag_embedding=tag_node.embedding, 
                            importance=tag_node.importance,
                        )
                else:
                    update_tag_hpqa_ver(tag=tag_node.tag, 
                            tag_id=tag_node.tag_id, 
                            semantic_ids=[x.semantic_id for x in tag_node.semantic_nodes], 
                            time=tag_node.time, 
                            tag_embedding=tag_node.embedding, 
                            importance=tag_node.importance,
                        )
                self.logger.info(f"current saved tag node id: {len(self.tag_nodes)}")
        
        # === save bro semantic nodes ===
        for sem_node, sem_str in zip(curr_sem_nodes,curr_sem_strs):
            bro_sem_ids = [node.semantic_id for node in curr_sem_nodes if node.semantic_id != sem_node.semantic_id]
            save_semantic_hpqa_ver(semantic_memory_str = sem_str, 
                          semantic_embedding = sem_node.embedding,
                          semantic_id = sem_node.semantic_id, 
                          episodic_ids = episodic_ids,
                          episodic_id = episodic_ids[0] if len(episodic_ids) == 1 else None,
                          bro_semantic_ids = bro_sem_ids,
                          tags = sem_node.tags, 
                          tag_ids = [tag_node.tag_id for tag_node in sem_node.tag_nodes],
                          time = sem_node.time,
                          )
            self.logger.info(f"current saved sem node id: {sem_node.semantic_id}")

        # === save procedural nodes ===
        for proce_mem_item, proce_mem_emb_item in zip(memory.memory["procedural"], memory.memory_embedding["procedural"]):
            procedural_memory_str = proce_mem_item["procedural_memory"]
            if procedural_memory_str is None or procedural_memory_str == "":
                self.logger.info(f"skip saving procedural node because procedural_memory_str is None or empty")
                continue
            proced_node = ProceduralNode(procedural_memory = proce_mem_item,
                                        procedural_memory_embedding = proce_mem_emb_item["procedural_memory"],
                                        procedural_id = len(self.procedural_nodes), 
                                        time = self.procedural_time)
            proced_node.episodic_nodes += episodic_nodes
            
            # === save subgoal nodes ===
            # only one subgoal for each procedural node
            subgoal_str = proce_mem_item["subgoal"]
            subgoal_embedding = proce_mem_emb_item["subgoal"]
            

            subgoal_node = self.subgoal2node.get(subgoal_str)
            if subgoal_node is None:
                subgoal_node = SubgoalNode(subgoal_str, subgoal_embedding, len(self.subgoal_nodes), self.procedural_time)
                self.subgoal_nodes.append(subgoal_node)
                self.subgoal2node[subgoal_str] = subgoal_node
                subgoal_node.activation([proced_node])
            else:
                subgoal_str = get_new_subgoal(subgoal_node.get_subgoal(),subgoal_str)
                subgoal_node.embedding = subgoal_embedding
                subgoal_node.time = self.procedural_time
                subgoal_node.procedural_nodes.append(proced_node)
                                
            save_subgoal_hpqa_ver(subgoal = subgoal_str,
                         subgoal_id = subgoal_node.subgoal_id,
                         procedural_id = proced_node.procedural_id,
                         time = proced_node.time,
                         subgoal_embedding = subgoal_node.embedding,)
            self.logger.info(f"current saved subgoal node id: {subgoal_node.subgoal_id}")
            
            proced_node.subgoals.append(subgoal_str)
            proced_node.subgoals = list(set(proced_node.subgoals))
            proced_node.subgoal_nodes.append(subgoal_node)
            proced_node.subgoal_nodes = list(set(proced_node.subgoal_nodes))
            
            self.procedural_nodes.append(proced_node)
            save_procedural_hpqa_ver(procedural_memory_str = procedural_memory_str, 
                            procedural_embedding = proced_node.embedding,
                            procedural_id = proced_node.procedural_id, 
                            subgoal = subgoal_str,
                            subgoal_id = subgoal_node.subgoal_id,
                            episodic_ids = episodic_ids,
                            episodic_id = episodic_ids[0] if len(episodic_ids) == 1 else None,
                            time = proced_node.time,
                            _return = proced_node.Return,
                            )
            self.logger.info(f"current saved procedural node id: {proced_node.procedural_id}")
    
    
    def retrieve_tag_nodes(self, tag: str, tag_embedding=None, value_func: ValueBase=None, make_tag_nodes: bool = False, write: bool=False):
        if tag_embedding is None:
            tag_embedding = get_embedding(tag)
        embedding = tag_embedding
        values = []
        for tag_node in self.tag_nodes:
            if tag_node.embedding is None:
                tag_node.embedding = get_embedding(tag_node.tag)
            Relevance = get_similarity(embedding, tag_node.embedding)
            Recency = self.semantic_time - tag_node.time
            Importance = tag_node.importance
            Return = 0
            Credibility = 0
            value = value_func.evaluate(
                Relevance = Relevance,
                Recency = Recency,
                Importance = Importance,
                Return = Return,
                Credibility = Credibility
            )
            values.append((value, tag_node.tag_id))
        values.sort(reverse=True, key=lambda x: x[0])
        values = values[:value_func.k]
        re = []
        tag_id2node={x.tag_id:x for x in self.tag_nodes}
        for value, index in values:
            if(value < value_func.value_threshold):
                break
            re.append(tag_id2node[index])
        if write == True:
            DIR_PATH = os.environ.get("DIR_PATH", None)
            with open(DIR_PATH+"/retrieve.jsonl","a",) as output:
                output.write(json.dumps({"retrieve_tag": tag, "retrieved_tag": [tag_node.tag for tag_node in re], "retrieved_tag_semantic_nodes": [[semantic_node.semantic_id for semantic_node in tag_node.semantic_nodes] for tag_node in re]})+"\n")
        if len(re) == 0 and make_tag_nodes == True:
            self.logger.info("[new] creating new tag node...")
            tag_node = TagNode(tag=tag, tag_embedding=tag_embedding, tag_id=len(self.tag_nodes), time=self.semantic_time)
            self.tag_nodes.append(tag_node)
            re.append(tag_node)
        return re


    def retrieve_semantic_nodes(
        self,
        semantic_memory: Dict[str, Any],
        semantic_memory_embedding: Optional[Dict[str, Any]] = None,
        value_func_tag: Optional[ValueBase] = None,
        value_func: Optional[ValueBase] = None,
        write: bool = False,):
        if value_func_tag is None or value_func is None:
            raise ValueError("value_func_tag and value_func must not be None.")

        # 0. Prepare query embeddings (semantic text + tag embeddings)
        if semantic_memory_embedding is None:
            semantic_memory_embedding = {
                "semantic_memory": get_embedding(semantic_memory["semantic_memory"]),
                "tags": [get_embedding(tag) for tag in semantic_memory["tags"]],
            }

        query_text = semantic_memory["semantic_memory"]
        query_tags: List[str] = semantic_memory.get("tags", [])
        query_embedding = semantic_memory_embedding["semantic_memory"]
        semantic_id2node: Dict[int, SemanticNode] = {
            node.semantic_id: node for node in self.semantic_nodes
        }
        tag_id2node = {
            node.tag_id:node for node in self.tag_nodes
        }
        ids = list(semantic_id2node.keys())
        max_w = len(str(max(ids)))  
        
        # 1. Retrieve top-k similar SemanticNodes based on query text embedding
        sem_node_topk = 5
        sem_node_sim_list: List[tuple[float, int]] = []

        for node in self.semantic_nodes:
            if not node.is_active:
                continue
            # Ensure each semantic node has an embedding
            if getattr(node, "embedding", None) is None:
                node.embedding = get_embedding(node.get_semantic_memory())

            sim = get_similarity(query_embedding, node.embedding)
            sem_node_sim_list.append((sim, node.semantic_id))

        # Sort by similarity descending and keep top-k
        sem_node_sim_list.sort(reverse=True, key=lambda x: x[0])
        top_sim_ids = [sem_id for sim, sem_id in sem_node_sim_list[:sem_node_topk]]

        # Map ids back to SemanticNode objects
        sem_node_from_q_text = [semantic_id2node[sem_id] for sem_id in top_sim_ids]

        self.logger.info(f"\nretrieved sem node from query text emb directly: ")
        # print(f"\nretrieved sem node from query text emb directly: ")
        for n in sem_node_from_q_text:
            self.logger.info(f"{n.semantic_id:<{max_w}}: {n.get_semantic_memory()}")
            # print(f"{n.semantic_id:<{max_w}}: {n.get_semantic_memory()}")
            
        # 2. Retrieve TagNodes for each query tag
        tag_nodes: List[TagNode] = []
        for tag, tag_emb in zip(query_tags, semantic_memory_embedding["tags"]):
            new_tag_nodes = self.retrieve_tag_nodes(
                tag=tag,
                tag_embedding=tag_emb,
                value_func=value_func_tag,
                write=write,
            )
            tag_nodes.extend(new_tag_nodes)

        self.logger.info(f"\nretrieved tags from query tags: \n{[x.tag for x in tag_nodes]}\n")
        # print(f"\nretrieved tags from query tags: \n{[x.tag for x in tag_nodes]}")
        
        # 3. Aggregate votes for SemanticNodes via TagNodes
        #    vote[semantic_id] = {"cnt": count_of_supporting_tags,
        #                         "importance": accumulated_tag_importance}
        tag_vote: Dict[int, Dict[str, float]] = {}
        for tag_node in tag_nodes:
            if tag_node is None:
                continue
            for semantic_node in tag_node.semantic_nodes:
                sem_id = semantic_node.semantic_id

                if sem_id not in tag_vote:
                    tag_vote[sem_id] = {"cnt": 0, "importance": 0.0}

                tag_vote[sem_id]["cnt"] += 1

                # Increase importance; boost more if tag exactly matches a query tag
                if tag_node.tag in query_tags:
                    tag_vote[sem_id]["importance"] += 5.0 * tag_node.importance
                else:
                    tag_vote[sem_id]["importance"] += float(tag_node.importance)
                    
        for sem_node in sem_node_from_q_text:
            sem_id = sem_node.semantic_id
            if sem_id not in tag_vote:
                tag_vote[sem_id] = {"cnt": 0, "importance": 0.0}
            tag_vote[sem_id]["cnt"] += 1
            tag_vote[sem_id]["importance"] += 2.0
        
        # Optional: write voting statistics to disk
        if write:
            DIR_PATH = os.environ.get("DIR_PATH", None)
            if DIR_PATH is not None:
                with open(os.path.join(DIR_PATH, "retrieve.jsonl"), "a", encoding="utf-8") as output:
                    output.write(json.dumps({"vote": tag_vote}) + "\n")
                    
        # 4. Collect candidate SemanticNodes that received any votes
        candidate_nodes: List[SemanticNode] = []
        for semantic_id in tag_vote.keys():
            node = semantic_id2node.get(semantic_id)
            if node is not None:
                candidate_nodes.append(node)

        candidate_nodes = list(set(candidate_nodes + sem_node_from_q_text))

        # 5. Score each candidate SemanticNode
        values: List[tuple[float, int]] = []
        for semantic_node in candidate_nodes:
            # Ensure the node has an embedding
            if getattr(semantic_node, "embedding", None) is None:
                semantic_node.embedding = get_embedding(semantic_node.get_semantic_memory())

            relevance = get_similarity(query_embedding, semantic_node.embedding)

            # Importance is normalized by the number of tags on the semantic node
            num_tags = max(1, len(semantic_node.tags))  # avoid division by zero
            importance_score = tag_vote[semantic_node.semantic_id]["importance"] / num_tags
            # importance_score = 1
            
            # Recency based on semantic_time and the node's time
            if isinstance(semantic_node.time, int):
                recency = self.semantic_time - semantic_node.time
            else:
                recency = 0

            ret = 0
            credibility = semantic_node.Credibility

            value = value_func.evaluate(
                Relevance=relevance,
                Recency=recency,
                Importance=importance_score,
                Return=ret,
                Credibility=credibility,
            )
            values.append((value, semantic_node.semantic_id))

            # Optional: write per-node scoring details
            if write:
                DIR_PATH = os.environ.get("DIR_PATH", None)
                if DIR_PATH is not None:
                    with open(os.path.join(DIR_PATH, "retrieve.jsonl"), "a", encoding="utf-8") as output:
                        output.write(json.dumps(
                                {
                                    "value": value,
                                    "id": semantic_node.semantic_id,
                                    "semantic_memory": semantic_node.get_semantic_memory(),
                                }
                            )+ "\n"
                        )

        # 6. Sort by score, apply top-k
        values.sort(reverse=True, key=lambda x: x[0])
        values = values[: value_func.k]
        self.logger.info(f"\nremained top-{value_func.k} semantic nodes: ")
        # print(f"\nremained top-{value_func.k} semantic nodes: ")
        for value in values:
            self.logger.info(f"v:{value[0]:.4f}, node {value[1]:<{max_w}}: {semantic_id2node[value[1]].get_semantic_memory()}")
            # print(f"v:{value[0]:.4f}, node {value[1]:<{max_w}}: {semantic_id2node[value[1]].get_semantic_memory()}")
            
        # 7. Apply value threshold and return final SemanticNodes
        result_nodes: List["SemanticNode"] = []
        for value, sem_id in values:
            if value < value_func.value_threshold:
                break
            result_nodes.append(semantic_id2node[sem_id])

        return result_nodes


    # ================== LongMemEval specific ==================
    def retrieve_semantic_nodes_wo_tag(self, semantic_memory, semantic_memory_embedding=None,  value_func: ValueBase=None, write: bool=False):
        if semantic_memory_embedding == None:
            semantic_memory_embedding = {}
            semantic_memory_embedding["semantic_memory"] = get_embedding(semantic_memory["semantic_memory"])
        
        
        embedding = semantic_memory_embedding["semantic_memory"]
        values = []
        for semantic_node in self.semantic_nodes:
            Relevance = get_similarity(embedding, semantic_node.embedding)
            Importance = 0
            Recency = self.semantic_time - semantic_node.time
            Return = 0
            Credibility = semantic_node.Credibility
            value = value_func.evaluate(
                Relevance = Relevance,
                Recency = Recency,
                Importance = Importance,
                Return = Return,
                Credibility = Credibility
            )
            values.append((value, semantic_node.semantic_id))
        values.sort(reverse=True, key=lambda x: x[0])
        values = values[:value_func.k]
        re = []
        for value, index in values:
            if(value < value_func.value_threshold):
                break
            re.append(self.semantic_nodes[index])

        return re


    # ================== LongMemEval specific ==================
    def retrieve_episodic_nodes(self, observation: str=None):
        embedding = get_embedding(observation)
        DIR_PATH = os.environ.get("DIR_PATH", None)
        WRITE = os.environ.get("WRITE", None)
        semantic_nodes = self.retrieve_semantic_nodes_wo_tag(
            semantic_memory={
                "semantic_memory": observation,
            },
            value_func = self.semantic_relevant4episodic,
            write=True
        )
        if not WRITE == None and WRITE == "TRUE":
            for semantic_node in semantic_nodes:
                with open(DIR_PATH + "/retrieve_1.jsonl","a",) as output:
                    output.write(json.dumps({
                        "memory_type": "semantic",
                        "content": f"{semantic_node.get_semantic_memory()}",
                        "id": f"{semantic_node.semantic_id}",
                        "similarity": get_similarity(embedding, semantic_node.embedding)
                    })+"\n")
        episodic_memory_str = ""
        semantic_nodes = semantic_nodes[:30]
        '''merge episodic from same session'''
        vote_session = {}
        for semantic_node in semantic_nodes:
            if semantic_node.session_id in vote_session:
                vote_session[semantic_node.session_id] += 1
            else:
                vote_session[semantic_node.session_id] = 1
        cnt = 0
        for key, value in vote_session.items():
            if value >= 3:
                episodic_memory_str += f"Relevant Memory {cnt}:\n{self.get_session_memory(key)}"
                cnt += 1
        for i, semantic_node in enumerate(semantic_nodes):
            if vote_session[semantic_node.session_id] < 3:
                episodic_memory_str += f"Relevant Memory {cnt}:\n{semantic_node.get_semantic_memory()}\n"
                cnt += 1
        return episodic_memory_str
    
    
    # ================== LongMemEval specific ==================
    def get_session_memory(self, session_id):
        episodic_memory_str = ""
        if session_id not in self.session_ids:
            return "There is no relevant memory"
        episodic_memory_str += f"{self.session_ids[session_id][0].get_date()}\n"
        for episodic_node in self.session_ids[session_id]:
            episodic_memory_str += f"{episodic_node.get_episodic_memory(date = False)}\n"
        return episodic_memory_str

    

    def retrieve_subgoal_nodes(self, subgoal: str, subgoal_embedding=None, value_func: ValueBase=None):
        if subgoal_embedding is None:
            subgoal_embedding = get_embedding(subgoal)
        embedding = subgoal_embedding
        best_value = -1
        best_subgoal_node = None
        for subgoal_node in self.subgoal_nodes:
            Relevance = get_similarity(embedding, subgoal_node.embedding)
            Importance = subgoal_node.importance
            Recency = self.procedural_time - subgoal_node.time
            Return = 0
            Credibility = 0
            value = value_func.evaluate(
                Relevance = Relevance,
                Recency = Recency,
                Importance = Importance,
                Return = Return,
                Credibility = Credibility
            )
            if value > best_value:
                best_subgoal_node = subgoal_node
                best_value = value

        if best_value < value_func.value_threshold:
            best_subgoal_node = None
        else:
            best_subgoal_node.importance += 1

        return best_subgoal_node


    def retrieve_procedural_nodes(self, subgoal: str, value_func_subgoal: ValueBase, value_func: ValueBase):
        
        embedding = get_embedding(subgoal)
        subgoal_node = self.retrieve_subgoal_nodes(
            subgoal = subgoal,
            value_func = value_func_subgoal
        )
        if subgoal_node == None:
            return []
        else:
            values = []
            for procedural_node in subgoal_node.procedural_nodes:
                Relevance = get_similarity(embedding, procedural_node.embedding)
                Importance = 0
                Return = procedural_node.Return
                Credibility = 0
                Recency = self.procedural_time - procedural_node.time
                value = value_func.evaluate(
                    Importance = Importance,
                    Relevance = Relevance,
                    Return = Return,
                    Recency = Recency,
                    Credibility = Credibility
                )
                values.append((value, procedural_node.procedural_id))
            values.sort(reverse=True, key=lambda x: x[0])
            values = values[:value_func.k]
            re = []
            procedural_id2node = {x.procedural_id:x for x in self.procedural_nodes}
            
            for value, index in values:
                if(value < value_func.value_threshold):
                    break
                re.append(procedural_id2node[index])
            return re


    def retrieve_memory(self, goal: str=None, subgoal: str=None, state: str=None, observation: str=None, prompt_template: PromptBase=None, time: str="", task_type: str="", mode: str=None):
        
        self.logger.info(f"query_text: {observation}")
        # print(f"query_text: {observation}")
        
        next_subgoal, query_tags = get_plan(
            goal = goal, 
            subgoal = subgoal, 
            state = state,
            observation = observation
        )
        self.logger.info(f"query_tags: {query_tags}\n")
        # print(f"query_tags: {query_tags}")
        self.logger.info(f"next subgoal: {subgoal}")
        # print(f"next subgoal: {subgoal}")
        
        if mode == None:
            mode = get_mode(
                observation = observation,
                task_type = task_type
            )
        self.logger.info(f"task_type: {task_type}")
        self.logger.info(f"----- mode -----: {mode}")
        
        if mode == "episodic_memory":
            prompt_template = DefaultEpisodicPrompt()
        elif mode == "semantic_memory":
            prompt_template = DefaultSemanticPrompt()
        else:
            prompt_template = DefaultProceduralPrompt()
        DIR_PATH = os.environ.get("DIR_PATH", None)
        with open(DIR_PATH+"/retrieve.jsonl","a",) as output:
            output.write(json.dumps({"next_subgoal": next_subgoal, "query_tags": query_tags})+"\n")
        semantic_nodes, procedural_nodes = [], []
        if mode in ["semantic_memory", "episodic_memory"]:
            semantic_nodes = self.retrieve_semantic_nodes(
                semantic_memory={
                    "semantic_memory": observation,
                    "tags": query_tags
                }, 
                value_func_tag = self.tag_relevant,
                value_func = self.semantic_relevant,
                write=True
            )
        if mode in ["procedural_memory", "episodic_memory"]:
            procedural_nodes = self.retrieve_procedural_nodes(
                subgoal = next_subgoal, 
                value_func_subgoal = self.subgoal_relevant, 
                value_func = self.procedural_relevant
            )
            
        semantic_memory_str = ""
        procedural_memory_str = ""
        episodic_memory_str = ""
        
        if mode == "episodic_memory":
            
            episodic_memory_str = self.retrieve_episodic_memory(
                observation=observation
            )
            
            self.logger.info(f"---- episodic_memory_str: \n{episodic_memory_str}")
            
        elif mode == "semantic_memory":
            semantic_ids_all = []
            for semantic_node in semantic_nodes:
                semantic_ids_all.append(semantic_node.semantic_id)
                # for bro_semantic_node in semantic_node.bro_semantic_nodes:
                #     semantic_ids_all.append(bro_semantic_node.semantic_id)
            semantic_ids_all = list(set(semantic_ids_all))
            semantic_memory_str = ""

            if len(semantic_ids_all) == 0:
                semantic_memory_str = "No relevant fact"
            semantic_id2node: Dict[int, Any] = {
                node.semantic_id: node for node in self.semantic_nodes
            }
            ids = list(semantic_id2node.keys())
            max_w = len(str(max(ids)))  
            for i, semantic_id in enumerate(semantic_ids_all):
                # semantic_memory_str += f"Relevant Fact {i}: {semantic_id2node[semantic_id].get_semantic_memory()}\n"
                semantic_memory_str += f"Fact {i} (Sem Node {semantic_id:<{max_w}}): {semantic_id2node[semantic_id].get_semantic_memory()}\n"
                
            self.logger.info(f"---- semantic_memory_str: \n{semantic_memory_str}")
            
        elif mode == "procedural_memory":
            if len(procedural_nodes) == 0:
                procedural_memory_str = "No relevant experiences"

            for i, procedural_node in enumerate(procedural_nodes):
                # procedural_memory_str += f"Relevant Experience {i}: {procedural_node.get_procedural_memory()}\n"
                procedural_memory_str += f"Experience {i} (Proc Node {procedural_node.procedural_id:<{max_w}}): {procedural_node.get_procedural_memory()}\n"
                
            self.logger.info(f"---- procedural_memory_str: \n{procedural_memory_str}")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        prompt_obj = prompt_template
        variables = {
            "goal": goal,
            "subgoal": subgoal,
            "state": state,
            "observation": observation,
            "semantic_memory": semantic_memory_str,
            "procedural_memory": procedural_memory_str,
            "episodic_memory": episodic_memory_str,
            "time": time
        }
        messages = prompt_obj.build_messages(variables)
        messages = [{"role": m.role, "content": m.content} for m in messages]
        sel_type = ""
        if mode == "episodic_memory":
            sel_type = "episodic_memory"
        elif mode == 'semantic_memory':
            sel_type = "semantic_memory"
        elif mode == "procedural_memory":
            sel_type = "procedural_memory"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        return messages, variables, sel_type
    
    def retrieve_and_reason(
        self,
        goal: str=None,
        subgoal: str=None,
        state: str=None,
        observation: str=None,
        prompt_template: PromptBase=None,
        time: str="",
        task_type: str="",
        mode: str=None,
        llm_client: Callable[..., str]=call_qwen,
        llm_kwargs: Optional[Dict[str, Any]]=None,
    ) -> str:
        messages, _, _ = self.retrieve_memory(
            goal=goal,
            subgoal=subgoal,
            state=state,
            observation=observation,
            prompt_template=prompt_template,
            time=time,
            task_type=task_type,
            mode=mode,
        )
        if llm_kwargs is None:
            llm_kwargs = {}
        reasoning = llm_client(messages=messages, **llm_kwargs)
        return reasoning
    
    
    def merge_semantic(self, id1: int, id2: int):
        epis_id2node = {node.episodic_id: node for node in self.episodic_nodes}
        semantic_id2node={x.semantic_id:x for x in self.semantic_nodes}
        tag_id2node={x.tag_id:x for x in self.tag_nodes}
        
        semantic_node_1: SemanticNode = semantic_id2node[id1]
        semantic_node_2: SemanticNode = semantic_id2node[id2]
        merge_decision = get_new_semantic(semantic_node_1.get_semantic_memory(), semantic_node_2.get_semantic_memory())
        merged_semantic_str = merge_decision["merged_statement"]
        if_del_node1 = merge_decision["deactivate_earlier"]
        if_del_node2 = merge_decision["deactivate_later"]
        simple_reasoning = merge_decision["simple_reasoning"]
        print(f"----- simple_reasoning: {simple_reasoning}")
        
        embedding = get_embedding(merged_semantic_str)
        merged_sem_node = SemanticNode(
            semantic_memory = {}, 
            semantic_memory_str = merged_semantic_str, 
            semantic_memory_embedding = embedding, 
            semantic_id = len(self.semantic_nodes), 
            time = self.semantic_time,
            son=[semantic_id2node[id1], semantic_id2node[id2]]
        )
        self.semantic_nodes.append(merged_sem_node)
        # self.semantic_time += 1
        episodic_node_ids = [episodic_node.episodic_id for episodic_node in semantic_node_1.episodic_nodes] + [episodic_node.episodic_id for episodic_node in semantic_node_2.episodic_nodes]
        episodic_node_ids = list(set(episodic_node_ids))
        
        merged_sem_node.episodic_nodes = [epis_id2node[episodic_id] for episodic_id in episodic_node_ids]
        tag_node_ids = [tag_node.tag_id for tag_node in semantic_node_1.tag_nodes] + [tag_node.tag_id for tag_node in semantic_node_2.tag_nodes]
        tag_node_ids = list(set(tag_node_ids))
        merged_sem_node.tag_nodes = [tag_id2node[tag_id] for tag_id in tag_node_ids]
        merged_sem_node.tags = semantic_node_1.tags + semantic_node_2.tags
        merged_sem_node.tags = list(set(merged_sem_node.tags))
        return merged_sem_node,if_del_node1,if_del_node2
    
    
    # ================ HPQA specific ================
    def update_semantic_subgraph(
        self,
        *,
        write_to_disk: bool = False,
        allow_merge_with_common_episodic_nodes: bool = False,
        merge_threshold: float = 0.5,
        max_merges_per_node: int = 1,
        max_candidates_per_tag: int = 200,
        max_total_candidates: int = 800,
        min_credibility_to_keep_active: int = -10,
        credibility_decay: int = 0,
        only_update_recent_window: Optional[int] = None,) -> Dict[str, int]:
        update_stats = {
            "scanned_semantic": 0,
            "skipped_inactive": 0,
            "merged_pairs": 0,
            "new_semantic_nodes": 0,
            "soft_deactivated": 0,
            "index_rebuilt": 0,
            "repaired_edges": 0,
        }

        # ---------- helpers ----------
        def _ensure_semantic_active_flag():
            for s in self.semantic_nodes:
                if not hasattr(s, "is_active"):
                    s.is_active = True

        def _repair_bidirectional_edges_for_semantic(s):
            """
            Ensure: for each tag in s.tag_nodes, tag.semantic_ids contains s.semantic_id.
            """
            nonlocal update_stats
            for t in getattr(s, "tag_nodes", []):
                if hasattr(t, "semantic_ids"):
                    if s.semantic_id not in t.semantic_ids:
                        t.semantic_ids.add(s.semantic_id)
                        update_stats["repaired_edges"] += 1
                if hasattr(t, "semantic_nodes"):
                    # best-effort; avoid O(deg) 'in' check by allowing duplicates rarely
                    t.semantic_nodes.append(s)

        def _iter_update_scope(time_st: int) -> range:
            # Decide which semantic ids to scan
            if only_update_recent_window is None:
                # all nodes with time < time_st
                end = 0
                for i, s in enumerate(self.semantic_nodes):
                    if s.time >= time_st:
                        end = i
                        break
                else:
                    end = len(self.semantic_nodes)
                return range(end)

            # only nodes whose time in [time_st - window, time_st)
            low = max(0, time_st - only_update_recent_window)
            ids = []
            for i, s in enumerate(self.semantic_nodes):
                if s.time < low:
                    continue
                if s.time >= time_st:
                    break
                ids.append(i)
            return ids  # list of ids


        # ---------- preparation ----------
        time_st = self.semantic_time  # freeze boundary
        _ensure_semantic_active_flag()

        # optional: credibility decay for old nodes
        if credibility_decay != 0:
            for sem_node in self.semantic_nodes:
                if sem_node.time < time_st and getattr(sem_node, "is_active", True):
                    sem_node.Credibility -= credibility_decay

        # ---------- main loop ----------
        scope = _iter_update_scope(time_st)
        print(f'len(scope): {len(scope)}')
        for sid in tqdm(scope):
            sem_node: SemanticNode = self.semantic_id2node[sid]
            # print('\n--------------------------------')
            # print(f'curr_sem_node.semantic_id: {sem_node.semantic_id}')
            
            update_stats["scanned_semantic"] += 1
            
            # -- early stopping 
            if not getattr(sem_node, "is_active", True):
                update_stats["skipped_inactive"] += 1
                continue
            if getattr(sem_node, "updated", False):
                continue

            # -- soft deactivate based on credibility (do not physically delete)
            if getattr(sem_node, "Credibility", 0) < min_credibility_to_keep_active:
                if hasattr(sem_node, "is_active"):
                    sem_node.is_active = False
                update_stats["soft_deactivated"] += 1
                continue

            # -- collect candidate sem_node to be merged via tags, with degree capping
            cand_ids = set()
            for tag_node in getattr(sem_node, "tag_nodes", []):
                ids = [semantic_node.semantic_id for semantic_node in tag_node.semantic_nodes]
                if not ids:
                    continue

                # cap per tag to avoid blow-up for high-degree tags
                if len(ids) > max_candidates_per_tag:
                    # heuristic: prefer newer nodes by time (if you have time ordering)
                    # fall back to random sample if needed
                    # Here: sample for simplicity; can be replaced with "take most recent" if you maintain recency lists
                    ids = random.sample(ids, k=max_candidates_per_tag)

                cand_ids.update(ids)
                if len(cand_ids) >= max_total_candidates:
                    break
                
            # IMPORTANT: for each tag's cand ids, delete its own semantic id it belongs to
            cand_ids.discard(sem_node.semantic_id)
            
            
            # Filter candidates (freeze boundary, ordering constraint, active)
            filtered_cand_ids: List[int] = []
            for cand_id in cand_ids:
                cand_sem_node = self.semantic_id2node[cand_id]
                if not getattr(cand_sem_node, "is_active", True):
                    continue
                if cand_sem_node.time >= time_st:
                    continue
                if getattr(cand_sem_node, "updated", False):
                    continue
                # IMPORTANT: avoid symmetric duplicates / repeated work
                if cand_sem_node.semantic_id <= sem_node.semantic_id:
                    continue
                filtered_cand_ids.append(cand_id)

            if not filtered_cand_ids:
                continue
            
            
            # Score candidates and take top-k
            scored: List[Tuple[float, int]] = []
            for cand_id in filtered_cand_ids:
                cand_sem_node = self.semantic_id2node[cand_id]
                rel = get_similarity(sem_node.embedding, cand_sem_node.embedding)
                val = self.semantic_equal.evaluate(Relevance=rel)
                scored.append((val, cand_id))
            # print(f'scored: {scored}')

            k = getattr(self.semantic_equal, "k", 10)
            topk_cands = heapq.nlargest(k, scored, key=lambda x: x[0])

            merges_done = 0
            for val, cand_id in topk_cands:
                if val < merge_threshold:
                    continue
                print('\n--------------------------------')
                print(f'----- filtered_cand_ids: {filtered_cand_ids}')
                print(f'----- revisited: scored candidates: {scored}')
                print(f'----- sem_id1: {sem_node.semantic_id}')
                print(f'----- sem_id2: {cand_id}')
                print(f'----- sem_node1.str: {sem_node.get_semantic_memory()}')
                print(f'----- sem_node2.str: {self.semantic_id2node[cand_id].get_semantic_memory()}')
                
                epid_1=[x.episodic_id for x in sem_node.episodic_nodes]
                epid_2=[x.episodic_id for x in cand_sem_node.episodic_nodes]
                if len(set(epid_1) & set(epid_2)) > 0 and not allow_merge_with_common_episodic_nodes:
                    print(f'----- epid_1: {epid_1}')
                    print(f'----- epid_2: {epid_2}')
                    print(f'----- skip merging because of common episodic nodes')
                    continue
                
                # Merge (creates a new semantic node)
                num_sem_node_before = len(self.semantic_nodes)
                new_node,if_del_node1,if_del_node2 = self.merge_semantic(sem_node.semantic_id, cand_id)
                if if_del_node1:
                    self.semantic_id2node[sem_node.semantic_id].is_active = False
                    if write_to_disk:
                        update_semantic_hpqa_ver(semantic_id = sem_node.semantic_id, is_active = False)
                if if_del_node2:
                    self.semantic_id2node[cand_id].is_active = False
                    if write_to_disk:
                        update_semantic_hpqa_ver(semantic_id = cand_id, is_active = False)
                
                if write_to_disk:
                    save_semantic_hpqa_ver(semantic_memory_str = new_node.get_semantic_memory(), 
                            semantic_embedding = new_node.embedding,
                            semantic_id = new_node.semantic_id, 
                            episodic_ids = [x.episodic_id for x in new_node.episodic_nodes],
                            bro_semantic_ids = [x.semantic_id for x in sem_node.bro_semantic_nodes+cand_sem_node.bro_semantic_nodes],
                            tags = new_node.tags, 
                            tag_ids = [tag_node.tag_id for tag_node in new_node.tag_nodes],
                            time = new_node.time,
                            son_semantic_ids = [x.semantic_id for x in new_node.son_semantic],
                            )
                
                update_stats["merged_pairs"] += 1
                update_stats["new_semantic_nodes"] += (len(self.semantic_nodes) - num_sem_node_before)
                self.semantic_id2node = {x.semantic_id: x for x in self.semantic_nodes}
                
                # # Decay credibility along subtree (optional)
                # if credibility_decay > 0:
                #     # if you have dfs, call it; otherwise best-effort
                #     if hasattr(self, "dfs_semantic_decay"):
                #         self.dfs_semantic_decay(sem_node)
                #     else:
                #         sem_node.Credibility -= 1

                # Repair edges for the newly created node to keep tag recall closed-loop
                
                print(f'----- new_node.id: {new_node.semantic_id}')
                print(f'----- merged_semantic_str: {new_node.get_semantic_memory()}')
                print(f'----- if_del_node1: {if_del_node1}')
                print(f'----- if_del_node2: {if_del_node2}')
                print(f'----- new_node.tags: {new_node.tags}')
                _repair_bidirectional_edges_for_semantic(new_node)

                merges_done += 1
                if merges_done >= max_merges_per_node:
                    break

            # for current semantic node, update its 'updated' flag to True
            if merges_done > 0:
                sem_node.updated = True
                #### if at least 3 merge is done, break the loop
                if merges_done >= 3:
                    break
                else:
                    continue
                
        print('----- update_stats:')
        print(update_stats)
           
        node_num_stat = {
            "semantic_nodes": len(self.semantic_nodes),
            "tag_nodes": len(self.tag_nodes),
            "episodic_nodes": len(self.episodic_nodes),
            "procedural_nodes": len(self.procedural_nodes),
            "subgoal_nodes": len(self.subgoal_nodes),
            }
        print(f"Memory Graph updated. Node Num Statistics: \n{node_num_stat}")
        
        return update_stats
