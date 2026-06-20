import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the correct paths to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/eval/longmemeval"))

from plugmem_client import PlugMemClient

class DummyMemory:
    def __init__(self, goal="", episodic=None, semantic=None, procedural=None):
        self.goal = goal
        self.memory = {
            "episodic": episodic or [],
            "semantic": semantic or [],
            "procedural": procedural or []
        }
        self.session_id = "test-session"


class TestPlugMemClient(unittest.TestCase):

    @patch("plugmem_client._post")
    def test_insert_2d_episodic(self, mock_post):
        # 2D episodic memory structure: list of list of dicts
        mem = DummyMemory(
            goal="Test 2D Goal",
            episodic=[[
                {"observation": "obs1", "action": "act1", "state": "state1", "reward": 1.0, "subgoal": "sub1", "time": 100}
            ]],
            semantic=[{"semantic_memory": "fact", "tags": ["tag"]}]
        )
        
        client = PlugMemClient(graph_id="test-graph", auto_create=False)
        client.insert(mem)
        
        # Verify the body payload passed to _post
        mock_post.assert_called_with(
            "/graphs/test-graph/memories",
            {
                "mode": "structured",
                "session_id": "test-session",
                "episodic": [[
                    {"observation": "obs1", "action": "act1", "state": "state1", "reward": "1.0", "subgoal": "sub1", "time": 100}
                ]],
                "semantic": [{"semantic_memory": "fact", "tags": ["tag"]}],
                "procedural": []
            }
        )

    @patch("plugmem_client._post")
    def test_insert_1d_episodic(self, mock_post):
        # 1D episodic memory structure: list of dicts (HotpotQA format)
        mem = DummyMemory(
            goal="Test 1D Goal",
            episodic=[
                {"observation": "obs1", "action": "act1", "state": "state1", "reward": 1.0, "subgoal": "sub1", "time": 100}
            ],
            semantic=[{"semantic_memory": "fact", "tags": ["tag"]}]
        )
        
        client = PlugMemClient(graph_id="test-graph", auto_create=False)
        client.insert(mem)
        
        # Verify the body payload passed to _post. The episodic field should be normalized to 2D list.
        mock_post.assert_called_with(
            "/graphs/test-graph/memories",
            {
                "mode": "structured",
                "session_id": "test-session",
                "episodic": [[
                    {"observation": "obs1", "action": "act1", "state": "state1", "reward": "1.0", "subgoal": "sub1", "time": 100}
                ]],
                "semantic": [{"semantic_memory": "fact", "tags": ["tag"]}],
                "procedural": []
            }
        )

    @patch("plugmem_client._post")
    def test_insert_trajectory_mode_1d(self, mock_post):
        # Raw trajectory mode (no semantic/procedural memory), 1D episodic
        mem = DummyMemory(
            goal="Test Goal",
            episodic=[
                {"observation": "obs1", "action": "act1"}
            ]
        )
        
        client = PlugMemClient(graph_id="test-graph", auto_create=False)
        client.insert(mem)
        
        # Verify the body payload passed to _post
        mock_post.assert_called_with(
            "/graphs/test-graph/memories",
            {
                "mode": "trajectory",
                "goal": "Test Goal",
                "steps": [{"observation": "obs1", "action": "act1"}],
                "session_id": "test-session"
            }
        )
