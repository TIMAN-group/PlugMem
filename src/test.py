'''
Write any submodule you wanna test here
'''
import re
import json
from utils import call_qwen, call_gpt, get_embedding
from memory_retrieving.retrieving_inference import get_mode

def test_get_mode(observation: str=None, task_type: str=None):
    observation="User Say: How many months have passed since I participated in two charity events in a row, on consecutive days?"
    task_type="temporal-reasoning"
    print(get_mode(observation=observation, task_type=task_type))

test_get_mode()

