from __future__ import annotations

from typing import List, Mapping

from prompt_base import PromptBase, ChatMessage


class GetPlanPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an assistant whose job is to decide what to recall for an LLM agent."
        )
        # user_template = (
        #     "Following information will be provided:\n"
        #     "Goal: A description of the overall task the agent aims to complete.\n"
        #     "Current Subgoal: The subgoal the agent is currently pursuing.\n"
        #     "Current State: A description of the agent’s current internal state.\n"
        #     "Current Observation: A description of what the agent just observed.\n"
        #     "Your Task:\n"
        #     "1. Generate some query tags. A tag is one word about entity related to the information the agent needs to retrieve from the external memory database (don't generate tag 'user'). The number of tags is flexible."
        #     "2. Generate the next subgoal the agent should pursue. This subgoal should describe the next concrete action or step the agent plans to execute.\n"
        #     "Step:\n"
        #     "1. Reasoing\n"
        #     "- Analyze the goal, current subgoal, state, and observation.\n"
        #     "- Generate the tags for the information agent needs to pursue tha goal\n"
        #     "- Tags should important nouns from the observation that reflect the factual information the agent needs from the memory database.\n"
        #     "- The next subgoal should directly move the agent closer to the overall goal.\n"
        #     "Output Format:\n"
        #     "### Reasoning\n"
        #     "[You process of analyzing the information and completing the task]\n"
        #     "### Tags\n"
        #     "**Tags:** [\"tag0\", \"tag1\", \"tag2\", \"tag3\", ...]\n (for example: \"Central Area\", \"focal point\",\"famous for\", etc)" 
        #     "### Next Subgoal\n"
        #     "[A single best next subgoal that the agent should pursue now.]\n"
        #     "---\n"
        #     "Input:\n"
        #     "Goal: {goal}\n"
        #     "Current Subgoal: {subgoal}\n"
        #     "Current State: {state}\n"
        #     "Current Observation: {observation}\n"
        # )
        
        user_template = (
            "You are an expert at analyzing an agent’s goal and current observation and generating retrieval tags for a goal-directed question-answering system.\n"
            "Your setting:\n"
            "- Goal: The agent’s overall objective to accomplish.\n"
            "- Current Subgoal: The subgoal the agent is currently pursuing. (can be None) \n"
            "- Current State: A description of the agent’s current internal state. (can be None) \n"
            "- Input (Current Observation): The agent’s most recent observation (typically a question or task instruction).\n"
            "- Task: Extract a prioritized set of high-quality tags that are most likely to retrieve information that directly helps accomplish the Goal.\n"
            "\n"
            "Instructions:\n"
            "1) Goal-directed Tag Selection (CRITICAL):\n"
            "- Read the Goal and the Current Observation carefully.\n"
            "- Only generate tags that are HIGHLY LIKELY to retrieve evidence needed to solve/complete the Goal.\n"
            "- Prefer tags that identify:\n"
            "  - The target entity/entities the Goal is asking about (people, organizations, places, works, events).\n"
            "  - Bridge entities implied by the observation that are likely required for multi-hop retrieval.\n"
            "  - Explicit constraints: dates, years, roles, titles, unique descriptors, numbers.\n"
            "- Avoid low-signal or generic tags that are unlikely to retrieve helpful evidence (e.g., \"known for\", \"famous\", \"character\" unless the Goal specifically depends on them).\n"
            "\n"
            "2) Concrete, Grounded Tags:\n"
            "- The MAJORITY of tags MUST be short text spans copied VERBATIM from the Goal or the Current Observation (exact substrings).\n"
            "- You MAY add a SMALL number of non-literal tags only if they are short, strongly implied, and clearly necessary for retrieval (e.g., a canonical name expansion or a standard alias).\n"
            "- If a tag is a verb, you MUST use its base (lemma) form (e.g., \"direct\", not \"directed\" or \"directing\").\n"
            "\n"
            "3) Prioritization & Quantity:\n"
            "- Output 5–12 tags total (not \"as many as possible\").\n"
            "- Sort tags by expected retrieval usefulness (most useful first).\n"
            "- Ensure tags are content-bearing and relatively short.\n"
            "\n"
            "4) CRITICAL – Forbidden tags:\n"
            "- Do NOT generate the tag \"user\".\n"
            "- Do NOT use meta-labels or type names such as \"Name\", \"Person\", \"Year\", \"Date\", \"City\", \"Country\", \"Location\", \"Genre\", \"FilmTitle\", etc.\n"
            "\n"
            "Output Format:\n"
            "### Reasoning\n"
            "[You process of analyzing the information and completing the task]\n"
            "### Tags\n"
            "**Tags:** [\"tag0\", \"tag1\", \"tag2\", \"tag3\", ...]\n (for example: \"Central Area\", \"focal point\",\"famous for\", etc)" 
            "### Next Subgoal\n"
            "[A single best next subgoal that the agent should pursue now.]\n"
            "---\n"
            "Input:\n"
            "Goal: {goal}\n"
            "Current Subgoal: {subgoal}\n"
            "Current State: {state}\n"
            "Current Observation: {observation}\n"
        )

        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetNewSemanticPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are tasked with merging two pieces of similar information into a single, coherent statement."
        )
        # user_template = (
        #     "The two pieces of information are related to the same topic, but one came earlier (Information 1) and the other came later (Information 2). "
        #     "Your goal is to combine them into a new, more comprehensive statement, ensuring it is clear, concise, and provides a better overall understanding of the topic.\n"
        #     "Information 1 (Earlier Information): {memory_earlier}\n"
        #     "Information 2 (Later Information): {memory_later}\n"
        #     "Please merge the two pieces of information, ensuring you keep track of the timeline and provide a unified, improved summary."
        # )
        user_template = """
You are given two memory items about a related topic. One came earlier (Information 1) and the other came later (Information 2).

Your tasks:
(1) Merge them into ONE improved, clear, concise statement. Do not invent new facts.
(2) Decide whether to deactivate (soft delete) the original two nodes after merging.

Inputs:
Information 1 (Earlier Information): {memory_earlier}
Information 2 (Later Information): {memory_later}

Deactivation decision rules (choose exactly ONE case):

Case A: "UPDATE_SAME_FACT"
- Condition: Information 1 and 2 are essentially describing the same fact/event, and Information 2 mainly updates/corrects/refines details of Information 1.
- Action: deactivate BOTH originals (earlier and later) because the merged node fully supersedes them.

Case B: "SAME_TOPIC_MERGE_WELL"
- Condition: Information 1 and 2 are strongly related under the same topic, and the merged statement reads naturally as a unified summary (not an awkward splice).
- Action: deactivate BOTH originals (earlier and later).

Case C: "WEAK_RELATED_STITCH_RISK"
- Condition: Information 1 and 2 are only weakly related; merging feels like stitching two segments; and deactivating either original would likely harm future retrieval because the merged embedding may become a mixed “四不像”.
- Action: deactivate NEITHER original.

Hard constraints:
- Output MUST be valid JSON (no Markdown, no extra text).
- relationship MUST be one of the three labels above.
- If relationship is Case A or B => deactivate_earlier=true AND deactivate_later=true.
- If relationship is Case C => deactivate_earlier=false AND deactivate_later=false.
- If the two memories conflict, prefer Information 2 as the more up-to-date.
- Output the simple reasoning of why you made the decision.

Output MUST be valid JSON with exactly these keys:
- merged_statement (string)
- relationship ("UPDATE_SAME_FACT" | "SAME_TOPIC_MERGE_WELL" | "WEAK_RELATED_STITCH_RISK")
- deactivate_earlier (boolean)
- deactivate_later (boolean)
- simple_reasoning (string)

Return ONLY the JSON object.
"""
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetNewSubgoalPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an assistant that merges two similar goals into one unified goal."
        )
        user_template = (
            'Each goal may contain overlapping or complementary information.\n'
            'Your task is to carefully combine them into a single, coherent, and well-structured goal that preserves all important details from both.\n'
            'Avoid redundancy and ensure the merged goal sounds natural and consistent in tone.\n'
            'Input:\n'
            'Earlier goal: {goal_1}\n'
            'Later goal: {goal_2}\n'
            'Output:\n'
            'Merged goal: [Write the unified goal here]'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]
        
class GetModePrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are a helpful assistant."
        )
        user_template = (
            'You are given a task type that the agent is pursuing and the observation from the task\n'
            'Please analyze the task type and observation to determine the type of memory required to complete it effectively. There are three possible memory types:\n'
            'Episodic Memory: This is needed if the task requires you to answer questions about specific events, such as recalling what happened during a certain situation or interaction.\n'
            'Semantic Memory: This is needed if the task requires you to recall objective information, such as details about the user (preferences, information) or general knowledge to make recommendations or decisions.\n'
            'Procedural Memory: This is needed if the task is completing a subgoal under an interactive environment (etc. website) that agent need to perform a workflow.\n'
            'First analyze that task and observation and decide which only one memory type needed\n'
            'Output Fomat:\n'
            '### Reasoning\n'
            '[Your analyze of which memory is needed depending on task and observation]\n'
            '### Memory Type\n'
            '[Your final deciosn, episodic_memory or semantic_memory or procedural_memory]\n'
            '---\n'
            'Input:\n'
            'Task type: {task_type}\n'
            'Observation: {observation}'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]