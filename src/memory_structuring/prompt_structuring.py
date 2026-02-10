from __future__ import annotations

from typing import Mapping, List

from prompt_base import PromptBase, ChatMessage


class GetSubgoalPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are analyzing the behavior of an intelligent agent (an LLM-based agent) that is working toward a specific goal in a dynamic environment."
        )
        user_template = (     
            'At time t, the agent takes an action based on its state, observation, and overall goal.\n'
            'Your task is to infer the subgoal — the immediate or intermediate objective — that best explains why the agent chose this action.\n'
            'Use the following information as context:\n'
            'Overall Goal: {goal}\n'
            'Current State (summary of past context): {state}\n'
            'Current Observation: {observation}\n'
            'Action at time t: {action}\n'
            'Step 1: Reasoning\n'
            'Analyze how the current state and observation relate to the overall goal.\n'
            'Explain how the given action helps the agent make progress toward that goal — possibly by achieving a smaller intermediate objective.\n'
            'Be explicit and causal: describe why this action makes sense given the context.\n'
            'Step 2: Subgoal Inference'
            'After reasoning, infer the agent’s likely subgoal — a short natural-language statement that describes the immediate purpose behind the action.\n'
            'Output Format:\n'
            '### Reasoning\n'
            '[Your reasoning process — a few sentences explaining how the action relates to the goal and context]\n'
            '### Subgoal\n'
            '[A short sentence describing the inferred subgoal]'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetRewardPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an expert trajectory evaluator. Your task is to analyze one step of an agent’s decision-making process within a larger goal-directed task."
        )
        user_template = (
            'You will be given:\n'
            'Goal: the agent’s overall objective.\n'
            'State (at time t): what the agent knew and had done before taking the action.\n'
            'Action (at time t): the single action the agent chose.\n'
            'Observation (at time t + 1): the immediate outcome produced by that action.\n'
            "Your task is to infer the reward — that is, a evaluation in natural language on how the agent's action contributes (positively or negatively) to achieving the overall goal, based on the resulting observation.\n"
            'Follow these steps carefully:\n'
            '1. Reasoning Process:\n'
            'Explain how the action relates to the Goal given the State, and whether the Observation matches the expected helpful or unhelpful outcome.\n'
            'Consider whether the action advances progress, causes setbacks, reveals new useful information, or wastes effort.\n'
            'Summarize your reasoning about the causal contribution of the action to the goal.\n'
            '2. Final Reward:\n'
            "Use descriptive language to write a concise natural-language evaluation of the agent's action.\n"
            'The reward should express how much and in what way the action helped or hindered achieving the goal.\n'
            'Input:\n'
            'Goal: {goal}\n'
            'State (at time t): {state}\n'
            'Action (at time t): {action}\n'
            'Observation (at time t + 1): {observation}\n'
            'Output format:\n'
            '### Reasoning\n'
            '[Your reasoning process here]\n'
            '### Reward\n'
            "[Natural-language reward statement that evaluate the agent's action]"
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetStatePrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an LLM reasoning engine that updates an agent’s internal state representation based on its ongoing trajectory."
        )
        user_template = (
            'You will receive four pieces of information:\n'
            "Goal: The agent's current objective or task.\n"
            "Previous State (at time t): A natural language summary describing the agent's context, history, and partial progress so far.\n"
            'Action (at time t): The action the agent decided to take next, expressed in natural language.\n'
            'Observation (at time t+1): The outcome or feedback resulting from that action.\n'
            'Your task is to derive the new updated state — a coherent natural language summary that integrates all relevant information from the previous state, the action, and the new observation.'
            'Steps to Follow:\n'
            'Interpret the Inputs: Examine the goal, the previous state, the action, and the observation to understand what has changed in the agent’s situation and the detailed information about location and time.\n'
            'Reason about the Update: Describe the logical process by which the new state should differ from the previous one. Identify what progress has been made, what new information was gained, and how the context or focus may have shifted.\n'
            'Generate the Updated State: Write a clear and concise natural-language description summarizing the new state of the agent at time t+1. The new state should:\n'
            '-Included all the detailed information in action and observation, especially information about location and time\n'
            '-Integrate the outcome of the latest action and observation,\n'
            'Output Format:\n'
            '### Reasoning\n'
            '(Explain step by step how the new state should be updated based on the inputs.)\n'
            '### State\n'
            '(Provide the final updated state summary here.)\n'
            '---\n'
            'Input:\n'
            'Goal: {goal}\n'
            'Previous State (at time t): {state}\n'
            'Action (at time task): {action}\n'
            'Observation (at time t+1): {observation}'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetSemanticPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an expert information extractor. "
        )
        user_template = (
            "You are an expert at extracting precise, factual information from documents. "
            "Your output must prioritize specificity, avoid ambiguity, eliminate redundancy, and strictly follow all formatting rules.\n"
            "\n"
            "**CORE INSTRUCTIONS:**\n"
            "\n"
            "1) Fact/Statement Extraction & Deduplication:\n"
            "    * Identify distinct, factual statements from the document.\n"
            "    * Resolving Vague References: If the subject of a statement is a pronoun or a vague description (e.g., 'the film', 'the band', 'the company', 'he', 'she', 'they'), you MUST rewrite the statement based on understanding of whole document so that the subject is a fully specified and concrete entity name taken from the document. You are NOT allowed to keep vague subjects in the final fact/statement. \n"
            "    * Example of Resolving Vague References:\n"
            "      BAD: 'The film was directed by xxx.', 'This movie is produced by xxx'\n"
            "      GOOD: 'Vaada Poda Nanbargal was directed by Manikai.', 'Vaada Poda Nanbargal is produced by xxx'\n"
            "\n"
            "    * Concrete Phrasing: Every statement MUST be phrased using explicit, identifiable names or titles. "
            "      You are FORBIDDEN from using ANY vague references including but not limited to: 'the tour', 'the film', 'the movie', 'the band', 'it', 'he', 'she', or 'they'.\n"
            "    * Example of Concrete Phrasing:\n"
            "       BAD: 'The tour earned over $50 million.'\n"
            "       GOOD: 'NSYNC's Second II None Tour earned over $50 million.'\n"
            "\n"
            "    * Statement Length Policy (IMPORTANT): Each fact/statement does NOT have to be a single short sentence. "
            "      A statement MAY be a compact multi-sentence block that groups tightly related information, but it must contain AT MOST 4 sentences total. "
            "      These sentences should come from the original document material, potentially lightly edited ONLY to resolve vague references.\n"
            "    * Avoid Redundancy: You MUST merge similar or overlapping facts into single, comprehensive statements. "
            "Do NOT create multiple statements that repeat the same core information with minor variations.\n"
            "\n"
            "2) Tag Generation:\n"
            "    * For each fact/statement, generate a list of tags.\n"
            "    * The number of tags per fact is flexible and should reflect the information density of the statement.\n"
            "    * Tags should cover key spans such as: entity names, years, numbers, nationalities, languages, genres, roles, object types, descriptive words,etc.\n"
            "    * The MAJORITY of tags SHOULD be SHORT TEXT SPANS copied VERBATIM from the statement (exact substrings). These tags are directly grounded in the surface form of the text.\n"
            "    * You MAY occasionally create additional tags that are not literal substrings of the current statement, in the following cases:\n"
            "        - You combine adjacent words into a single phrase (e.g., 'romantic comedy film').\n"
            "        - You import a surface span (exact phrase) from ANOTHER part of the document to make the meaning of the current statement explicit (for example, when resolving pronouns or phrases like 'the film', 'the band', etc.).\n"
            "    * When the subject of a statement is a pronoun or a vague description (e.g., 'the movie', 'the band', 'the hotel'), you MUST add at least one tag that names the underlying entity explicitly, using the exact surface form that appeared elsewhere in the document. This cross-statement tag may come from a previous or later sentence as long as it is clearly the same entity.\n"
            "    * If a tag is a verb, you MUST use its base (lemma) form (e.g., `play`, not `played`, `playing`, or `to play`).\n"
            # "    * Preferred behavior: whenever possible, use literal spans from the statement; only a small minority of tags should be created beyond the exact text.\n"
            "    * No Schema/Type Labels: You are FORBIDDEN from using meta-labels or ontology-like category names such as: 'Name', 'Person', 'Year', 'Date', 'City', 'Country', 'Location', 'Genre', 'Language', 'FilmTitle', etc. Tags are NOT type labels; they are content-bearing phrases.\n"
            "    * Tags should be relatively short and as fine-grained as needed. For example, adjectives and modifiers like 'Indian', 'Tamil-language', 'romantic', 'comedy' SHOULD usually be separate tags if they appear in the statement.\n"
            "\n"
            "OUTPUT CONSTRAINTS:\n"
            "    * Extract up to 10 facts, but prioritize QUALITY over quantity. "
            "If there are fewer than 10 truly distinct facts, output fewer.\n"
            "    * Each fact must provide unique information not covered by other facts.\n"
            "    * ABSOLUTELY NO generic references – every statement must explicitly name the specific entity.\n"
            "\n"
            "**DOCUMENT:**\n"
            "{observation}\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "### Facts\n"
            "1. **Statement:** [statement]\n"
            "   **Tags:** [tag0, tag1, tag2, tag3, ...]\n"
            "2. **Statement:** [statement]\n"
            "   **Tags:** [tag0, tag1, tag2, tag3, ...]\n"
            "...\n"
        )
        
        user_template_discarded = (
            "You are an expert at extracting factual information from documents. "
            "Your output must prioritize specificity, avoid ambiguity, eliminate redundancy, and strictly follow all formatting rules.\n"
            "\n"
            "**CORE INSTRUCTIONS:**\n"
            "\n"
            "1) Fact/Statement Extraction & Deduplication:\n"
            "    * Identify distinct, factual statements from the document.\n"
            "    * Resolving Vague References: If the subject of a statement is a pronoun or a vague description (e.g., 'the film', 'the band', 'the company', 'he', 'she', 'they'), you MUST rewrite the statement based on understanding of whole document so that the subject is a fully specified and concrete entity name taken from the document. You are NOT allowed to keep vague subjects in the final fact/statement. \n"
            "      - Example of Resolving Vague References:\n"
            "        BAD: 'The film was directed by xxx.', 'This movie is produced by xxx'\n"
            "        GOOD: 'Vaada Poda Nanbargal was directed by Manikai.', 'Vaada Poda Nanbargal is produced by xxx'\n"
            "\n"
            "    * Statement Length Policy (IMPORTANT): Each fact/statement does NOT have to be a single short sentence. "
            "      A statement MAY be a compact multi-sentence block that groups tightly related information"
            "      These sentences should come from the original document material, potentially lightly edited ONLY to resolve vague references.\n"
            "    * Avoid Redundancy: You MUST merge similar or overlapping facts into single, comprehensive statements. Do NOT create multiple statements that repeat the same core information with minor variations.\n"
            "    * The number of statements should be as few as possible and **NO MORE THAN 2**, but the information should be complete and accurate."
            "\n"
            "2) Tag Generation:\n"
            "    * For each fact/statement, generate a list of tags.\n"
            "    * The number of tags per fact is flexible and should reflect the information density of the statement.\n"
            "    * Tags should cover key spans such as: entity names, years, numbers, nationalities, languages, genres, roles, object types, etc.\n"
            "    * The MAJORITY of tags SHOULD be SHORT TEXT SPANS copied VERBATIM from the statement (exact substrings). These tags are directly grounded in the surface form of the text.\n"
            "    * You MAY occasionally create additional tags that are not literal substrings of the current statement, in the following cases:\n"
            "        - You combine adjacent words into a single phrase (e.g., 'romantic comedy film').\n"
            "        - You import a surface span (exact phrase) from ANOTHER part of the document to make the meaning of the current statement explicit (for example, when resolving pronouns or phrases like 'the film', 'the band', etc.).\n"
            "    * When the subject of a statement is a pronoun or a vague description (e.g., 'the movie', 'the band', 'the hotel'), you MUST add at least one tag that names the underlying entity explicitly, using the exact surface form that appeared elsewhere in the document. This cross-statement tag may come from a previous or later sentence as long as it is clearly the same entity.\n"
            "    * If a tag is a verb, you MUST use its base (lemma) form (e.g., `play`, not `played`, `playing`, or `to play`).\n"
            # "    * Preferred behavior: whenever possible, use literal spans from the statement; only a small minority of tags should be created beyond the exact text.\n"
            "    * No Schema/Type Labels: You are FORBIDDEN from using meta-labels or ontology-like category names such as: 'Name', 'Person', 'Year', 'Date', 'City', 'Country', 'Location', 'Genre', 'Language', 'FilmTitle', etc. Tags are NOT type labels; they are content-bearing phrases.\n"
            "    * Tags should be relatively short and as fine-grained as needed. For example, adjectives and modifiers like 'Indian', 'Tamil-language', 'romantic', 'comedy' SHOULD usually be separate tags if they appear in the statement.\n"
            "\n"
            "**OUTPUT CONSTRAINTS:**\n"
            "    * Extract up to 10 facts, but prioritize QUALITY over quantity. "
            "If there are fewer than 10 truly distinct facts, output fewer.\n"
            "    * Each fact must provide unique information not covered by other facts.\n"
            "    * ABSOLUTELY NO generic references – every statement must explicitly name the specific entity.\n"
            "\n"
            "**DOCUMENT:**\n"
            "{observation}\n"
            "\n"
            "OUTPUT FORMAT:\n"
            "### Facts\n"
            "1. **Statement:** [statement]\n"
            "   **Tags:** [tag0, tag1, tag2, tag3, ...]\n"
            "2. **Statement:** [statement]\n"
            "   **Tags:** [tag0, tag1, tag2, tag3, ...]\n"
            "...\n"
        )



        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class GetProceduralPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an analytical model trained to summarize the behavior of an intelligent agent through its recorded trajectory."
        )
        user_template = (
            "You will be given an utterance of an agent.\n"
            "Your task is to analyze the utterance and derive\n"
            "- an main goal that the agent is pursuing.\n"
            "- an experiential insight — a concise reflection that summarizes the agent's behaviour.\n"
            "Follow this process when generating your response:\n"
            "1. Reasoning:\n"
            "Analyze the utterance and write down the generalizable information and patterns that would be useful as memory for future tasks.\n"
            "2. Output goal and experiential insight:\n"
            "Produce one sentence describing the general goal of the utterance, using abstract language.\n"
            "Produce one paragraph in natural language that clearly expresses the experiential insight and the reflection that summarizes the agent's behaviour.\n"
            "Output format:\n"
            "### Reasoning\n"
            "[Your reasoning process]\n"
            "### Goal\n"
            "[A sentence that conclude the goal]\n"
            "### Experiential Insight\n"
            "[The paragraph that experess the experiential insight]\n"
            "---\n"
            "Input:\n"
            "Trajectory: {trajectory}"
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]

class GetReturnPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an evaluator. Your task is to judge how well an agent pursued and completed a given goal."
        )
        user_template = (
            "You will receive:\n"
            "Goal Description: A text explaining what the agent was trying to achieve.\n"
            "Process Description: A text describing the agent's actions, decisions, and progress toward that goal.\n"  
            "Your task:\n"
            "Analyze the agent's process and determine how much of the goal was completed, considering the following aspects:\n"
            "Grading Criteria (Score 1-10):\n"
            "10: The agent fully accomplishes the goal with no significant omissions; actions are fully aligned with the goal.\n"
            "8-9: The agent completes most of the goal with only minor gaps; strong alignment but not perfect.\n"
            "6-7: Partial completion; the agent covers many key elements but leaves notable parts unfinished or poorly executed.\n"
            "4-5: Limited progress; the agent attempts the goal but completes less than half or does so in an ineffective way.\n"
            "2-3: Very little completion; actions barely connect to the goal or achieve only minimal results.\n"
            "1: No meaningful progress; actions do not contribute to achieving the goal at all.\n"
            "Important Instructions:\n"
            "Base the score only on completion level and alignment with the stated goal.\n"
            "Do not provide explanations or commentary unless requested.\n"
            "Output must follow the format below exactly.\n"
            "Output Format:\n"
            "### Score\n"
            "[number from 1 to 10]\n"
            "Input:\n"
            "---\n"
            "Goal:\n"
            "{subgoal}\n"
            "Process:\n"
            "{procedural_memory}"
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]