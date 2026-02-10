from __future__ import annotations

from typing import List, Mapping

from prompt_base import PromptBase, ChatMessage


class DefaultEpisodicPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an assistant to extract information from memory and answer the user's question."
        )
        # user_template = (
        #     'I will give you several memory of history chats between you and a user. Extract all the useful information relevant to the question step by step: First analyze the relationship between each memory and the question, then obtain the useful information.\n'
        #     'Output format:\n'
        #     '---\n'
        #     '### Reasoning\n'
        #     '(process of extract information)\n'
        #     '### Information\n'
        #     '(The useful information you extract)\n'
        #     '---\n'
        #     'Input:\n'
        #     'History chats: {episodic_memory_semantic}\n'
        #     'Current Date: {time}\n'
        #     'Question: {observation}'
        # )
        user_template = (
            'I will give you several memory items. Extract all the useful information relevant to the question step by step: First analyze the relationship between each memory and the question, then obtain the useful information.\n'
            # 'If no useful information found, just return "null".\n'
            'If no useful information is found, just concatenate all the facts as the output, like: Fact 0: <fact> \\nFact 1: <fact> \\n...)\n'
            'Output format:\n'
            '---\n'
            '### Reasoning\n'
            '(process of extract information, if the token bugdet is limited, you can return a very brief reasoning process or even skip this process with only a placeholder <SKIPPED>)\n'
            '### Information\n'
            '(The useful information you extract)\n'
            '---\n'
            'Input:\n'
            'Possible Relevant Items: {episodic_memory}\n'
            'Question: {observation}'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]

class DefaultSemanticPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an assistant to extract useful information from fact and answer the user's question."
        )
        user_template = (
            'I will give you several retrieved facts. Extract all the useful information relevant to the question. \n'
            'In the output reasoning information, use the original wording from the retrieved facts as much as possible, and do not replace it with synonyms or near-synonyms.'
            'If no useful information found, just return "null".\n'
            # 'If no useful information is found, just concatenate all the facts as the output, like: Fact 0: <fact> \\nFact 1: <fact> \\n...)\n'
            'Output format:\n'
            '---\n'
            '### Reasoning\n'
            '(process of extract information)\n'
            '### Information\n'
            '(The useful information you extract)\n'
            '---\n'
            'Input:\n'
            'Facts: {semantic_memory}\n'
            'Current Date: {time}\n'
            'Question: {observation}'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]


class DefaultProceduralPrompt(PromptBase):
    def build_messages(self, variables: Mapping[str, object]) -> List[ChatMessage]:
        system_template = (
            "You are an assistant helping an intelligent agent decide which information from memory should be used to answer the user's question."
        )
        user_template = (
            'Following information will be provided:\n'
            'Question: The question the user is asking.\n'
            'Information: Several pieces of information that may be relevant to the question.\n'
            'Your task:\n'
            "1. Carefully read the user's question.\n"
            '2. Analyze each piece of retrieved information and determine how relevant and useful it is for answering the question.\n'
            '3. Based on your analysis, integrate all the useful information into a single coherent piece of content that help the agent to answer the question.\n'
            "Based on your analysis, integrate all the useful information into a single coherent piece of content that help the agent to answer the question. When you think information is insufficient or contradictory, generate the most possible information. The integrated content should be concise, accurate, and relevant to the user's question"
            'Output format:\n'
            '---\n'
            '### Reasoning\n'
            '(Your reasoning for analiysis of given question and information.)\n'
            '### Final Information\n'
            '(The synthesized information that should be provided to the agent.)\n'
            '---\n'
            'Input:\n'
            'Question: {observation}\n'
            'Information: {procedural_memory}'
        )
        return [
            ChatMessage("system", self.format_text(system_template, variables)),
            ChatMessage("user", self.format_text(user_template, variables)),
        ]