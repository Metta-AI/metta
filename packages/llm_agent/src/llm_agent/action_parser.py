"""Action parsing for LLM responses."""

import json
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mettagrid.simulator import Action


def _find_action_by_name(name: str, actions: list["Action"]) -> "Action | None":
    """Find action matching the given name (case-insensitive)."""
    name = name.lower()
    for action in actions:
        if action.name.lower() == name:
            return action
    return None


def _find_action_in_words(words: list[str], actions: list["Action"]) -> "Action | None":
    """Find action name in list of words, checking from end to start."""
    for word in reversed(words):
        clean_word = word.strip(".,!?;:")
        if action := _find_action_by_name(clean_word, actions):
            return action
    return None


def _find_action_as_substring(text: str, actions: list["Action"]) -> "Action | None":
    """Find action name appearing as substring in text."""
    text = text.lower()
    for action in actions:
        if action.name.lower() in text:
            return action
    return None


def _try_parse_json(text: str, actions: list["Action"]) -> tuple["Action | None", str]:
    """Try parsing response as JSON with action and reasoning fields."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "action" in parsed:
            action = _find_action_by_name(parsed["action"].strip(), actions)
            reasoning = parsed.get("reasoning", "")
            return action, reasoning
    except json.JSONDecodeError:
        pass
    return None, ""


def parse_action(response_text: str, available_actions: list["Action"]) -> tuple["Action", str]:
    """Parse LLM response and return valid Action and reasoning."""
    text = response_text.strip()

    action, reasoning = _try_parse_json(text, available_actions)
    if action:
        return action, reasoning

    clean_text = text.strip("\"'").lower()

    if action := _find_action_by_name(clean_text, available_actions):
        return action, reasoning

    words = clean_text.split()
    if len(words) > 1:
        if action := _find_action_in_words(words, available_actions):
            return action, reasoning

    if action := _find_action_as_substring(clean_text, available_actions):
        return action, reasoning

    return random.choice(available_actions), reasoning
