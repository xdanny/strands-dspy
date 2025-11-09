"""Pre-built input and output extractors for common patterns."""

from typing import Any, Dict, List


def extract_first_user_text(messages: List[Dict]) -> Dict[str, Any]:
    """Extract the first user message text as 'question' field.

    Args:
        messages: List of Strands message dictionaries

    Returns:
        Dictionary with 'question' field containing the first user message text

    Example:
        >>> def input_extractor(messages):
        ...     return extract_first_user_text(messages)
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        return {"question": block["text"]}
    return {"question": ""}


def extract_last_assistant_text(result) -> Dict[str, Any]:
    """Extract the last assistant message text as 'answer' field.

    Args:
        result: Strands agent result object

    Returns:
        Dictionary with 'answer' field containing the last assistant message text

    Example:
        >>> def output_extractor(result):
        ...     return extract_last_assistant_text(result)
    """
    if not hasattr(result, "message"):
        return {"answer": ""}

    message = result.message
    content = message.get("content", [])

    if content and isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                return {"answer": block["text"]}

    return {"answer": ""}


def extract_all_user_messages(messages: List[Dict]) -> Dict[str, Any]:
    """Extract all user messages as 'questions' field (list).

    Args:
        messages: List of Strands message dictionaries

    Returns:
        Dictionary with 'questions' field containing list of all user message texts

    Example:
        >>> def input_extractor(messages):
        ...     return extract_all_user_messages(messages)
    """
    questions = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        questions.append(block["text"])
    return {"questions": questions}


def extract_field_from_message(
    messages: List[Dict],
    field_name: str = "question",
    role: str = "user",
    index: int = 0,
) -> Dict[str, Any]:
    """Extract a specific message into a named field.

    Args:
        messages: List of Strands message dictionaries
        field_name: Name of the field to extract to (default: "question")
        role: Role to filter by (default: "user")
        index: Which message of that role to extract (default: 0 = first)

    Returns:
        Dictionary with field_name containing the extracted text

    Example:
        >>> def input_extractor(messages):
        ...     # Get second user message as "followup"
        ...     return extract_field_from_message(messages, "followup", "user", 1)
    """
    matching_messages = []
    for msg in messages:
        if msg.get("role") == role:
            content = msg.get("content", [])
            if content and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        matching_messages.append(block["text"])
                        break  # Only first text block per message

    if index < len(matching_messages):
        return {field_name: matching_messages[index]}
    return {field_name: ""}


def combine_extractors(*extractors) -> callable:
    """Combine multiple extractors into one by merging their outputs.

    Args:
        *extractors: Variable number of extractor functions

    Returns:
        A new extractor function that merges all outputs

    Example:
        >>> def input_extractor(messages):
        ...     return combine_extractors(
        ...         extract_first_user_text,
        ...         lambda m: extract_field_from_message(m, "context", "system")
        ...     )(messages)
    """
    def combined_extractor(data):
        result = {}
        for extractor in extractors:
            result.update(extractor(data))
        return result
    return combined_extractor
