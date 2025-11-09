"""Pre-built success criteria for common patterns."""

from typing import Any, Callable


def always_success(result: Any) -> tuple[bool, float]:
    """Always mark as successful - collect all examples.

    Useful when you want to collect all interactions regardless of outcome.

    Args:
        result: Agent result (unused)

    Returns:
        Tuple of (True, 1.0)

    Example:
        >>> collector = TrainingCollector(
        ...     success_criteria=always_success,
        ...     input_extractor=extract_first_user_text,
        ...     output_extractor=extract_last_assistant_text
        ... )
    """
    return True, 1.0


def end_turn_success(result: Any) -> tuple[bool, float]:
    """Mark successful if agent ended its turn naturally.

    Checks if stop_reason is "end_turn" (vs "max_tokens" or "error").
    Useful for filtering out incomplete or truncated responses.

    Args:
        result: Agent result with stop_reason attribute

    Returns:
        Tuple of (is_end_turn, 1.0 if end_turn else 0.0)

    Example:
        >>> collector = TrainingCollector(
        ...     success_criteria=end_turn_success,
        ...     input_extractor=extract_first_user_text,
        ...     output_extractor=extract_last_assistant_text
        ... )
    """
    if not hasattr(result, "stop_reason"):
        return False, 0.0

    is_success = result.stop_reason == "end_turn"
    return is_success, 1.0 if is_success else 0.0


def no_error_success(result: Any) -> tuple[bool, float]:
    """Mark successful if no errors occurred during execution.

    Checks for error attributes or exceptions in the result.
    Useful for filtering out failed executions.

    Args:
        result: Agent result

    Returns:
        Tuple of (no_error, 1.0 if no error else 0.0)

    Example:
        >>> collector = TrainingCollector(
        ...     success_criteria=no_error_success,
        ...     input_extractor=extract_first_user_text,
        ...     output_extractor=extract_last_assistant_text
        ... )
    """
    # Check common error indicators
    if hasattr(result, "error") and result.error:
        return False, 0.0

    if hasattr(result, "exception") and result.exception:
        return False, 0.0

    if hasattr(result, "stop_reason") and result.stop_reason == "error":
        return False, 0.0

    return True, 1.0


def min_length_success(min_chars: int = 50) -> Callable[[Any], tuple[bool, float]]:
    """Create success criteria requiring minimum answer length.

    Returns a function that checks if the assistant's response meets
    a minimum character count. Useful for filtering out too-short answers.

    Args:
        min_chars: Minimum number of characters required (default: 50)

    Returns:
        Success criteria function

    Example:
        >>> collector = TrainingCollector(
        ...     success_criteria=min_length_success(min_chars=100),
        ...     input_extractor=extract_first_user_text,
        ...     output_extractor=extract_last_assistant_text
        ... )
    """
    def check_length(result: Any) -> tuple[bool, float]:
        if not hasattr(result, "message"):
            return False, 0.0

        message = result.message
        content = message.get("content", [])

        # Extract text from content blocks
        total_length = 0
        if content and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total_length += len(block["text"])

        is_success = total_length >= min_chars
        return is_success, 1.0 if is_success else 0.0

    return check_length


def combine_criteria(*criteria: Callable[[Any], tuple[bool, float]]) -> Callable[[Any], tuple[bool, float]]:
    """Combine multiple success criteria with AND logic.

    All criteria must pass for overall success. Score is the minimum
    of all individual scores.

    Args:
        *criteria: Variable number of success criteria functions

    Returns:
        Combined success criteria function

    Example:
        >>> collector = TrainingCollector(
        ...     success_criteria=combine_criteria(
        ...         end_turn_success,
        ...         no_error_success,
        ...         min_length_success(min_chars=50)
        ...     ),
        ...     input_extractor=extract_first_user_text,
        ...     output_extractor=extract_last_assistant_text
        ... )
    """
    def combined(result: Any) -> tuple[bool, float]:
        all_success = True
        min_score = 1.0

        for criterion in criteria:
            is_success, score = criterion(result)
            if not is_success:
                all_success = False
            min_score = min(min_score, score)

        return all_success, min_score if all_success else 0.0

    return combined
