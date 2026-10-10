"""Separate the final answer from a Qwen thinking token stream."""
def split_output(ids, thinking, end_token):
    if not thinking:
        return ids, [], None
    if end_token not in ids:
        return [], ids, 'thinking_not_completed'
    index = len(ids) - ids[::-1].index(end_token)
    return ids[index:], ids[:index], None
