messages = [
    {'role': 'user', 'content': 'What the hell am I doing?', 'image':"hrefsomething"},
    {'role': 'assistant', 'content': 'Youre doing great mate'},
    {'role': 'user', 'content': 'What the hell am I doing?', 'image':"hrefsomething"},
    {'role': 'assistant', 'content': 'Youre doing great mate'},
    {'role': 'user', 'content': 'Elaborate pleej'}
]

from typing import List, Dict, Tuple, Any

def generate_history_internvl2(messages: List[str]) -> Tuple[List[Tuple], str]:
    """
    Separates the history and query from the list of messages in the current game instance.
    Compatible with InternVL2 and Nvidia NVLM models.

    Args:
        messages: A list containing user messages, system messages or assistant responses.
    
    Returns:
        A list of tuples containing the history and a user message string, passed to the model in the current game instance.

    Raises:
        ValueError: if msg['role'] is different than 'user', 'system', or 'assistant'.
    """

    history = []
    for msg in messages:
        if msg['role'] == 'system':
            continue # Skip the system message, Not passed to the model. Ref - https://huggingface.co/OpenGVLab/InternVL2-40B 
        elif msg['role'] == 'user':
            if 'image' in msg:
                user_message = f"</image>\n{msg['content']}" # Add <image> token if image is passed in this instance.
            else:
                user_message = msg['content']
        elif msg['role'] == 'assistant':
            history.append((user_message, msg['content']))
        else:
            raise ValueError(f"Invalid role: {msg['role']}. Expected 'user', 'system', or 'assistant'.")

    return history, user_message


history, usr = generate_history_internvl2(messages)


