# utils.py - This is the class for helper functions
from typing import List, Dict
from openai import OpenAI

def build_messages_from_template(user_prompt: str) -> List[Dict]:
    """
    Create a chat-style messages list suitable for chat completions.
    This keeps a consistent system instruction for the fine-tuned assistant.
    """
    return [
        {"role": "system", "content": "You are a helpful car sales assistant."},
        {"role": "user", "content": user_prompt}
    ]

def call_finetuned_model(client: OpenAI, ft_model: str, messages: List[Dict], temperature: float=0.0, max_tokens:int=300) -> str:
    """
    Call the fine-tuned model (must pass the exact ft model id).
    Example ft_model value: "gpt-3.5-turbo-0125:car_dealer_sales"
    """
    resp = client.chat.completions.create(
        model=ft_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content

def call_base_model(client: OpenAI, messages: List[Dict], temperature: float=0.0, max_tokens:int=300) -> str:
    """
    Call a base model (fallback). call gpt-3.5-turbo-0125 for parity.
    """
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content
