import os
from typing import Any, Dict
from pathlib import Path

import yaml


def load_yaml(file_path: str)-> Dict[str, Any]:
   
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Config file not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        cfg_data = yaml.safe_load(file)

    if cfg_data is None:
        return {}    

    return cfg_data  

def build_prompt(prompt_config: Dict[str, Any], prompt_type: str):

    prompt_str = ""
    prompt = prompt_config.get(prompt_type)

    if role := prompt.get("role"):
        prompt_str = prompt_str + role

    if instruction := prompt.get("instruction"):
        prompt_str = prompt_str + "\n\n" + instruction

    for constraint in prompt.get("output_constraints"):
        prompt_str = prompt_str + "\n\n" + constraint  

    for tone in prompt.get("style_or_tone"):
        prompt_str = prompt_str + "\n\n" + tone      

    if goal := prompt.get("goal"):
        prompt_str = prompt_str + "\n\n Your goal is the ff: " + goal    
    
    return prompt_str    
        