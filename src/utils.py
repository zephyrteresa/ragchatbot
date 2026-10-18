import os
from typing import Any, Dict
from pathlib import Path
import yaml
import logging

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

    output_constraints = prompt.get("output_constraints")
    if output_constraints:
        for constraint in output_constraints:
            prompt_str = prompt_str + "\n\n" + constraint  

    tone = prompt.get("style_or_tone")
    if tone:
        for tone in tone:
            prompt_str = prompt_str + "\n\n" + tone      

    if goal := prompt.get("goal"):
        prompt_str = prompt_str + "\n\n Your goal is the ff: " + goal    
    
    return prompt_str    

def get_logger(log_name: str = __name__):

    try: 
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Create file handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "app.log"
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    except Exception as e:
        print(f"Error setting up logger: {str(e)}")

    return logger

def get_config_value(key:str):
    default_config = load_yaml(Path("config") / "config.yaml")

    return default_config.get(key)

def get_prompt(prompt_key: str):
    config_dir = Path("config").resolve()
    prompt_config_file = os.path.join(config_dir, "prompt_config.yaml")
    prompt_config = load_yaml(prompt_config_file) 
    prompt = build_prompt(prompt_config, prompt_key)

    return prompt;