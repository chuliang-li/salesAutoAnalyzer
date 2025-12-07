import os
import sys
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client.llm_qwen import QwenOllamaClient
from db.duckdb_client import DuckDBClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"



def load_prompt(path: str) -> ChatPromptTemplate:
    """加载 prompts.txt 文件为ChatPromptTemplate (Runnable)."""
    text = open(path, "r", encoding="utf-8").read()
    return ChatPromptTemplate.from_template(text)


def load_prompts_from_dir(directory: str):
    """Return dict {filename: PromptTemplate} sorted by filename."""
    prompts = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            key = filename.replace(".txt", "")
            prompts[key] = load_prompt(os.path.join(directory, filename))
    return prompts

if __name__ == "__main__":
    
    step_prompts =  load_prompts_from_dir(PROMPTS_DIR)
    
    