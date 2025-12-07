# Refactored llm_qwen.py tailored to dynamic prompt loading for xls and sql tasks

import json
from pathlib import Path
from langchain_ollama import ChatOllama

class QwenOllamaClient:
    """
    通用 Qwen (Ollama) 模型调用接口。

    支持：
    1. 动态加载不同场景的 prompt（xls_prompts.txt / sql_prompts.txt 等）。
    2. 仍然使用 Ollama 本地 qwen 模型。
    3. xls 模式下会读取 prompts/xls_prompts.txt。
    4. sql 模式下会读取 prompts/sql_prompts.txt。
    5. 未来可扩展更多 prompt 类型。
    """

    def __init__(self,
                 invoke_type: str = "freestyle",
                 model_name: str = "qwen3:4b",
                 temperature: float = 0.0,
                 prompt_file: str = None):

        self.model_name = model_name
        self.temperature = temperature
        root = Path(__file__).resolve().parents[1]


        if invoke_type == "freestyle":        
            prompts_dir = root / "prompts"
            self.prompt_file_path = (
            Path(prompt_file) if prompt_file else (prompts_dir / "xls_prompts.txt")
        )

        elif invoke_type == "langchain":
            prompts_dir = root / "promptsLC"
            self.prompt_file_path = (
            Path(prompt_file) if prompt_file else (prompts_dir / "sql_prompts.txt")
        )

        
        # 默认 prompt 文件

        self.base_prompt = self._read_prompt(self.prompt_file_path)
        self.prompts_dir = prompts_dir

        # 初始化本地 Qwen 模型（Ollama）
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)

    # -------------------------------------------
    # 读取 prompt 文件
    # -------------------------------------------
    def _read_prompt(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {path}")
        return path.read_text(encoding="utf-8")

    # -------------------------------------------
    # 动态加载 prompts/xls_prompts.txt
    # -------------------------------------------
    def use_xls_prompt(self):
        path = self.prompts_dir / "xls_prompts.txt"
        self.base_prompt = self._read_prompt(path)
        self.prompt_file_path = path

    # -------------------------------------------
    # 动态加载 prompts/sql_prompts.txt
    # -------------------------------------------
    def use_sql_prompt(self):
        path = self.prompts_dir / "sql_prompts.txt"
        self.base_prompt = self._read_prompt(path)
        self.prompt_file_path = path

    # -------------------------------------------
    # 核心：使用 prompt + kwargs 调 LLM
    # -------------------------------------------
    def run_prompt(self, **kwargs) -> str:
        content = self.base_prompt
        for k, v in kwargs.items():
            content = content.replace(f"{{{k}}}", str(v))

        result = self.llm.invoke(content)
        return result.content


if __name__ == "__main__":
    client = QwenOllamaClient()

    client.use_sql_prompt()
    print(client.run_prompt(user_question="示例问题：今天涨幅最高的 ETF？"))
