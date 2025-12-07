from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableLambda

from .loader import load_prompts_from_dir
import json
import re
import pathlib as Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client.llm_qwen import QwenOllamaClient
from db.duckdb_client import DuckDBClient

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re

def execute_plotly_code(chain_output: dict):
    """
    安全执行 LLM 生成的 Plotly 代码。
    
    输入: 包含 'plotly_code' 字符串和 'df' Pandas DataFrame 的字典
    输出: 包含 'fig' Plotly Figure 对象的字典，或 None
    """
    code_str = chain_output["plotly_code"]
    df = chain_output["df"]
    
    # 移除markdown代码块标记（如果LLM加了 ```python ... ```）
    code_str = re.sub(r'```python|```', '', code_str).strip()
    
    # 准备执行环境
    local_vars = {
        'df': df, 
        'px': px, 
        'go': go, 
        'fig': None
    }
    
    try:
        # 使用 exec 安全地执行代码
        exec(code_str, {"pd": pd, "px": px, "go": go}, local_vars)
        
        # 检查是否成功生成了 Plotly Figure 对象
        if isinstance(local_vars['fig'], go.Figure):
            return {"fig": local_vars['fig']}
        else:
            return {"fig": None, "error": "绘图代码未生成有效的 Plotly Figure 对象。"}
            
    except Exception as e:
        return {"fig": None, "error": f"执行绘图代码时发生错误: {e}"}

# 将此函数添加到 LangChain 链的最后

def execute_sql_query(chain_output: str):
    try:
            # 1. 将字符串解析为 Python 字典
            # data_dict = json.loads(parsed_sql)
            data_dict = chain_output
            # 2. 检查并提取 'sql_parsed' 字段
            if 'sql_parsed' in data_dict:
                # 'sql_parsed' 的值是一个 JSON 字符串（表示一个列表）
                sql_parsed_string = data_dict['sql_parsed']
                
                # 3. 将 'sql_parsed' 字符串解析为 Python 列表
                sql_list = json.loads(sql_parsed_string)
                
                # 4. 检查列表是否非空，并提取第一个元素中的 'sql' 键的值
                if sql_list and isinstance(sql_list, list) and 'sql' in sql_list[0]:
                    sql = sql_list[0]['sql']
                    db_client = DuckDBClient()
                    query_result = db_client.query(sql)
                    data_dict['df'] = query_result
                    return data_dict
                    
    except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return ""
    except (KeyError, IndexError) as e:
            print(f"键或索引错误: {e}")
            return ""
        
    return ""

def generate_plotly_code(message:str):    
    pass
    return message

def build_eda_pipeline():
    # 1. load prompt templates
    prompts = load_prompts_from_dir("promptsLC")

    # 2. local ollama model
    llm_client  = QwenOllamaClient(invoke_type="langchain")
    llm = llm_client.llm

    rag_chain = (
        # 第1步：根据用户问题生成 sql
        ({
            "my_question": RunnablePassthrough(),
            "sql_parsed":prompts["sql_prompts"] | llm | StrOutputParser()
            
        }) 

        # # 第2步：使用sql查询数据库返回pandas结果集
        | RunnableLambda(execute_sql_query)

        # 第 3 步：LLM 生成 Plotly 代码
        | 
        {
            # 构造 LLM Prompt 的输入
            "plotly_code": 
                {
                "user_question": lambda x: x["my_question"]["user_question"], 
                "df_head": lambda x: x["df"].head(3).to_markdown() 
                } 
                | prompts["plotly_prompts"] | llm | StrOutputParser(),
            
            # 保留 user_question 到下一个步骤
            "user_question":lambda x: x["my_question"]["user_question"],
            # 保留 DataFrame 供执行步骤使用
            "df": lambda x: x["df"] 
        }  

        # 最后一步：执行代码并生成 Figure 对象
        | RunnableLambda(execute_plotly_code)  

    )

    # 5. return parallel runnable pipeline
    return RunnableSequence(rag_chain)
    # return RunnableParallel(chains)
