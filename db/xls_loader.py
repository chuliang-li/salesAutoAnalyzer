# db/xls_loader.py
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client.llm_qwen import QwenOllamaClient
from db.duckdb_client import DuckDBClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input_xls"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


# -----------------------------
# 调试辅助函数：保存llm output到当前目录下
# -----------------------------

import json
import os
from typing import Dict, Any

def save_llm_json(file_name: str, data: Dict[str, Any]) -> bool:
    """
    将LLM的Python字典输出保存为当前目录下的JSON文件。

    Args:
        file_name (str): 要保存的文件名 (例如: 'output.json')。
        data (Dict[str, Any]): LLM返回的Python字典数据。

    Returns:
        bool: 如果保存成功返回 True，否则返回 False。
    """
    try:
        # 获取文件保存的完整路径，确保是当前目录
        # 虽然只使用文件名默认就是当前目录，但这样更明确
        save_path = os.path.join(os.getcwd(), file_name)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # 使用 json.dump() 将 Python 对象写入文件
            # indent=4 使JSON格式化，ensure_ascii=False 支持中文
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"🎉 成功保存JSON数据到: {save_path}")
        return True
        
    except IOError as e:
        print(f"❌ 写入文件时发生错误 ({file_name}): {e}")
        return False
    except TypeError as e:
        print(f"❌ 数据类型错误，请确保传入的是有效的Python字典: {e}")
        return False


# -----------------------------
# 从 LLM 返回中提取 JSON 的辅助函数
# -----------------------------
def extract_json_block(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("LLM 返回中没有 JSON 段")

    json_str = text[start:end + 1]
    return json.loads(json_str)


# -----------------------------
# 处理单个 Excel 文件
# -----------------------------
def process_excel_file(excel_path: Path, duck: DuckDBClient, llm: QwenOllamaClient):
    print(f"\n=== 处理 Excel 文件: {excel_path.name} ===")
    excel_file=excel_path.as_posix()
    # 读取前 5 行
    df_preview = pd.read_excel(excel_path, nrows=5)
    preview_json = json.dumps(df_preview.where(pd.notnull(df_preview), None).to_dict(orient="records"), ensure_ascii=False)

    # 切换 LLM 到 xls prompts
    llm.use_xls_prompt()

    # 调用 LLM
    llm_output = llm.run_prompt(excel_preview=preview_json)

    # save_llm_json("debug.json",llm_output)

    # 解析 JSON
    meta = extract_json_block(llm_output)

    table_name = meta["table_name"]
    create_sql = meta["create_sql"]
    columns = meta["columns"]                # [{cn,en,type}]
    table_meta_inserts = meta["table_meta_inserts"]  # ["INSERT INTO table_meta ..."]

    print(f"LLM 生成表名: {table_name}")

    # -----------------------------
    # 执行 CREATE TABLE
    # -----------------------------
    print("\n执行建表 SQL：")
    duck.init_table_meta()
    duck.clear_db()
    print(create_sql)
    duck.query(create_sql)

    # -----------------------------
    # 写入 table_meta
    # -----------------------------
    print("\n写入 table_meta...")
    for sql in table_meta_inserts:
        start_index = sql.find('(')
        new_sql = f"INSERT INTO table_meta VALUES ('{excel_file}',{sql[start_index+1:]}"
        duck.query(new_sql)

    # -----------------------------
    # 加载 Excel 全量数据
    # -----------------------------
    print("\n加载 Excel 数据到表...")

    df = pd.read_excel(excel_path)

    # 生成中英文映射
    cn_to_en = {col["cn"]: col["en"] for col in columns}

    # 用英文列名重命名 df
    df.rename(columns=cn_to_en, inplace=True)

    # 用 DuckDB 的 COPY 或 INSERT
    tmp_parquet = excel_path.with_suffix(".parquet")
    df.to_parquet(tmp_parquet, index=False)

    load_sql = f"""
        COPY {table_name}
        FROM '{tmp_parquet}'
        (FORMAT 'parquet');
    """

    duck.query(load_sql)

    print(f"完成加载: {excel_path.name} → {table_name}")

    return {
        "file": excel_path.name,
        "table": table_name,
        "columns": columns
    }


# -----------------------------
# 主程序：处理 input_xls 下所有 Excel 文件
# -----------------------------
def main():
    print("=== Excel → DuckDB Loader 启动 ===")

    llm = QwenOllamaClient()     # 统一 LLM 客户端
    duck = DuckDBClient()        # 使用你已有的 DuckDB 客户端

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"input_xls 目录不存在: {INPUT_DIR}")

    files = list(INPUT_DIR.glob("*.xls*"))

    if not files:
        print("input_xls 下没有 Excel 文件")
        return

    results = []
    for f in files:
        try:
            info = process_excel_file(f, duck, llm)
            results.append(info)
        except Exception as e:
            print(f"处理失败 {f.name}: {e}")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['file']} → {r['table']}")


if __name__ == "__main__":
    main()
