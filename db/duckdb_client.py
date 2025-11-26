import duckdb
import pandas as pd
from datetime import date
import os

class DuckDBClient:
    def __init__(self, db_path='salesRAG.duck'):
        # 确保数据库路径是正确的，即使程序是从其他目录运行的
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_db_path = os.path.join(script_dir, db_path)
        print(f"Connecting to database at: {full_db_path}") # 调试信息
        # 使用 check_same_thread=False 解决 Streamlit 多线程环境下的问题
        self.con = duckdb.connect(full_db_path, read_only=False)

    def query(self, sql: str):
        try:
            # 打印查询语句，便于调试
            print(f"Executing SQL: {sql}")
            return self.con.execute(sql).fetchdf()
        except Exception as e:
            print(f"Query Error: {e}") # 打印错误信息
            return pd.DataFrame([{"error": str(e)}])

    def init_table_meta(self):
            """
            初始化或更新 table_meta 表。

            table_meta 表结构：(table_name TEXT, column_en TEXT, column_cn TEXT, column_type TEXT)
            
            Args:
                table_meta_data (list): 包含表元数据的列表，每个元素是一个字典，
                                        键为 table_name, column_en, column_cn, column_type。
            """
            
            # 1. 创建 table_meta 表（如果不存在）
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS table_meta (
                    source      VARCHAR,
                    table_name  VARCHAR,
                    column_en   VARCHAR,
                    column_cn   VARCHAR,
                    column_type VARCHAR,
                    PRIMARY KEY (source, table_name, column_cn)
                );            
                
                """
            try:
                print("Executing SQL: Creating table_meta if not exists...")
                self.con.execute(create_table_sql)
            except Exception as e:
                print(f"Error creating table_meta: {e}")
                pass

    def get_table_list_with_sources(self) -> pd.DataFrame:
        """
        从 table_meta 表中获取所有独特的表名及其描述 (source)。
        
        Returns:
            pd.DataFrame: 包含 table_name 和 source 的 DataFrame。
        """
        sql = "SELECT DISTINCT table_name, source FROM main.table_meta ORDER BY table_name;"
        return self.query(sql)

    def generate_table_schema_prompt(self, selected_table_name: str) -> str:
        """
        从 table_meta 表中提取指定表的结构信息，生成供 LLM 使用的数据库表结构说明 prompt。

        Args:
            selected_table_name (str): 需要生成 schema 的表名。

        Returns:
            str: 包含指定表结构的字符串 prompt。
        """
        # 1. 查询 table_meta 表以获取指定表的元数据
        query_sql = f"SELECT column_en, column_cn, column_type FROM main.table_meta WHERE table_name = '{selected_table_name}' ORDER BY column_en"
        meta_df = self.query(query_sql)

        if meta_df.empty or "error" in meta_df.columns:
            print(f"Warning: table_meta for '{selected_table_name}' is empty or query failed. Cannot generate schema prompt.")
            return f"No table schema information available for table '{selected_table_name}'."

        # 2. 构建表结构
        prompt_parts = [f"数据库包含以下表结构：\n{selected_table_name} ("]

        column_definitions = []
        # 假设所有列都存在且是字符串类型，确保它们能被正确访问
        for index, row in meta_df.iterrows():
            column_en = row.get('column_en', 'UNKNOWN')
            column_cn = row.get('column_cn', '未知字段')
            column_type = row.get('column_type', 'VARCHAR')

            # 格式: column_en column_type -- column_cn
            definition = f"    {column_en} {column_type} -- {column_cn}"
            column_definitions.append(definition)

        # 将列定义用逗号连接
        table_schema = ",\n".join(column_definitions)
        prompt_parts.append(table_schema)
        prompt_parts.append(");")
        
        return "\n".join(prompt_parts)
    
    def clear_db(self, keep_meta_table: bool = True) -> None:
        """
        清空数据库中所有业务数据表（即 table_meta 中登记的表）。
        
        参数:
            keep_meta_table: 是否保留 table_meta 表本身，默认 True（推荐保留）
        
        返回:
            None
        """
        # 1. 先获取所有登记在 table_meta 中的表名
        table_df = self.get_table_list_with_sources()
        
        if table_df.empty:
            print("No tables registered in table_meta, nothing to drop.")
            return
        
        dropped_count = 0
        for table_name in table_df['table_name'].unique():
            # 可选：如果你想连 table_meta 里的记录也一起删，可以在这里 DELETE FROM table_meta ...
            drop_sql = f"DROP TABLE IF EXISTS main.{table_name};"
            try:
                print(f"Dropping table: {table_name}")
                self.con.execute(drop_sql)
                dropped_count += 1
            except Exception as e:
                print(f"Failed to drop table {table_name}: {e}")
        
        # 如果你希望同时清除 table_meta 中的元数据记录（彻底清理），可以取消下面注释
        # if not keep_meta_table:
        #     self.con.execute("DELETE FROM main.table_meta;")
        
        print(f"Clear database completed. {dropped_count} table(s) dropped.")    

# --- 以下是新增的调试块 ---
if __name__ == "__main__":
    # 1. 实例化客户端
    client = DuckDBClient()
    client.init_table_meta()
    # 打印表列表
    # print("\n--- Test Table List ---")
    # print(client.get_table_list_with_sources())
    # print(client.generate_table_schema_prompt('order_details'))
    
    # 打印单个表结构
    # print("\n--- Test Schema Prompt ---")
    # print(client.generate_table_schema_prompt('order_details')) # 假设 'order_details' 是一个存在的表名

    client.con.close()