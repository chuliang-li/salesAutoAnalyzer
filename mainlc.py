from pipelines.llm_da_pipeline import build_eda_pipeline
from db.duckdb_client import DuckDBClient

if __name__ == "__main__":

    duckdb_client = DuckDBClient()
    df = duckdb_client.get_table_list_with_sources()
    table_name = df['table_name'].iloc[0]

    schema = duckdb_client.generate_table_schema_prompt(selected_table_name=table_name)

    eda = build_eda_pipeline()

    output = eda.invoke({"table_schema": schema,"user_question":"按区域统计销售额"})

    print(output)
