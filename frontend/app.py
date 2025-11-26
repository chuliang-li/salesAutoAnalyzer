# frontend/app.py
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# 路径配置：确保可以导入 db/duckdb_client 和 llm_client/llm_qwen
# ----------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent 

# 将 'db' 和 'llm_client' 目录添加到 sys.path
# 注意：这里使用 append 而非 insert(0) 确保路径被正确加载
if str(root_dir / "db") not in sys.path:
    sys.path.append(str(root_dir / "db"))
if str(root_dir / "llm_client") not in sys.path:
    sys.path.append(str(root_dir / "llm_client"))

# 现在可以安全地导入模块了
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client.llm_qwen import QwenOllamaClient
from db.duckdb_client import DuckDBClient


# ----------------------------------------------------------------------
# 辅助函数 1: 解析 LLM 输出的 JSON (保持不变)
# ----------------------------------------------------------------------

def parse_llm_json_output(json_str: str) -> list:
    """尝试解析 LLM 输出的 JSON 字符串，返回包含 'sql' 和 'description' 的列表。"""
    try:
        # 清理代码块标记
        json_str = json_str.strip()
        if json_str.startswith("```"):
            start_index = json_str.find('\n')
            if start_index != -1:
                json_str = json_str[start_index+1:].strip()
            else:
                 json_str = json_str[3:].strip()
            
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()

        data = json.loads(json_str)
        
        if isinstance(data, list) and all(isinstance(item, dict) and 'sql' in item and 'description' in item for item in data):
            return data
        elif isinstance(data, dict) and 'sql_list' in data and isinstance(data['sql_list'], list):
            return data['sql_list']
        else:
            return []

    except json.JSONDecodeError as e:
        # st.error(f"解析 LLM 输出的 JSON 失败。错误: {e}")
        return []
    except Exception as e:
        # st.error(f"处理 LLM 输出时发生未知错误。错误: {e}")
        return []

# ----------------------------------------------------------------------
# 辅助函数 2: Plotly 绘图 (修改为中文标签)
# ----------------------------------------------------------------------

def plot_data(df: pd.DataFrame, title: str, column_mapping: dict = None):
    """
    根据 DataFrame 的列类型尝试进行绘图，并使用中文标签。
    Args:
        df (pd.DataFrame): 待绘图数据。
        title (str): 图表标题 (中文)。
        column_mapping (dict): 英文列名到中文描述的映射，用于轴标签。
    """
    if df.empty or "error" in df.columns:
        st.warning(f"无法绘图：查询结果为空或包含错误。")
        st.dataframe(df)
        return

    df = df.copy() 
    
    # 构建中文列名映射，用于绘图时显示中文标签
    # Plotly.express 默认使用 df.columns 作为标签，通过 names 参数进行映射
    if column_mapping:
        # 为 Plotly 创建一个英文到中文的映射
        name_mapping = {col: column_mapping.get(col, col) for col in df.columns}
    else:
        name_mapping = {col: col for col in df.columns}
    
    # 尝试日期转换
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if 'date' in first_col.lower() or 'day' in first_col.lower():
            try:
                df[first_col] = pd.to_datetime(df[first_col], errors='ignore')
            except Exception:
                pass 

    # 1. 单个聚合值 -> Indicator
    if len(df.columns) == 1 and pd.api.types.is_numeric_dtype(df.dtypes[0]):
        value = df.iloc[0, 0]
        fig = go.Figure(go.Indicator(
            mode = "number",
            value = value,
            title = {"text": f"总览: {title}"}
        ))
        st.plotly_chart(fig, use_container_width=True)
        return

    # 2. 两列 (维度+度量) -> Line 或 Bar
    elif len(df.columns) == 2:
        x_col, y_col = df.columns
        
        # 使用映射后的中文名作为轴标签
        labels = {
            x_col: name_mapping[x_col],
            y_col: name_mapping[y_col]
        }
        
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            chart_type = 'line'
        elif pd.api.types.is_numeric_dtype(df[y_col]):
            chart_type = 'bar'
        else:
            chart_type = 'bar' 

        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title, labels=labels)
        else:
            if pd.api.types.is_numeric_dtype(df[y_col]):
                 df = df.sort_values(by=y_col, ascending=False)
            fig = px.bar(df, x=x_col, y=y_col, title=title, labels=labels)
        
        st.plotly_chart(fig, use_container_width=True)
        return

    # 3. 默认：显示表格
    st.warning("自动绘图逻辑无法识别最佳图表类型，显示数据表格。")
    st.dataframe(df)


# ----------------------------------------------------------------------
# Streamlit 主程序
# ----------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RAG-SQL 自动分析工具", 
        layout="wide"
    )
    
    st.title("💡 LLM 自动数据分析师")
    st.caption("基于 DuckDB 获取表结构，LLM 自动生成 SQL，并使用 Plotly 绘图。")

    # 1. 初始化客户端
    try:
        duckdb_client = DuckDBClient()
        llm_client = QwenOllamaClient(model_name="qwen3:4b", temperature=0.0) 
        llm_client.use_sql_prompt() 
        st.sidebar.success("客户端初始化成功。")
    except Exception as e:
        st.error(f"初始化客户端失败：请检查 Ollama 服务、Qwen 模型和数据库路径配置。错误: {e}")
        st.stop()


    # 2. 获取表列表并选择 (新增功能)
    st.sidebar.header("数据表选择")
    try:
        table_list_df = duckdb_client.get_table_list_with_sources()
        if table_list_df.empty or "error" in table_list_df.columns:
            st.error("无法获取数据库表列表，请检查 table_meta 表。")
            st.stop()
        
        # 构造用于选择器的选项列表： "表名 (来源描述)"
        options = [f"{row['table_name']} ({row['source']})" for index, row in table_list_df.iterrows()]
        
        selected_option = st.sidebar.selectbox(
            "请选择需要分析的数据表:",
            options=options,
            index=0
        )
        
        # 提取选中的表名
        selected_table_name = selected_option.split(' ')[0]
        
    except Exception as e:
        st.error(f"加载表列表失败: {e}")
        selected_table_name = None
        st.stop()


    # 3. 获取选中表的表结构
    st.sidebar.subheader("选中表结构提示")
    with st.spinner(f"正在获取表 '{selected_table_name}' 的结构..."):
        # 使用修正后的参数名 selected_table_name
        table_schema_prompt = duckdb_client.generate_table_schema_prompt(selected_table_name)
    
    st.sidebar.code(table_schema_prompt, language="sql")
    
    # 4. 调用 LLM 生成 SQL 列表 
    st.header(f"🤖 LLM 自动生成 '{selected_table_name}' 的数据分析图")
    
    if st.button(f"一键生成 数据分析图", type="primary"):
        # 清理旧数据
        if 'llm_raw_output' in st.session_state: del st.session_state['llm_raw_output']
        if 'sql_list' in st.session_state: del st.session_state['sql_list']
        
        with st.spinner("LLM (Qwen) 正在思考并生成数据分析图表..."):
            try:
                # 调用 run_prompt，替换 sql_prompts.txt 中的 {table_schema} 宏
                llm_output = llm_client.run_prompt(table_schema=table_schema_prompt)
                
                st.session_state['llm_raw_output'] = llm_output
                sql_list = parse_llm_json_output(llm_output)
                st.session_state['sql_list'] = sql_list
                
                if sql_list:
                    st.success(f"成功生成 {len(sql_list)} 条 SQL 分析建议。")
                else:
                    st.warning("LLM 未能生成有效的 SQL 列表。")
                
            except Exception as e:
                st.error(f"调用 LLM 失败：{e}")

    # 显示原始 LLM 输出 (用于调试)
    if 'llm_raw_output' in st.session_state:
        with st.expander("查看 LLM 原始 JSON 输出"):
            st.code(st.session_state['llm_raw_output'], language="json")

    # 5. 执行 SQL 并绘图 
    if 'sql_list' in st.session_state and st.session_state['sql_list']:
        st.header("📈 SQL 执行结果与 Plotly 绘图")
        
        # --- 获取中文列名映射 (用于 Plotly 标签) ---
        schema_df = duckdb_client.query(f"SELECT column_en, column_cn FROM main.table_meta WHERE table_name = '{selected_table_name}';")
        column_mapping = {}
        if not schema_df.empty and "error" not in schema_df.columns:
             column_mapping = schema_df.set_index('column_en')['column_cn'].to_dict()
        # -------------------------------------------
        
        descriptions = [item['description'] for item in st.session_state['sql_list'] if 'description' in item]
        if not descriptions:
            st.warning("解析出的 SQL 列表缺少描述信息，无法创建标签页。")
            return
            
        tabs = st.tabs(descriptions)
        
        for i, sql_item in enumerate(st.session_state['sql_list']):
            sql_query = sql_item['sql']
            description = sql_item.get('description', f"分析查询 {i+1}")
            
            with tabs[i]:
                st.subheader(f"分析 {i+1}: {description}")
                
                st.code(sql_query, language="sql")
                
                with st.spinner(f"正在执行 SQL [{description}]..."):
                    result_df = duckdb_client.query(sql_query)
                
                if "error" in result_df.columns:
                    st.error(f"SQL 执行失败: {result_df['error'].iloc[0]}")
                else:
                    try:
                        # 传入中文列名映射，用于 Plotly 轴标签
                        plot_data(result_df, description, column_mapping)
                        with st.expander("查看原始数据"):
                            st.dataframe(result_df)
                    except Exception as e:
                        st.error(f"绘图或数据显示失败。错误: {e}")
                        st.dataframe(result_df)


if __name__ == '__main__':
    # 确保路径被正确添加到 sys.path
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir / "db") not in sys.path:
        sys.path.append(str(root_dir / "db"))
    if str(root_dir / "llm_client") not in sys.path:
        sys.path.append(str(root_dir / "llm_client"))
        
    main()