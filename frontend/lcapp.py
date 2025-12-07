# frontend/app.py
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.llm_da_pipeline import build_eda_pipeline
from db.duckdb_client import DuckDBClient


duckdb_client = DuckDBClient()
df = duckdb_client.get_table_list_with_sources()
table_name = df['table_name'].iloc[0]

schema = duckdb_client.generate_table_schema_prompt(selected_table_name=table_name)

st.set_page_config(
        page_title="LLM 驱动的绘图工具", 
        layout="wide"
    )

st.title("📊 LLM 驱动的 SQL 到 Plotly 图表")

user_question = st.text_input(
    "输入您想从数据库中查询并绘制图表的问题:",
    value="绘制每个城市的平均销售额柱状图"
)

if st.button("生成图表"):
    if not user_question:
        st.warning("请输入您的问题。")
    else:
        st.info("正在执行 LangChain (SQL生成 -> DB查询 -> Plotly代码生成 -> 代码执行)...")
        
        with st.spinner("正在努力生成图表中..."):
            try:
                # 运行最终的 LangChain 
                
                eda = build_eda_pipeline()
                result = eda.invoke({"table_schema": schema,"user_question":user_question}) 
                
                # 检查结果
                if result.get("fig") is not None:
                    st.success("图表生成成功！")
                    st.plotly_chart(result["fig"], use_container_width=True)
                else:
                    st.error("图表生成失败。")
                    st.code(result.get("error", "未知错误"))
                    
            except Exception as e:
                st.error(f"LangChain 运行发生严重错误: {e}")