from frontend.app import st

if __name__ == "__main__":
    # 目前直接运行 Streamlit
    # 或者调试模块
    print("请按以下步骤运行：")
    print('-'*30)    
    print("步骤一，加载数据到Duckdb，执行：")
    print("请使用：python.exe db/xls_loader.py")
    print()
    print("步骤二，执行 llm 自动数据分析师主程序：")
    print("请使用：streamlit run frontend/app.py")
