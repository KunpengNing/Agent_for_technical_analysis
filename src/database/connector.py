import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

class LocalShareClient:

    def __init__(self, user="postgres", password="123456", host="localhost", port="5432"):
        db_url = f"postgresql://{user}:{password}@{host}:{port}/findata"
        self.engine = create_engine(db_url)

    def get_daily_quote(self, 
            symbol: str,
            start_date: str = "20000101", 
            end_date: str = "20251231", 
            adjust: str = "back_adj"
        ) -> pd.DataFrame:

        # 1. 股票代码标准化 (000001 -> sz000001)
        if symbol.startswith('6'): full_code = f"sh{symbol}"
        elif symbol.startswith(('0', '3')): full_code = f"sz{symbol}"
        elif symbol.startswith(('8', '9', '4')): full_code = f"bj{symbol}"
        else: full_code = symbol

        # 2. 日期格式化适配 Timestamptz
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 00:00:00"
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 23:59:59"

        # 3. SQL 核心逻辑构建

        # --- 前复权 (qfq) 逻辑 ---
        if adjust == "pre_adj":
            print("提示：不建议使用前复权数据进行策略研究，暂未实现此功能。")
            return None

        # --- 不复权 (nfq) 逻辑 ---
        elif adjust == 'no_adj':
            query_sql = f"""
            SELECT 
                (ts AT TIME ZONE 'Asia/Shanghai')::date AS "time",
                open AS "open",
                close AS "close",
                high AS "high",
                low AS "low",
                amount AS "amount",
                turnover AS "turnover"
            FROM stock_daily
            WHERE code = :code AND ts BETWEEN :start_dt AND :end_dt
            ORDER BY ts ASC;
            """

        # --- 后复权 (hfq) 逻辑 ---
        elif adjust == "back_adj":
            query_sql = f"""
            SELECT 
                (ts AT TIME ZONE 'Asia/Shanghai')::date AS "time",
                open * adj_factor AS "open",
                close * adj_factor AS "close",
                high * adj_factor AS "high",
                low * adj_factor AS "low",
                amount AS "amount",
                turnover AS "turnover"
            FROM stock_daily
            WHERE code = :code AND ts BETWEEN :start_dt AND :end_dt
            ORDER BY ts ASC;
            """

        # 无效参数
        else:
            print(f"提示：无效的复权参数 '{adjust}', 请使用 'pre_adj', 'no_adj', 'back_adj'。")
            return None

        # 6. 统一执行查询
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query_sql), conn, params={
                "code": full_code,
                "raw_symbol": symbol, # akshare 输出的是无后缀代码
                "start_dt": start_dt,
                "end_dt": end_dt
            })

        return df


if __name__ == "__main__":
    client = LocalShareClient()
    df = client.get_daily_quote(symbol="600015", start_date="20250101", end_date="20251231", adjust="no_adj")
    print(df)
