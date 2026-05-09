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

    def get_stock_list(self,
            available_from: str = None, 
            available_until: str = None,
            exchange: list = ['sh', 'sz'],
        ) -> list[str]:
        """
        获取数据库中去重且排序后的股票代码列表。
        - 交易所筛选：通过 WHERE 子句匹配 code 前缀。
        - 上市时间推算：通过计算 stock_daily 表中每只股票的最小日期(MIN(ts))。
        """
        query_sql = "SELECT code FROM stock_daily"
        params = {}
        
        # 1. 处理交易所筛选 (WHERE 子句)
        if exchange:
            # 动态生成类似 (code LIKE :ex_0 OR code LIKE :ex_1) 的条件
            ex_conditions = []
            for i, ex in enumerate(exchange):
                param_key = f"ex_{i}"
                ex_conditions.append(f"code LIKE :{param_key}")
                params[param_key] = f"{ex.lower()}%"
            
            query_sql += f" WHERE ({' OR '.join(ex_conditions)})"

        # 2. 按股票代码分组 (必须在 WHERE 之后，HAVING 之前)
        query_sql += " GROUP BY code"

        # 3. 处理上市时间筛选 (HAVING 子句)
        having_clauses = []
        if available_from:
            from_dt = f"{available_from[:4]}-{available_from[4:6]}-{available_from[6:]} 00:00:00"
            having_clauses.append("MIN(ts) >= :from_dt")
            params["from_dt"] = from_dt

        if available_until:
            until_dt = f"{available_until[:4]}-{available_until[4:6]}-{available_until[6:]} 23:59:59"
            having_clauses.append("MIN(ts) <= :until_dt")
            params["until_dt"] = until_dt

        if having_clauses:
            query_sql += " HAVING " + " AND ".join(having_clauses)

        # 4. 增加排序
        query_sql += " ORDER BY code ASC;"

        # 执行查询
        with self.engine.connect() as conn:
            result = conn.execute(text(query_sql), params)
            stock_list = [row[0] for row in result]

        return stock_list
        


if __name__ == "__main__":
    client = LocalShareClient()
    stock_list = client.get_stock_list()
    print(stock_list)

    # df = client.get_daily_quote(symbol="600015", start_date="20250101", end_date="20251231", adjust="no_adj")
    # print(df)
