"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            # 从开始日期到截止日期，按照tic中的股票代码依次下载
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1: # 检查列索引是否有多级
                temp_df.columns = temp_df.columns.droplevel(1) # 删除第二级索引
            temp_df["tic"] = tic
            # 将temp_df数据合并到data_df数据中，按行维度axis=0进行合并
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1 # 记录下载失败的次数
        if num_failures == len(self.ticker_list): # 提示用户所有股票的数据均下载失败
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates 
        # 这个操作是 Pandas 中处理 DataFrame 索引的核心功能之一，特别是在金融时间序列数据分析中非常常见
        # data_df_0 = data_df # 这个是保留原始数据的操作，方便对比Yahoo财经提供的原始数据和处理后的数据变化
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names 对原始数据中的特征名进行重命名
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp", # 已复权收盘价（后复权）
                    "Close": "close", # 原始收盘价
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            if not auto_adjust: # 是否自动应用复权调整（雅虎默认行为）
                data_df = self._adjust_prices(data_df)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0) 
        # 创建一个day column，星期一 = 0，星期二 = 1，星期三 = 2， 星期四 = 3， 星期五 = 4，礼拜一至礼拜四才是交易日
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data 丢弃数据为NaN的行，并重置索引值index序列号
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())
        
        # 最终数据重拍序，先按日期date升序排列，相同date中，再根据tic升序排列
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df #, data_df_0

    # 这个函数执行的是股票价格的后复权调整，是金融数据处理中的核心操作
    def _adjust_prices(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # use adjusted close price instead of close price
        # 复权因子 adj = 复权价 / 原始价
        data_df["adj"] = data_df["adjcp"] / data_df["close"]
        # 将所有价格列（开盘/最高/最低/收盘）乘以复权因子，使历史价格反映当前资本结构下的可比价格
        for col in ["open", "high", "low", "close"]:
            data_df[col] *= data_df["adj"]

        # drop the adjusted close price column
        # "adjcp", "adj"完成计算使命，就可以丢弃了！
        return data_df.drop(["adjcp", "adj"], axis=1)

    def select_equal_rows_stock(self, df):
        # select_equal_rows_stock 方法实现了一个股票数据平衡筛选的关键功能，主要用于解决金融时间序列分析中常见的数据不均衡问题。
        # 核心目的：筛选出数据记录数量达到平均水平的股票，确保不同股票在数据集中的样本量相对均衡。避免模型出现偏差。
        
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
