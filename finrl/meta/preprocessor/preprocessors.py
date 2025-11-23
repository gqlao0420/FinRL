from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler
from stockstats import StockDataFrame as Sdf

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)] 
        # 筛选并提取从起始日期至结束日期的数据，target_date_col是以"date"为特征的变量
    data = data.sort_values([target_date_col, "tic"], ignore_index=True) 
        # 将提取的数据按照优先target_date_col特征，兼顾"tic"特征进行排序
    data.index = data[target_date_col].factorize()[0] 
        # 排序后的数据，以target_date_col特征为分类变量（字符串、类别等），映射为整数数值，同时保留原始值与数值的对应关系，并将其设置为index
        # .factorize()[0] - 就是编码成整数
    return data # 使用pd.loc[index, :]索引，就能索引到对应index日期下，所有股票的所有信息！！！方便env进行输入～


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class GroupByScaler(BaseEstimator, TransformerMixin):
    """Sklearn-like scaler that scales considering groups of data.

    In the financial setting, this scale can be used to normalize a DataFrame
    with time series of multiple tickers. The scaler will fit and transform
    data for each ticker independently.
    """

    def __init__(self, by, scaler=MaxAbsScaler, columns=None, scaler_kwargs=None):
        """Initializes GoupBy scaler.

        Args:
            by: Name of column that will be used to group.
            scaler: Scikit-learn scaler class to be used.
            columns: List of columns that will be scaled.
            scaler_kwargs: Keyword arguments for chosen scaler.
        """
        self.scalers = {}  # dictionary with scalers
        self.by = by
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs

    def fit(self, X, y=None):
        """Fits the scaler to input data.

        Args:
            X: DataFrame to fit.
            y: Not used.

        Returns:
            Fitted GroupBy scaler.
        """
        # if columns aren't specified, considered all numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns
        # fit one scaler for each group
        for value in X[self.by].unique():
            X_group = X.loc[X[self.by] == value, self.columns]
            self.scalers[value] = self.scaler(**self.scaler_kwargs).fit(X_group)
        return self

    def transform(self, X, y=None):
        """Transforms unscaled data.

        Args:
            X: DataFrame to transform.
            y: Not used.

        Returns:
            Transformed DataFrame.
        """
        # apply scaler for each group
        X = X.copy()
        for value in X[self.by].unique():
            select_mask = X[self.by] == value
            X.loc[select_mask, self.columns] = self.scalers[value].transform(
                X.loc[select_mask, self.columns]
            )
        return X


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not 
            
            湍流指数：
            湍流指数是机构投资者应对极端市场环境的强大工具，尤其在黑天鹅事件频发的现代金融市场中，它能提供传统指标无法捕捉的风险信号。
            正确应用湍流指数可显著提升投资组合的抗风险能力，避免灾难性损失。
            
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.ffill().bfill()
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        # 重置索引值
        # factorize()返回 0-codes 和 1-uniques。uniques是对应列表去重后的值，codes是列表中每个元素在uniques中的位置(从0开始的整数序列)。
        # 讲解链接：https://blog.csdn.net/weixin_45144170/article/details/115376518
        
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        # 透视表 pd.pivot_table，一个表格基本组成三要素：行索引，列特征，数值，分别对应三个变量。
        # 讲解链接：https://blog.csdn.net/anshuai_aw1/article/details/88402641
        # 透视表是一种可以对数据动态排布并且分类汇总的表格格式。
        # 要谨记：股票数据，是一种动态数据，而非静态！！！
        
        merged_closes = merged_closes.dropna(axis=1)
        # 舍弃含空值的列
        
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        # 舍弃含空值的列后，列特征即tic就会有所变化，所以要用新的tic更新
        
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        
        在当前的 add_technical_indicator 方法中，看似只是简单的添加股票指标特征的操作。
        然而技术指标的实际计算逻辑并未直接体现在代码中，而是通过调用 stockstats 库（即 Sdf.retype 转换后的对象）隐式完成的。
        
        问题核心
            1.指标计算的黑箱：            
            代码中的 stock[stock.tic == unique_ticker[i]][indicator] 直接从 stockstats 转换后的对象中提取预计算的指标值，但未展示如何生成这些指标。            
            例如：
            rsi_14 或 macd 的具体计算公式未在代码中体现。
            
            2.依赖外部库：            
            实际计算由 stockstats 库内部实现（通过列名映射到预定义的指标计算规则）。
            例如：            
            列名 close_12_ema 会自动触发 12 日指数移动平均的计算。            
            列名 rsi_14 会自动计算 14 日相对强弱指数。
        
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        # 按各股计算股票技术指标，并添加进股票数据中
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_in    dicator, ignore_index=True
                    # )
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                # 异常处理
                except Exception as e:
                    print(e)
                    
            # 将计算后的指标按 tic 和 date 合并回原始数据
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        # 按 date 和 tic 重新排序，确保数据一致性
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
