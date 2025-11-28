from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    """
    self.data - 代表按同一个日期进行索引，包含账户及所有股票的完整信息(包含现金、各股收盘价、各股持仓量、各股的技术指标等)
    # 状态向量的标准结构（多股票情况）：
    self.state = [
        cash,           # 索引0: 现金
        price_1,        # 索引1: 股票1价格
        price_2,        # 索引2: 股票2价格
        ...,
        price_N,        # 索引N: 股票N价格
        shares_1,       # 索引N+1: 股票1持仓
        shares_2,       # 索引N+2: 股票2持仓
        ...,
        shares_N,       # 索引2N: 股票N持仓
        tech_1_stock1,  # 索引2N+1: 技术指标1-股票1
        tech_1_stock2,  # 索引2N+2: 技术指标1-股票2
        ...,
        tech_M_stockN   # 索引2N + M*N: 技术指标M-股票N
    ]
    
    self.stock_dim - 股票数量（通常问有多少只股票的意思，比如股票代码AAPL，AMGN，AXP等的总数量）
    
    索引规则：
    self.state[0] - 现金
    self.state[(self.stock_dim * 0 + 1):(self.stock_dim * 1 + 1)] - 各股收盘价
    self.state[(self.stock_dim * 1 + 1):(self.stock_dim * 2 + 1)] - 各股持仓量
    self.state[(self.stock_dim * 2 + 1):(self.stock_dim * 3 + 1)] - 技术指标1
    self.state[(self.stock_dim * 3 + 1):(self.stock_dim * 4 + 1)] - 技术指标2
    self.state[(self.stock_dim * 4 + 1):(self.stock_dim * 5 + 1)] - 技术指标3
    self.state[(self.stock_dim * 5 + 1):(self.stock_dim * 6 + 1)] - 技术指标5
    ......
    
    """

    metadata = {"render.modes": ["human"]}
        # 这个是字典变量，用于配置环境的渲染选项
        # metadata - 环境元数据配置字典
        # render.modes - 指定环境支持的渲染模式
        # ["human"] - 表示环境支持"human"这种渲染模式：1）屏幕会弹出可视化窗口，2）实时显示智能体与环境的交互过程，3）适合人类观察训练过程或测试效果
        # 其他模式：1) "rgb_array" - 返回环境的RGB数组，用于程序化处理; 2) "ansi" - 返回文本渲染结果

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int, # 股票数量N
        hmax: int, # 每次交易的最大股数限制，这个环境中，并没有设置持股动作，只有买和卖两个动作，通过限制最高操作股数来默认其他股票是持股动作。
                   # 这种做法有些脱离实际，后续可以加入持股动作，使交易环境更加接近现实！！！
        initial_amount: int, # 初始资金量
        num_stock_shares: list[int], # 每只股票的持仓数量
        buy_cost_pct: list[float], # 每只股票的买入交易成本率（百分比）
        sell_cost_pct: list[float], # 每只股票的卖出交易成本率（百分比）
        reward_scaling: float, # 奖励缩放因子，用于调整奖励值的范围，越小越适合强化学习
        state_space: int, # 状态空间的维度，通常 = 股票数量 * (价格特征数 + 技术指标数) + 账户信息
        action_space: int, # 动作空间的维度，通常是股票数量的3倍，每个股票有3种动作：买入、卖出、持有
        tech_indicator_list: list[str], # 使用的技术指标列表
        turbulence_threshold=None, # 市场波动阈值，超过此值时可能限制交易或清仓
        risk_indicator_col="turbulence", # 数据框中表示风险指标的列名，此处使用turbulence列作为风险指标
        make_plots: bool = False, # 是否在训练过程中生成图表
        print_verbosity=10, # 打印信息的频率
        day=0, # 当前交易日索引
        initial=True, # 是否是初始状态
        previous_state=[], # 前一个状态，用于状态转移
        model_name="", # 模型名称标识
        mode="", # 运行模式：train 或 trade
        iteration="", # 迭代次数或版本标识
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space # 这里赋值应该是状态的维度，是整数
        self.action_space = action_space # 这里赋值应该是动作的维度，是整数
        self.tech_indicator_list = tech_indicator_list
        
            # 在Gym中，spaces.Box 用于定义连续的 动作空间 或 观测(状态)空间。它表示一个n维的空间，其中每个维度都有上下界。
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,)) # 这里是通过动作维度生成一个连续的动作空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        ) # 这里是通过状态维度生成一个连续的状态空间
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0 # 奖励 - 直接指导智能体学习的方向
        self.turbulence = 0 # 波动率/湍流风险管理，避免在动荡市场中过度交易
        self.cost = 0 # 现实性考虑，反映真实交易中的摩擦成本，累计交易成本，用于计算净收益
        self.trades = 0 # 统计交易次数，用于控制交易频率，防止过度交易
        self.episode = 0 # 训练进度跟踪和超参数调度
        
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
            # 记录投资组合总价值的历史变化，这个是核心绩效指标，直接反映策略盈利能力
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        """
        这个函数操作单个股票卖出
        """
        def _do_sell_normal():
            # print(f"self.day = {self.day}, de_sell_normal: self.state[{index} + 2 * {self.stock_dim} + 1] = {self.state[index + 2 * self.stock_dim + 1]}") # 验证这个是MACD技术指标值，数值，不是True或False
            # 这个可交易信号是如何计算的啊？在_initiate_state()中，可是没有的啊。。。
            # 这里索引到的是MACD技术指标，有明显的逻辑错误，需要修改代码，明确可交易信号如何计算！！！！
            # 这里是故意简化，没有直接删除，就是希望每个人在真实的交易环境下，根据市场交易逻辑，设定可交易信号，用可交易信号判断买入卖出！！！良心！！！
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index - 交易信号是直接加在技术指标之前吗？
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                        # 确保action提供的数值，不能超过持有的股票数量，要在action和现持有股票数量之间取一个最小值，否则违背交易逻辑
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                        # 计算卖出股票的收益，要减去交易成本，注意交易成本是在卖出收益中减去
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance - 更新现金cash、更新持有的股票数量、更新交易成本
                        # 卖出股票收益后，要更新cash，要加
                    self.state[0] += sell_amount
                        # 卖出股票后，需要更新持有的股票数量
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                        # 计算卖出股票的交易成本，并累计
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                            # 这是清盘，股票全部清空啊
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance - 更新现金cash、更新持有的股票数量、更新交易成本
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """
        这个函数操作单个股票买入
        """
        def _do_buy():
            # 同理_do_buy()函数，这个判断可买入的信号，后期也要按照正确的逻辑梳理
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # 在采取买入之前，需要考虑现有资金还可以买多少股，结合action给的数量，取两者最小值。
                # print('available_amount:{}'.format(available_amount))

                # update balance 买入股票，要计算交易成本
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                # 买入股票，更新cash，要减少
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        # print(f"Without hmax - action = {actions}, action's shape is {actions.shape}, action's type is {type(actions)}")
        self.terminal = self.day >= len(self.df.index.unique()) - 1
            # 如果 self.day ≥ 最后一个交易日的索引 → self.terminal = True，结束
            # 否则 → self.terminal = False，未结束
        if self.terminal: # 强化学习已经结束，进行最后统计
            # print(f"Episode: {self.episode}")
            if self.make_plots: # 初始值默认是False
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, False, {}

        else: # terminal == False, 强化学习训练未结束，继续下一步！！！
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            # print(f"With hmax - action = {actions}, action's shape is {actions.shape}, action's type is {type(actions)}")
            actions = actions.astype(int)  # convert into integer because we can't by fraction of shares, 只取整数部分，小数部分舍弃
            
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            # 在此刻操作股票之前，统计总资产(账户现金+股票价值)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions) 
                # 按值大小，从小到大排序，但返回排序后的元素在原数组中的索引位置，而不是排序后的值本身。
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]] 
                # np.where(actions < 0)[0].shape[0] - 筛选actions中小于0的值的个数，np.where(actions < 0) - tuple类型, [0].shape[0] - 间接的出小于0的个数
                # [: np.where(actions < 0)[0].shape[0]] - 取值
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
                # argsort_actions[::-1] - 逆序排列，原本是从小到大，但这样操作就变成从大到小
                # [: np.where(actions > 0)[0].shape[0]] - 取直

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])
            print(f"actions are {actions}")
            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        """
        构建 state 成分，对应 state_space 的维度来详细说明：
        [0] - 对应当前时刻的总资产
        [1 : stock_dim + 1] - 对应当前时刻各股票的收盘价
        [stock_dim + 1 : stock_dim * 2 + 1] - 对应当前时刻各股票的持仓数量
        [stock_dim * 2 + 1: stock_dim * (2 + len(tech_indicator_list)) + 1] - 对应当前时刻各股票的技术指标值，这个切片中，每跨一个stock_dim长度，对应着同一个技术指标
            通过切片可以索引到同一个技术指标下，所有股票的指标数值，tech_num表示技术指标的序号，从0开始 - [stock_dim * (2 + tech_num) + 1 : stock_dim * (2 + tech_num + 1) + 1]
        """
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount] # 1. 初始资金
                    + self.data.close.values.tolist() # 2. 各股票收盘价
                    + self.num_stock_shares # 3. 各股票持仓数量
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    ) # 4. 所有技术指标
                )  # append initial stocks_share to initial state, instead of all zero
                   # 这里的 + 不是求和逻辑，而是Python列表的连接(concatenation)操作，而 sum(..., []) 是用于将嵌套列表扁平化。两者作用不同，但共同构建了最终的状态表示。
                   # # 状态向量有清晰的结构：
                        # state = [
                        #    cash,           # 资金信息
                        #    *prices,        # 市场信息  
                        #    *holdings,      # 持仓信息
                        #    *technical      # 技术信号
                        # ]
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
            # 设置随机数种子的方法，确保实验的可重复性
        self.np_random, seed = seeding.np_random(seed)
            # 1.将创建的随机数生成器赋值给 self.np_random
            # 2.同时获取实际使用的种子值 seed
        return [seed] # 返回包含种子值的列表（符合OpenAI Gym的接口规范）

    def get_sb_env(self):
            # 用于创建Stable-Baselines兼容环境的封装方法
        e = DummyVecEnv([lambda: self]) 
            # 1.将单个环境包装成Stable-Baselines需要的向量化环境格式; 
            # 2.lambda: self: 创建一个返回当前环境实例的匿名函数
            # 3.DummyVecEnv：Stable-Baselines中的环境包装器，用于处理单个环境
        obs = e.reset() # 重置向量化环境，返回初始观测值
        return e, obs
