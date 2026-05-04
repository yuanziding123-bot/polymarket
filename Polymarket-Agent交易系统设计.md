# Polymarket Agent 交易系统设计

> **状态：** 设计稿 v1.0 | 2026-04-20
> **定位：** 基于概率定价偏差的 AI Agent 套利系统
> **声明：** 仅供研究，不构成投资建议

---

## 一、Polymarket 市场特性

### 核心机制

Polymarket 是基于 Polygon 链的去中心化**预测市场**，采用 CLOB（中央限价订单簿）模型：

```
每个市场 = 一个二元期权合约
  YES token：事件发生 → 收敛至 $1 USDC
  NO  token：事件不发生 → 收敛至 $1 USDC

当前价格 = 市场对该事件发生概率的隐含估计
（价格 0.23 = 市场认为发生概率 23%）
```

### 套利逻辑

**Alpha 的来源：**

```
市场隐含概率 ≠ 真实概率
          ↓
    定价偏差 > 阈值
          ↓
         Edge
```

核心公式：
```
Edge = P_true - P_market
预期收益 = Edge × 投入资金 × 赔率
```

案例验证（包不同，前高盛量化）：
- 只买 Edge > 6% 的合约
- 价格 7-19c 区间（低概率事件），真实概率 60-90%
- 3 个月：$2,000 → $8,191，Sharpe 2.30，胜率 81%

### 市场规模

- 活跃标的：6 万+
- 涵盖：政治、宏观经济、体育、加密、科技
- 每日成交量：数亿 USDC
- 清算：链上自动，无对手方风险

---

## 二、开源框架调研

### 2.1 Polymarket 官方 SDK

| 库 | 用途 |
|----|------|
| `py-clob-client` | CLOB 订单簿操作（下单、查询、撤单）|
| `polymarket-agents` | Polymarket 官方 Agent SDK，封装了市场数据 + 交易接口 |
| `gamma-client` | Gamma Market（做市商 AMM 接口）|

```bash
pip install py-clob-client
pip install polymarket-agents  # Polymarket 官方维护
```

### 2.2 多 Agent 交易框架

| 框架 | Stars | 核心特点 | 适用性 |
|------|-------|---------|-------|
| **TradingAgents**（UCLA/MIT）| 高 | 模拟真实投研机构：基本面/技术面/情绪分析多角色，Bull/Bear 辩论机制 | ★★★★ 架构可复用 |
| **virattt/ai-hedge-fund** | 高 | 轻量多 Agent PoC，Python，股票市场 | ★★★ 学习参考 |
| **Vibe-Trading**（HKUDS）| 新兴 | 68 个专业 Skill，5 大数据源，一条命令跑通回测 | ★★★ Skill 可借鉴 |
| **发明者量化（Polymarket 专版）**| 闭源 | 双线调度、K线异常检测、多角色 AI | ★★★★★ 直接参考 |

### 2.3 TradingAgents 架构详解（最具参考价值）

```
┌─────────────────────────────────────────────────────┐
│                  TradingAgents                       │
│                                                      │
│  研究层          辩论层         决策层              │
│  ┌─────────┐   ┌──────────┐   ┌─────────────┐     │
│  │基本面分析│   │Bull Agent│   │             │     │
│  │技术面分析│ → │Bear Agent│ → │ 交易者 Agent│     │
│  │情绪分析 │   │风险管理  │   │ (多风险等级)│     │
│  │新闻分析 │   └──────────┘   └─────────────┘     │
│  └─────────┘                                         │
│                                                      │
│  特点：Bull vs Bear 辩论 → 避免单一角度偏见          │
└─────────────────────────────────────────────────────┘
```

### 2.4 发明者量化 Polymarket 系统（已验证有效）

双线调度设计：
```
主线（每 10 分钟）：筛选标的 → K线异常检测 → 新闻验证 → AI多角色分析 → 下单
副线（每 30 秒）：持仓监控 → 移动止损 → 到期赎回
```

K线异常检测 5 类信号：
1. **缓慢爬升** — 总涨 >5%，单根最大 <1.5%（刻意压节奏建仓）
2. **成交量线性增长** — 线性回归斜率 >0，R² >0.5
3. **回调收窄** — 近期回调深度 < 早期的 60%
4. **横盘突破** — MA60/MA120 偏差 <2%，当前价超 MA60 的 3%
5. **成交量突增** — 近 5 根均量 > 过去 60 根基准量的 2.5 倍

---

## 三、系统整体架构

### 设计原则

1. **不预测事件，追踪资金** — 聪明钱必然在价格上留痕
2. **概率定价偏差** — 只在 Edge 足够大时才交易
3. **多 Agent 验证** — 避免单一信号误判
4. **双线调度** — 主线找机会，副线管持仓，互不干扰
5. **Harness 架构** — 模型 + 工程系统，可测量可迭代

### 系统总架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  Polymarket Agent 交易系统                       │
│                                                                  │
│  ┌────────────────────────────────────┐                         │
│  │            数据层                  │                         │
│  │  Polymarket API │ Tavily新闻 │ X  │                         │
│  │  CoinGecko │ 历史交易数据         │                         │
│  └────────────────┬───────────────────┘                         │
│                   │                                              │
│  ┌────────────────▼───────────────────┐                         │
│  │         主线（每 10 分钟）          │                         │
│  │                                    │                         │
│  │  ① 市场扫描 & 粗筛（规则过滤）     │                         │
│  │       ↓                            │                         │
│  │  ② K线异常检测（资金信号）          │                         │
│  │       ↓                            │                         │
│  │  ③ 概率估算（多方法融合）           │                         │
│  │       ↓                            │                         │
│  │  ④ Edge 计算（定价偏差）           │                         │
│  │       ↓                            │                         │
│  │  ⑤ 多 Agent 验证（信号确认）       │                         │
│  │       ↓                            │                         │
│  │  ⑥ 执行决策（下单 / 跳过）         │                         │
│  └────────────────────────────────────┘                         │
│                                                                  │
│  ┌────────────────────────────────────┐                         │
│  │         副线（每 30 秒）            │                         │
│  │  持仓监控 → 移动止损 → 到期处理     │                         │
│  └────────────────────────────────────┘                         │
│                                                                  │
│  ┌────────────────────────────────────┐                         │
│  │         学习层（每日）              │                         │
│  │  复盘 Traces → 提炼 Context → 优化  │                         │
│  └────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、各模块详解

### 模块 1：市场扫描 & 粗筛

**目标：** 从 6 万+标的中，筛出 50-100 个值得深度分析的候选

**规则过滤条件：**

```python
FILTER_CONFIG = {
    # 价格区间：只关注低概率端（赔率弹性大）
    "min_price": 0.05,    # 最低 5c（避免接近归零的噪音）
    "max_price": 0.50,    # 最高 50c（超过 50% 已充分定价）
    
    # 流动性要求
    "min_volume_24h": 5000,      # 24h 成交量 > $5000 USDC
    "min_liquidity": 10000,      # 订单簿深度 > $10000
    "max_spread_pct": 0.05,      # 买卖价差 < 5%
    
    # 时间窗口：排除即将到期的（信息优势不足）
    "min_days_to_expiry": 3,
    "max_days_to_expiry": 90,
    
    # 活跃状态检查
    "must_be_active": True,
    "must_accept_orders": True,
}

# 同一事件 YES/NO 只保留价格更低的方向（赔率更高）
```

**输出：** 候选标的列表（市场ID、当前价格、基本元数据）

---

### 模块 2：K 线异常检测（资金信号层）

**核心思想：** 不预测事件，追踪智慧资金（Smart Money）留下的价格痕迹

```python
class SmartMoneyDetector:
    """
    检测 5 类资金行为异常，任意触发 2+ 类判为有效信号
    """
    
    def detect(self, closes: list, volumes: list) -> DetectionResult:
        signals = []
        score = 0
        
        # 信号1：缓慢爬升（刻意建仓）
        if self._slow_grind(closes):
            signals.append("slow_grind")
            score += 1
        
        # 信号2：成交量线性增长（持续入场）
        if self._volume_trend(volumes):
            signals.append("vol_trend")
            score += 1
        
        # 信号3：回调收窄（筹码稳定）
        if self._narrowing_pullback(closes):
            signals.append("narrow_pullback")
            score += 1
        
        # 信号4：横盘突破（突破压力位）
        if self._breakout(closes):
            signals.append("breakout")
            score += 1
        
        # 信号5：成交量突增（加速入场）
        if self._vol_spike(volumes):
            signals.append("vol_spike")
            score += 1
        
        return DetectionResult(
            triggered=score >= 2,
            score=score,
            signals=signals
        )
    
    def _slow_grind(self, closes):
        slice120 = closes[-120:]
        total_change = (slice120[-1] - slice120[0]) / slice120[0]
        max_single = max(abs(closes[i]-closes[i-1])/closes[i-1] 
                        for i in range(1, len(slice120)))
        return total_change > 0.05 and max_single < 0.015
    
    def _volume_trend(self, volumes):
        slope, r2 = linear_regression(volumes[-60:])
        return slope > 0 and r2 > 0.5
    
    def _narrowing_pullback(self, closes):
        # 近期回调深度 < 早期的 60%
        ...
    
    def _breakout(self, closes):
        ma60 = mean(closes[-60:])
        ma120 = mean(closes[-120:])
        bias = abs(ma60 - ma120) / ma120
        return bias < 0.02 and closes[-1] > ma60 * 1.03
    
    def _vol_spike(self, volumes):
        recent_avg = mean(volumes[-5:])
        baseline_avg = mean(volumes[-65:-5])
        return recent_avg > baseline_avg * 2.5
```

---

### 模块 3：概率估算（核心 Alpha 来源）

**目标：** 估算事件的真实概率 P_true，与市场价格 P_market 对比

**多方法融合：**

```
P_true = w1 × P_llm + w2 × P_base_rate + w3 × P_news + w4 × P_corr

其中：
  P_llm       = Claude 对事件的概率判断（主要权重）
  P_base_rate = 历史同类事件的基准发生率
  P_news      = 新闻情绪信号的隐含概率
  P_corr      = 相关市场的价格信号（交叉验证）

权重（初始）：w1=0.4, w2=0.2, w3=0.25, w4=0.15
（权重根据历史预测精度动态调整）
```

**Claude 概率估算 Prompt 模版：**

```python
PROBABILITY_PROMPT = """
你是一位专业的预测市场分析师。

事件：{event_title}
描述：{event_description}
到期时间：{expiry_date}
当前市场价格（隐含概率）：{market_price:.1%}

背景信息：
{news_context}

请分析：
1. 这个事件发生的概率是多少？（给出精确数字，如 0.34）
2. 你的主要依据是什么？（3条以内）
3. 最大的不确定性来源？
4. 置信度（低/中/高）？

只输出 JSON：
{{"probability": 0.XX, "reasoning": ["...", "..."], "uncertainty": "...", "confidence": "high/medium/low"}}
"""
```

---

### 模块 4：多 Agent 验证（TradingAgents 架构变体）

借鉴 TradingAgents 的 Bull vs Bear 辩论机制，适配预测市场：

```
输入：候选市场 + K线信号 + 概率估算结果
                    ↓
┌─────────────────────────────────────────────────────┐
│                 Agent 验证层                          │
│                                                      │
│  做多 Agent (Bull)        做空 Agent (Bear)          │
│  ─────────────────        ─────────────────          │
│  • 找支持事件发生的证据   • 找反对事件发生的证据     │
│  • 评估信息优势信号       • 识别市场过度反应         │
│  • 赔率合理性分析         • 流动性风险评估           │
│                    ↓辩论                             │
│              风险裁判 Agent                          │
│  ─────────────────────────────────────────────       │
│  • 综合两方论点                                      │
│  • 计算最终 Edge                                     │
│  • 确定头寸大小（Kelly 公式）                        │
│  • 输出：买入 / 跳过 / 做空                          │
└─────────────────────────────────────────────────────┘
```

**Kelly 公式计算头寸：**

```python
def kelly_position(p_true: float, p_market: float, 
                   bankroll: float, max_fraction: float = 0.05) -> float:
    """
    Kelly Criterion 计算最优头寸
    
    预测市场的赔率：买入 p_market，赢得 (1 - p_market)，输掉 p_market
    """
    if p_true <= p_market:
        return 0  # 无 Edge，不交易
    
    # 赔率 b = (1 - p_market) / p_market（赢/输的比率）
    b = (1 - p_market) / p_market
    p = p_true
    q = 1 - p_true
    
    # Kelly 分数
    kelly_f = (b * p - q) / b
    
    # 保守调整：使用 1/4 Kelly，并设置上限
    safe_f = min(kelly_f * 0.25, max_fraction)
    
    return bankroll * safe_f
```

---

### 模块 5：执行层

```python
from py_clob_client import ClobClient
from polymarket_agents import PolymarketAgentToolkit

class ExecutionEngine:
    def __init__(self, private_key: str, api_key: str):
        self.clob = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,  # Polygon
            private_key=private_key,
        )
        self.toolkit = PolymarketAgentToolkit(api_key=api_key)
    
    def execute_trade(self, decision: TradeDecision) -> TradeResult:
        if decision.action == "skip":
            return TradeResult(executed=False, reason="no_edge")
        
        # 计算限价单价格（在市场价基础上稍微调整，避免滑点）
        limit_price = self._calculate_limit_price(
            market_price=decision.market_price,
            side=decision.side,
            slippage_tolerance=0.005  # 0.5% 滑点容忍
        )
        
        # 下限价单
        order = self.clob.create_and_post_order({
            "token_id": decision.token_id,
            "price": limit_price,
            "size": decision.position_size,
            "side": decision.side,  # "buy" 或 "sell"
            "type": "GTC",  # Good Till Cancelled
        })
        
        return TradeResult(
            executed=True,
            order_id=order["id"],
            filled_price=limit_price,
            size=decision.position_size
        )
```

---

### 模块 6：风险控制层（副线，每 30 秒）

```python
class RiskManager:
    """
    持仓监控：止盈、止损、到期处理
    """
    
    RISK_RULES = {
        "stop_loss_pct": 0.40,      # 持仓亏损 40% 止损
        "take_profit_pct": 0.60,    # 持仓盈利 60% 止盈
        "trailing_stop_pct": 0.15,  # 移动止损：从最高点回撤 15%
        "min_days_to_hold": 1,      # 最少持有 1 天（避免频繁交易）
        "force_close_days": 1,      # 到期前 1 天强制平仓
    }
    
    def monitor_positions(self, positions: list[Position]) -> list[Action]:
        actions = []
        
        for pos in positions:
            current_price = self.get_current_price(pos.token_id)
            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            
            # 1. 到期处理
            if pos.days_to_expiry <= self.RISK_RULES["force_close_days"]:
                actions.append(CloseAction(pos, reason="near_expiry"))
                continue
            
            # 2. 止损
            if pnl_pct <= -self.RISK_RULES["stop_loss_pct"]:
                actions.append(CloseAction(pos, reason="stop_loss"))
                continue
            
            # 3. 移动止损（保护盈利）
            if pnl_pct >= 0.20:  # 盈利 20% 后启动移动止损
                drawdown = (pos.peak_price - current_price) / pos.peak_price
                if drawdown >= self.RISK_RULES["trailing_stop_pct"]:
                    actions.append(CloseAction(pos, reason="trailing_stop"))
                    continue
            
            # 4. 止盈
            if pnl_pct >= self.RISK_RULES["take_profit_pct"]:
                actions.append(CloseAction(pos, reason="take_profit"))
        
        return actions
```

---

### 模块 7：学习层（Agent 自我优化）

基于 Agent 学习飞轮（Traces → Context → 优化）：

```python
class LearningLoop:
    """
    每日复盘：分析交易 Traces，提炼改进 Context，更新系统参数
    """
    
    def daily_review(self):
        # 1. 收集所有交易 Traces
        traces = self.get_today_traces()
        
        # 2. 计算各信号类型的准确率
        signal_performance = self.analyze_signal_accuracy(traces)
        
        # 3. Claude 分析失败案例
        failures = [t for t in traces if t.pnl < 0]
        analysis = self.claude_analyze_failures(failures)
        
        # 4. 更新参数
        self.update_filter_thresholds(signal_performance)
        self.update_probability_weights(traces)
        
        # 5. 记录 Context 用于下一轮
        self.save_learned_context(analysis)
    
    def claude_analyze_failures(self, failures):
        prompt = f"""
        以下是今日亏损的交易，请分析失败原因并给出改进建议：
        
        {format_trades(failures)}
        
        输出：
        1. 主要失败模式（最多3个）
        2. 哪类信号可靠性最低
        3. 建议调整的参数
        """
        return call_claude(prompt)
```

---

## 五、策略变体

### 策略 A：概率偏差套利（低风险，入门）

**来源：** 包不同案例（高盛量化方法）

```
条件：Edge > 6%（市场价 vs 真实概率偏差 >6%）
目标价格区间：0.07 - 0.19（低价二元期权）
头寸：单笔最大 5% 仓位（Kelly 1/4）
预期：81% 胜率，3个月 409% 回报（历史验证）
```

### 策略 B：智慧资金跟踪（中风险，进阶）

**来源：** 发明者量化案例

```
条件：K线异常检测触发 2+ 类信号
验证：新闻情绪一致性确认
执行：顺着资金方向建仓
移动止损：从最高点回撤 15%
```

### 策略 C：策略逆向工程（高收益，高难度）

**来源：** 听风谈投资案例

```python
# Prompt 模板
"""
分析 Polymarket 用户 {wallet_address} 的历史交易：
- 交易品类集中度
- 入场价格特征
- 持仓周期
- 赚钱规律

然后：复写这个策略，生成可执行的代码
"""
```

---

## 六、技术栈

```
数据层：
├── polymarket-agents      # Polymarket 官方 Agent SDK
├── py-clob-client         # CLOB 订单簿操作
├── tavily-python          # 新闻搜索
└── ccxt                   # 加密货币市场数据对比

AI 层：
├── anthropic (claude-3-7-sonnet) # 主分析引擎
├── LangChain / Harness    # 工作流编排
└── 自定义 Prompt 模板      # 概率估算 + 多角色分析

执行层：
├── Polygon 链（USDC 结算）
├── MetaMask / 私钥管理
└── 限价单（GTC 类型）

监控层：
├── 每日 Traces 记录（CSV + SQLite）
├── 性能看板（Streamlit 或直接打印）
└── 微信/Telegram 报警（止损触发时通知）

部署：
├── VPS（月费 $5，推荐 DigitalOcean / Vultr）
├── Docker 容器化
└── Cron Job 调度（主线10分钟，副线30秒）
```

---

## 七、快速启动

### 第一步：环境准备

```bash
pip install py-clob-client polymarket-agents anthropic tavily-python

# 配置环境变量
export POLYMARKET_API_KEY="..."
export POLYMARKET_PRIVATE_KEY="..."  # Polygon 钱包私钥
export ANTHROPIC_API_KEY="..."
export TAVILY_API_KEY="..."
```

### 第二步：先用 Claude Code 逆向工程现有策略

```
# 直接发给 Claude Code：
"分析 Polymarket 钱包 0xde17f7144fbd0eddb2679132c10ff5e74b120988 
的历史交易，找出最近一个月赚钱最多的操作模式，
然后写一个 Python 脚本复制这个策略"
```

### 第三步：纸面交易验证

先跑 2 周只扫描不下单（dry run 模式），验证：
- 每日扫描到多少个 Edge > 6% 的机会
- 信号触发后 1 周的胜率
- 系统稳定性

### 第四步：小资金实盘

从 $500 USDC 开始，单笔不超过 $25（5% Kelly）。

---

## 八、风险提示与局限性

| 风险 | 说明 | 缓解方式 |
|------|------|---------|
| **链风险** | Polygon 链或 Polymarket 合约被攻击 | 分散资金，不在链上存放大额 |
| **流动性风险** | 冷门市场成交量不足，无法平仓 | 严格执行 24h 成交量下限过滤 |
| **模型失效风险** | Claude 概率估算偏差 | 多方法融合，持续回测校准 |
| **信息劣势** | 内幕玩家先行，K 线信号滞后 | 专注"追资金"而非预测事件本身 |
| **监管风险** | 预测市场在部分地区受限 | 请评估当地法规 |
| **过拟合风险** | 历史参数未必适用未来 | 定期重新校准，谨慎使用回测结果 |

---

## 九、参考资料

| 来源 | 关键贡献 |
|------|---------|
| `包不同：高盛量化交易员Polymarket` | 核心策略逻辑：概率偏差 > 6%，胜率 81% |
| `Polymarket 多角色智能分析交易系统` | 双线调度、K线异常检测代码实现 |
| `如何用AI对Polymarket策略做逆向工程` | 策略复制方法论，Claude Code 实现 |
| `TradingAgents（UCLA/MIT）` | 多角色 LLM 交易框架架构 |
| `virattt/ai-hedge-fund` | 轻量多 Agent 实现参考 |
| `AI量化研究员年化54.81%` | 自主因子发现框架（学术论文） |
| `68技能量化框架Vibe-Trading` | Skill 化量化工具集参考 |
