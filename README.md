# Polymarket Agent

基于 [Polymarket-Agent交易系统设计.md](Polymarket-Agent交易系统设计.md) v1.0 搭建的 AI Agent 概率套利系统。

## 设计原则

不预测事件，**追踪资金**。仅在 `P_true - P_market > 6%` 时进场。

## 架构

```
数据层  ──→  主线（10 分钟）                              副线（30 秒）
            scan → detect → estimate → debate → execute    risk monitor
                                  ↓
                          SQLite traces ──→ 学习层（每日复盘）
```

7 个模块：
1. [scanner](src/scanner/market_scanner.py) — 规则粗筛
2. [detector](src/detector/smart_money.py) — 5 类 K 线信号
3. [probability](src/probability/estimator.py) — 多源加权
4. [agents](src/agents/debate.py) — Bull / Bear / Risk Judge
5. [execution](src/execution/engine.py) — CLOB 限价单（dry-run/live）
6. [risk](src/risk/manager.py) — 止损 / 移动止损 / 到期
7. [learning](src/learning/loop.py) — 每日复盘

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env       # 填写 ANTHROPIC_API_KEY 即可跑读取/打分

python main.py scan-once            # 单次主线
python main.py monitor-once         # 单次副线
python main.py review               # 每日复盘
python main.py run                  # 调度器（dry_run 默认）
python main.py run --live           # 实盘（需 POLYMARKET_PRIVATE_KEY）
```

默认 `RUN_MODE=dry_run` — 只记录决策，不下真实订单。文档建议先纸面交易 2 周再上 $500 实盘。

## 关键决策

- **Python 3.11+**
- **Claude `claude-opus-4-7`**（设计文档里的 `claude-3-7-sonnet` 已弃用）
- **SQLite** 持久化（`data/traces.db`）
- **Edge floor 6%**，**1/4 Kelly**，**单笔上限 5% bankroll**

## 风险

详见设计文档第八章。本仓库仅供研究，不构成投资建议。
