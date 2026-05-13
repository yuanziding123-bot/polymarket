"""Main loop (every 10 min by default): scan → detect → estimate → debate → execute."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.debate import DebateOrchestrator
from src.data.news_client import NewsClient
from src.data.polymarket_client import PolymarketClient
from src.detector.smart_money import SmartMoneyDetector, is_whitelisted_combo
from src.execution.engine import ExecutionEngine
from src.notify.telegram import Notifier
from src.probability.estimator import ProbabilityEstimator
from src.probability.llm_client import LLMClient
from src.risk.circuit_breaker import CircuitBreaker
from src.scanner.market_scanner import FilterConfig, MarketScanner
from src.storage.db import TraceStore
from src.utils.logger import get_logger

log = get_logger("main_pipeline")


@dataclass
class PipelineComponents:
    client: PolymarketClient
    scanner: MarketScanner
    detector: SmartMoneyDetector
    estimator: ProbabilityEstimator
    debate: DebateOrchestrator
    execution: ExecutionEngine
    store: TraceStore
    news: NewsClient
    circuit_breaker: CircuitBreaker
    notifier: Notifier


def build_pipeline() -> PipelineComponents:
    client = PolymarketClient()
    store = TraceStore()
    news = NewsClient()
    llm = LLMClient()
    notifier = Notifier()

    return PipelineComponents(
        client=client,
        scanner=MarketScanner(client, FilterConfig()),
        detector=SmartMoneyDetector(),
        estimator=ProbabilityEstimator(llm, news),
        debate=DebateOrchestrator(llm),
        execution=ExecutionEngine(client, store, notifier=notifier),
        store=store,
        news=news,
        circuit_breaker=CircuitBreaker(store),
        notifier=notifier,
    )


def run_once(components: PipelineComponents, candidate_limit: int = 25) -> None:
    candidates = components.scanner.scan(raw_limit=500)
    n_filtered = len(candidates)
    if not candidates:
        log.info("No candidates after filtering")
        components.store.record_scan(0, 0, 0, 0)
        return

    n_signals = 0
    n_buys = 0

    # Cap per cycle to bound LLM cost; design doc targets 50-100 deeper analyses
    for market in candidates[:candidate_limit]:
        # interval = lookback window (1m = one month), fidelity = sample minutes
        candles = components.client.fetch_price_history(market.token_id, interval="1m", fidelity=60)
        detection = components.detector.detect(candles)
        if not detection.triggered:
            continue
        if not is_whitelisted_combo(detection.signals):
            log.debug(f"Signal triggered but not whitelisted {detection.signals} — skipping")
            continue
        n_signals += 1
        components.store.record_signal(market, detection.signals, detection.score)

        log.info(f"Signal HIT {market.question[:60]} score={detection.score} sigs={detection.signals}")

        prob = components.estimator.estimate(market, detection)
        news_items = components.news.search(market.question, max_results=5)
        decision = components.debate.run(market, detection, prob, news_items)
        components.store.record_decision(decision, prob.components)

        if decision.action == "buy":
            verdict = components.circuit_breaker.check(decision.position_size_usdc)
            if not verdict.allowed:
                components.notifier.circuit_breaker(verdict.reason)
                log.warning(f"Trade blocked by circuit breaker: {verdict.reason}")
                continue
            result = components.execution.execute(decision, market)
            if result.executed:
                n_buys += 1

    components.store.record_scan(
        n_raw=len(candidates), n_filtered=n_filtered,
        n_signals=n_signals, n_buys=n_buys,
    )
    log.info(f"Cycle done: filtered={n_filtered} signals={n_signals} buys={n_buys}")
