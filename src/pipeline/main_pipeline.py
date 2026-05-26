"""Main loop (every 10 min by default): scan → detect → estimate → debate → execute."""
from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS
from src.agents.debate import DebateOrchestrator
from src.data.news_client import NewsClient
from src.data.polymarket_client import PolymarketClient
from src.data.types import TradeDecision
from src.detector.smart_money import SmartMoneyDetector, is_whitelisted_combo
from src.execution.engine import ExecutionEngine
from src.notify.telegram import Notifier
from src.probability.estimator import ProbabilityEstimator
from src.probability.llm_client import LLMClient
from src.probability.ml_estimator import MLProbabilityEstimator
from src.probability.ml_short_horizon_estimator import (
    DECISION_THRESHOLD_PRED_RETURN,
    MLShortHorizonEstimator,
)
from src.risk.circuit_breaker import CircuitBreaker
from src.scanner.market_scanner import FilterConfig, MarketScanner
from src.storage.db import TraceStore
from src.utils.kelly import kelly_position_usdc
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

    mode = SETTINGS.estimator_mode.lower()
    if mode == "ml_short_horizon":
        estimator = MLShortHorizonEstimator(client=client, store=store)
        log.info("Estimator: ML short-horizon (6h forward, late-stage markets)")
    elif mode in ("ml", "ml_only"):
        estimator = MLProbabilityEstimator(client=client, store=store)
        log.info(f"Estimator: ML (mode={mode})")
    else:
        estimator = ProbabilityEstimator(llm, news)
        log.info("Estimator: LLM (Claude + news + base rate)")

    return PipelineComponents(
        client=client,
        scanner=MarketScanner(client, FilterConfig()),
        detector=SmartMoneyDetector(),
        estimator=estimator,
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

    mode = SETTINGS.estimator_mode.lower()
    if mode == "ml_short_horizon":
        n_signals, n_buys = _run_ml_short_horizon_cycle(components, candidates, candidate_limit)
        components.store.record_scan(
            n_raw=len(candidates), n_filtered=n_filtered,
            n_signals=n_signals, n_buys=n_buys,
        )
        log.info(f"Cycle done: filtered={n_filtered} signals={n_signals} buys={n_buys}")
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
        # 12h dedupe — mirrors the backtest `dedupe_bars=12`. Prevents paying
        # for the same LLM judgement on a market that has been in a triggered
        # state for many consecutive 10-min scans.
        if components.store.recent_signal_within(market.market_id, hours=12.0):
            log.debug(f"Signal dedupe hit (within 12h): {market.question[:60]}")
            continue
        n_signals += 1
        components.store.record_signal(market, detection.signals, detection.score)

        log.info(f"Signal HIT {market.question[:60]} score={detection.score} sigs={detection.signals}")

        prob = components.estimator.estimate(market, detection)

        if mode == "ml_only":
            decision = _ml_only_decision(market, detection, prob)
        else:
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


# ----------------------------------------------------------------------

ML_ONLY_EDGE_FLOOR = 0.06


def _ml_only_decision(market, detection, prob) -> TradeDecision:
    """Bypass Bull/Bear. Buy if ML edge > floor; size with conservative Kelly."""
    edge = prob.p_true - market.price
    if edge < ML_ONLY_EDGE_FLOOR or prob.confidence == "low":
        return TradeDecision(
            market_id=market.market_id, token_id=market.token_id, side="buy",
            market_price=market.price, p_true=prob.p_true, edge=edge,
            position_size_usdc=0.0, action="skip",
            reason=f"ml_edge<{ML_ONLY_EDGE_FLOOR} or low confidence",
        )
    size = kelly_position_usdc(
        p_true=prob.p_true, p_market=market.price,
        bankroll=SETTINGS.bankroll_usdc,
        fraction_multiplier=0.25,
        max_fraction=SETTINGS.max_position_fraction,
    )
    if size <= 0:
        return TradeDecision(
            market_id=market.market_id, token_id=market.token_id, side="buy",
            market_price=market.price, p_true=prob.p_true, edge=edge,
            position_size_usdc=0.0, action="skip", reason="kelly=0",
        )
    return TradeDecision(
        market_id=market.market_id, token_id=market.token_id, side="buy",
        market_price=market.price, p_true=prob.p_true, edge=edge,
        position_size_usdc=size, action="buy",
        reason=f"ml_only: edge={edge:+.4f} conf={prob.confidence}",
    )


def _run_ml_short_horizon_cycle(components, candidates, candidate_limit: int) -> tuple[int, int]:
    """ML-short-horizon decision loop. Bypasses K-line detector and whitelist
    entirely — the ML model is the alpha gate. Each scanner-passed candidate
    gets scored directly; trades fire when predicted 6h return >= threshold."""
    from src.data.types import DetectionResult

    n_evaluated = 0
    n_buys = 0
    fake_detection = DetectionResult(triggered=True, score=0, signals=["ml_short_horizon"])

    # 48h dedupe (was 12h). Live data showed model gives identical predictions
    # for the same market at different price levels — 12h cooldown was too
    # short and caused chasing tops + averaging down on Israel-Hezbollah:
    # bought at $0.097, $0.13, $0.084 all on the same +14% prediction.
    for market in candidates[:candidate_limit]:
        if components.store.recent_signal_within(market.market_id, hours=48.0):
            continue

        prob = components.estimator.estimate(market, fake_detection)
        if prob.confidence == "low":
            # Either market too old (>14d) or predicted return too small.
            continue

        n_evaluated += 1
        components.store.record_signal(market, ["ml_short_horizon"], 0)
        log.info(
            f"ML short-horizon eval: {market.question[:60]} "
            f"px={market.price:.4f} pred_ret={prob.p_true - market.price:+.4f} "
            f"conf={prob.confidence}"
        )

        decision = _ml_short_horizon_decision(market, fake_detection, prob)
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
    return n_evaluated, n_buys


def _ml_short_horizon_decision(market, detection, prob) -> TradeDecision:
    """Short-horizon decision: trade when predicted 6h return >= threshold.

    Since estimator sets p_true = market_price + predicted_return, edge IS
    the predicted_return. Threshold from sweep is +0.20 (sharpe +0.198).
    """
    predicted_return = prob.p_true - market.price
    if prob.confidence == "low":
        return TradeDecision(
            market_id=market.market_id, token_id=market.token_id, side="buy",
            market_price=market.price, p_true=prob.p_true, edge=predicted_return,
            position_size_usdc=0.0, action="skip",
            reason=prob.uncertainty or "low confidence (predicted return too small)",
        )
    if predicted_return < DECISION_THRESHOLD_PRED_RETURN:
        return TradeDecision(
            market_id=market.market_id, token_id=market.token_id, side="buy",
            market_price=market.price, p_true=prob.p_true, edge=predicted_return,
            position_size_usdc=0.0, action="skip",
            reason=f"predicted_return {predicted_return:+.4f} < {DECISION_THRESHOLD_PRED_RETURN}",
        )
    size = kelly_position_usdc(
        p_true=prob.p_true, p_market=market.price,
        bankroll=SETTINGS.bankroll_usdc,
        fraction_multiplier=0.25,
        max_fraction=SETTINGS.max_position_fraction,
    )
    if size <= 0:
        return TradeDecision(
            market_id=market.market_id, token_id=market.token_id, side="buy",
            market_price=market.price, p_true=prob.p_true, edge=predicted_return,
            position_size_usdc=0.0, action="skip", reason="kelly=0",
        )
    return TradeDecision(
        market_id=market.market_id, token_id=market.token_id, side="buy",
        market_price=market.price, p_true=prob.p_true, edge=predicted_return,
        position_size_usdc=size, action="buy",
        reason=f"ml_short_horizon: pred_ret={predicted_return:+.4f} conf={prob.confidence}",
    )
