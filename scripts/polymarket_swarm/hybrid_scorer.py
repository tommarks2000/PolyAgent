"""Hybrid Scoring System for Polymarket Predictions.

Combines multiple signal sources into a unified score:
1. Research-based edge (Perplexity/news analysis)
2. LLM swarm consensus
3. Order book liquidity
4. Historical performance calibration
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalStrength(str, Enum):
    """Signal strength levels."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    CONFLICTING = "CONFLICTING"


@dataclass
class HybridSignal:
    """Individual signal component."""
    name: str
    value: float  # Normalized -1 to +1 (negative = NO, positive = YES)
    confidence: float  # 0 to 1
    weight: float  # Relative weight in hybrid score
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridScore:
    """Combined hybrid score for a market."""
    market_id: str
    market_question: str
    timestamp: str

    # Individual signals
    signals: List[HybridSignal] = field(default_factory=list)

    # Combined scores
    raw_score: float = 0.0  # Weighted sum of signals (-1 to +1)
    confidence: float = 0.0  # Combined confidence (0 to 1)
    strength: SignalStrength = SignalStrength.WEAK

    # Final recommendation
    recommendation: str = "SKIP"  # YES, NO, or SKIP
    edge_estimate: float = 0.0  # Estimated edge percentage
    position_size_multiplier: float = 1.0  # Kelly-adjusted sizing

    # Diagnostics
    signal_agreement: float = 0.0  # How much signals agree (0 to 1)
    dominant_signal: str = ""  # Which signal is strongest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "timestamp": self.timestamp,
            "signals": [
                {
                    "name": s.name,
                    "value": s.value,
                    "confidence": s.confidence,
                    "weight": s.weight,
                }
                for s in self.signals
            ],
            "raw_score": self.raw_score,
            "confidence": self.confidence,
            "strength": self.strength.value,
            "recommendation": self.recommendation,
            "edge_estimate": self.edge_estimate,
            "position_size_multiplier": self.position_size_multiplier,
            "signal_agreement": self.signal_agreement,
            "dominant_signal": self.dominant_signal,
        }


class HybridScorer:
    """Combines multiple signals into hybrid trading score."""

    # Default weights for each signal type
    DEFAULT_WEIGHTS = {
        "research_edge": 0.35,      # Perplexity/news research
        "swarm_consensus": 0.25,    # LLM multi-model consensus
        "orderbook_quality": 0.20,  # Order book health
        "market_sentiment": 0.10,   # News sentiment
        "historical_edge": 0.10,    # Historical accuracy adjustment
    }

    # Thresholds
    MIN_SCORE_FOR_TRADE = 0.15  # Minimum raw score to recommend trade
    MIN_CONFIDENCE_FOR_TRADE = 0.55  # Minimum confidence
    STRONG_SIGNAL_THRESHOLD = 0.30  # Score above this = STRONG
    MODERATE_SIGNAL_THRESHOLD = 0.15  # Score above this = MODERATE

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize hybrid scorer.

        Args:
            weights: Optional custom weights for signals
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def calculate_hybrid_score(
        self,
        market_id: str,
        market_question: str,
        current_yes_price: float,
        # Research signals
        research_edge: Optional[float] = None,  # Edge % from research (-100 to +100)
        research_confidence: float = 0.5,
        # Swarm signals
        swarm_prediction: Optional[str] = None,  # YES, NO, or SKIP
        swarm_strength: float = 0.0,  # 0 to 1
        swarm_vote_count: int = 0,
        # Order book signals
        orderbook_healthy: bool = True,
        orderbook_spread: float = 0.0,  # Spread percentage
        orderbook_liquidity_score: float = 0.5,  # 0 to 1
        # Sentiment signals
        news_sentiment: float = 0.0,  # -1 to +1
        # Historical calibration
        historical_accuracy: float = 0.5,  # Our historical accuracy
    ) -> HybridScore:
        """Calculate hybrid score from all signal sources.

        Args:
            market_id: Market identifier
            market_question: The prediction question
            current_yes_price: Current YES price (0-1)
            research_edge: Edge percentage from research analysis
            research_confidence: Confidence in research edge
            swarm_prediction: LLM swarm consensus prediction
            swarm_strength: Swarm consensus strength (0-1)
            swarm_vote_count: Number of models that voted
            orderbook_healthy: Whether order book is healthy
            orderbook_spread: Spread percentage
            orderbook_liquidity_score: Liquidity score (0-1)
            news_sentiment: News sentiment score (-1 to +1)
            historical_accuracy: Our historical accuracy rate

        Returns:
            HybridScore with combined analysis
        """
        score = HybridScore(
            market_id=market_id,
            market_question=market_question,
            timestamp=datetime.now().isoformat(),
        )

        # 1. Research Edge Signal
        if research_edge is not None:
            # Normalize to -1 to +1 (divide by 50 to scale typical edges)
            normalized_edge = max(-1.0, min(1.0, research_edge / 50))
            score.signals.append(HybridSignal(
                name="research_edge",
                value=normalized_edge,
                confidence=research_confidence,
                weight=self.weights.get("research_edge", 0.35),
                raw_data={"edge_percent": research_edge},
            ))

        # 2. Swarm Consensus Signal
        if swarm_prediction and swarm_prediction != "SKIP":
            # Convert YES/NO to +1/-1
            swarm_value = 1.0 if swarm_prediction == "YES" else -1.0
            # Scale by consensus strength
            swarm_value *= swarm_strength

            score.signals.append(HybridSignal(
                name="swarm_consensus",
                value=swarm_value,
                confidence=swarm_strength,
                weight=self.weights.get("swarm_consensus", 0.25),
                raw_data={
                    "prediction": swarm_prediction,
                    "strength": swarm_strength,
                    "vote_count": swarm_vote_count,
                },
            ))

        # 3. Order Book Quality Signal
        # This is a modifier - good orderbook boosts confidence, bad reduces it
        if orderbook_healthy:
            ob_value = orderbook_liquidity_score * 0.5  # Slight positive bias for healthy books
        else:
            ob_value = -0.5  # Penalty for unhealthy books

        # Spread penalty
        spread_penalty = min(orderbook_spread / 20, 0.5)  # Max 0.5 penalty for 10%+ spread
        ob_value -= spread_penalty

        score.signals.append(HybridSignal(
            name="orderbook_quality",
            value=ob_value,
            confidence=orderbook_liquidity_score,
            weight=self.weights.get("orderbook_quality", 0.20),
            raw_data={
                "healthy": orderbook_healthy,
                "spread": orderbook_spread,
                "liquidity_score": orderbook_liquidity_score,
            },
        ))

        # 4. Market Sentiment Signal
        if news_sentiment != 0:
            score.signals.append(HybridSignal(
                name="market_sentiment",
                value=news_sentiment,
                confidence=min(abs(news_sentiment) + 0.3, 1.0),  # Higher sentiment = higher confidence
                weight=self.weights.get("market_sentiment", 0.10),
                raw_data={"sentiment": news_sentiment},
            ))

        # 5. Historical Edge Adjustment
        # If we've historically been accurate, boost confidence
        # If we've been inaccurate, reduce confidence
        hist_adjustment = (historical_accuracy - 0.5) * 2  # -1 to +1
        score.signals.append(HybridSignal(
            name="historical_edge",
            value=hist_adjustment * 0.2,  # Small adjustment
            confidence=min(historical_accuracy + 0.2, 1.0),
            weight=self.weights.get("historical_edge", 0.10),
            raw_data={"accuracy": historical_accuracy},
        ))

        # Calculate combined score
        self._calculate_combined_score(score)

        # Generate recommendation
        self._generate_recommendation(score, current_yes_price)

        return score

    def _calculate_combined_score(self, score: HybridScore):
        """Calculate the combined weighted score."""
        if not score.signals:
            return

        # Weighted sum of signal values
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0

        for signal in score.signals:
            weighted_sum += signal.value * signal.weight
            total_weight += signal.weight
            confidence_sum += signal.confidence * signal.weight

        score.raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        score.confidence = confidence_sum / total_weight if total_weight > 0 else 0.0

        # Calculate signal agreement
        if len(score.signals) > 1:
            # Check if signals point in the same direction
            positive_signals = sum(1 for s in score.signals if s.value > 0)
            negative_signals = sum(1 for s in score.signals if s.value < 0)
            total_directional = positive_signals + negative_signals

            if total_directional > 0:
                agreement = max(positive_signals, negative_signals) / total_directional
                score.signal_agreement = agreement
            else:
                score.signal_agreement = 0.5
        else:
            score.signal_agreement = 1.0

        # Find dominant signal
        if score.signals:
            dominant = max(score.signals, key=lambda s: abs(s.value * s.weight))
            score.dominant_signal = dominant.name

        # Determine strength
        abs_score = abs(score.raw_score)
        if abs_score >= self.STRONG_SIGNAL_THRESHOLD and score.signal_agreement > 0.7:
            score.strength = SignalStrength.STRONG
        elif abs_score >= self.MODERATE_SIGNAL_THRESHOLD:
            if score.signal_agreement < 0.5:
                score.strength = SignalStrength.CONFLICTING
            else:
                score.strength = SignalStrength.MODERATE
        else:
            if score.signal_agreement < 0.5:
                score.strength = SignalStrength.CONFLICTING
            else:
                score.strength = SignalStrength.WEAK

    def _generate_recommendation(self, score: HybridScore, current_yes_price: float):
        """Generate final recommendation from score."""

        # Check thresholds
        if abs(score.raw_score) < self.MIN_SCORE_FOR_TRADE:
            score.recommendation = "SKIP"
            score.edge_estimate = 0.0
            return

        if score.confidence < self.MIN_CONFIDENCE_FOR_TRADE:
            score.recommendation = "SKIP"
            score.edge_estimate = 0.0
            return

        if score.strength == SignalStrength.CONFLICTING:
            score.recommendation = "SKIP"
            score.edge_estimate = 0.0
            return

        # Generate recommendation
        if score.raw_score > 0:
            score.recommendation = "YES"
            # Edge = how much higher we think fair value is
            fair_value = current_yes_price + (score.raw_score * 0.3)  # Scale adjustment
            score.edge_estimate = (fair_value - current_yes_price) * 100
        else:
            score.recommendation = "NO"
            # Edge = how much lower we think fair value is
            fair_value = current_yes_price + (score.raw_score * 0.3)
            score.edge_estimate = (current_yes_price - fair_value) * 100

        # Ensure edge is positive (we're always expressing edge as positive)
        score.edge_estimate = abs(score.edge_estimate)

        # Position size multiplier (simplified Kelly)
        # Higher confidence + higher edge = larger position
        if score.edge_estimate > 0:
            # Kelly fraction = (p*b - q) / b where b = odds, p = prob of winning
            # Simplified: edge * confidence
            kelly = (score.edge_estimate / 100) * score.confidence
            # Cap at 0.25 (never bet more than 25% of bankroll)
            score.position_size_multiplier = min(kelly * 2, 0.25)
        else:
            score.position_size_multiplier = 0.0

    def explain_score(self, score: HybridScore) -> str:
        """Generate human-readable explanation of the score.

        Args:
            score: HybridScore to explain

        Returns:
            Explanation string
        """
        lines = [
            f"HYBRID SCORE ANALYSIS",
            f"{'='*50}",
            f"Market: {score.market_question[:50]}...",
            f"",
            f"SIGNALS:",
        ]

        for signal in score.signals:
            direction = "YES" if signal.value > 0 else "NO" if signal.value < 0 else "NEUTRAL"
            lines.append(
                f"  {signal.name}: {signal.value:+.2f} ({direction}) "
                f"[conf: {signal.confidence:.0%}, weight: {signal.weight:.0%}]"
            )

        lines.extend([
            f"",
            f"COMBINED:",
            f"  Raw Score: {score.raw_score:+.3f}",
            f"  Confidence: {score.confidence:.1%}",
            f"  Signal Agreement: {score.signal_agreement:.1%}",
            f"  Strength: {score.strength.value}",
            f"  Dominant Signal: {score.dominant_signal}",
            f"",
            f"RECOMMENDATION: {score.recommendation}",
            f"  Edge Estimate: {score.edge_estimate:.1f}%",
            f"  Position Size: {score.position_size_multiplier:.1%} of bankroll",
        ])

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("HYBRID SCORER TEST")
    print("=" * 70)

    scorer = HybridScorer()

    # Test case 1: Strong YES signal
    print("\nTest 1: Strong YES Signal")
    print("-" * 40)
    score1 = scorer.calculate_hybrid_score(
        market_id="test_1",
        market_question="Will the Federal Reserve cut rates in December 2025?",
        current_yes_price=0.75,
        research_edge=15.0,  # 15% edge
        research_confidence=0.8,
        swarm_prediction="YES",
        swarm_strength=0.85,
        swarm_vote_count=5,
        orderbook_healthy=True,
        orderbook_spread=2.0,
        orderbook_liquidity_score=0.7,
        news_sentiment=0.4,
        historical_accuracy=0.65,
    )
    print(scorer.explain_score(score1))

    # Test case 2: Conflicting signals
    print("\n\nTest 2: Conflicting Signals")
    print("-" * 40)
    score2 = scorer.calculate_hybrid_score(
        market_id="test_2",
        market_question="Will Trump win the 2028 election?",
        current_yes_price=0.45,
        research_edge=-10.0,  # Research says NO
        research_confidence=0.6,
        swarm_prediction="YES",  # But swarm says YES
        swarm_strength=0.7,
        swarm_vote_count=4,
        orderbook_healthy=True,
        orderbook_spread=3.0,
        orderbook_liquidity_score=0.6,
        news_sentiment=-0.2,
        historical_accuracy=0.55,
    )
    print(scorer.explain_score(score2))

    # Test case 3: Poor liquidity
    print("\n\nTest 3: Poor Liquidity")
    print("-" * 40)
    score3 = scorer.calculate_hybrid_score(
        market_id="test_3",
        market_question="Will nuclear weapon be used in 2025?",
        current_yes_price=0.05,
        research_edge=8.0,
        research_confidence=0.7,
        swarm_prediction="YES",
        swarm_strength=0.6,
        swarm_vote_count=3,
        orderbook_healthy=False,  # Bad order book
        orderbook_spread=15.0,  # Wide spread
        orderbook_liquidity_score=0.2,
        news_sentiment=0.1,
        historical_accuracy=0.6,
    )
    print(scorer.explain_score(score3))

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
