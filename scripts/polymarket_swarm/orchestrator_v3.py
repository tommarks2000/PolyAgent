"""Enhanced Swarm Orchestrator v3.0

Integrates all components:
1. Perplexity deep research (with OpenAI fallback)
2. Multi-model LLM predictions (swarm consensus)
3. Order book analysis
4. Hybrid scoring system
5. Historical performance tracking

This is the most comprehensive analysis system combining:
- Edge-based analysis (our original approach)
- LLM consensus (Moon Dev's approach)
- Liquidity analysis (order books)
- Performance feedback loop
"""
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from models import (
    Market, TradeOpportunity, Recommendation
)
from agents import AGENT_REGISTRY
from polymarket_client import PolymarketClient
from web_search_client import WebSearchClient
from orderbook_client import OrderBookClient
from llm_predictions import LLMSwarmClient, SwarmPrediction
from hybrid_scorer import HybridScorer, HybridScore
from performance_tracker import PerformanceTracker
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


class EnhancedOrchestratorV3:
    """Ultimate orchestrator combining all analysis methods."""

    def __init__(
        self,
        use_web_search: bool = True,
        use_llm_swarm: bool = True,
        use_orderbook: bool = True,
        use_hybrid_scoring: bool = True,
        track_performance: bool = True,
    ):
        """Initialize v3 orchestrator.

        Args:
            use_web_search: Enable Perplexity/OpenAI web search
            use_llm_swarm: Enable multi-model LLM predictions
            use_orderbook: Enable order book analysis
            use_hybrid_scoring: Enable hybrid scoring
            track_performance: Enable performance tracking
        """
        # Core clients
        self.polymarket = PolymarketClient()

        # Initialize agents (for backward compatibility)
        self.agents = {
            name: agent_class()
            for name, agent_class in AGENT_REGISTRY.items()
        }

        # Web search (Perplexity + OpenAI fallback)
        self.web_search = WebSearchClient() if use_web_search else None
        self.use_web_search = use_web_search and (
            self.web_search and self.web_search.is_available
        )

        # LLM swarm predictions
        self.llm_swarm = LLMSwarmClient() if use_llm_swarm else None
        self.use_llm_swarm = use_llm_swarm and (
            self.llm_swarm and len(self.llm_swarm.active_models) > 0
        )

        # Order book analysis
        self.orderbook = OrderBookClient() if use_orderbook else None
        self.use_orderbook = use_orderbook

        # Hybrid scoring
        self.scorer = HybridScorer() if use_hybrid_scoring else None
        self.use_hybrid_scoring = use_hybrid_scoring

        # Performance tracking
        self.tracker = PerformanceTracker() if track_performance else None
        self.track_performance = track_performance

        # Caches for reporting
        self.research_cache: Dict[str, Any] = {}
        self.swarm_cache: Dict[str, SwarmPrediction] = {}
        self.orderbook_cache: Dict[str, Dict] = {}
        self.score_cache: Dict[str, HybridScore] = {}

        # Print initialization status
        print("=" * 60)
        print("ORCHESTRATOR V3 INITIALIZED")
        print("=" * 60)
        print(f"Web Search: {'ENABLED' if self.use_web_search else 'DISABLED'}")
        if self.web_search:
            providers = []
            if self.web_search.perplexity_available:
                providers.append("Perplexity")
            if self.web_search.openai_available:
                providers.append("OpenAI")
            print(f"  Providers: {', '.join(providers)}")
        print(f"LLM Swarm: {'ENABLED' if self.use_llm_swarm else 'DISABLED'}")
        if self.llm_swarm:
            print(f"  Models: {list(self.llm_swarm.active_models.keys())}")
        print(f"Order Book: {'ENABLED' if self.use_orderbook else 'DISABLED'}")
        print(f"Hybrid Scoring: {'ENABLED' if self.use_hybrid_scoring else 'DISABLED'}")
        print(f"Performance Tracking: {'ENABLED' if self.track_performance else 'DISABLED'}")
        print("=" * 60)

    def analyze_market(
        self,
        market: Market,
        deep_analysis: bool = True,
    ) -> Optional[TradeOpportunity]:
        """Run comprehensive analysis on a single market.

        Args:
            market: Market data from Polymarket API
            deep_analysis: Whether to run full analysis (slower but better)

        Returns:
            TradeOpportunity if opportunity found, None otherwise
        """
        market_id = str(market.id)
        question = market.question or "Unknown"

        # 1. Web Search Research
        news_context = None
        research_edge = None
        research_confidence = 0.5
        key_factors = []

        if self.use_web_search and deep_analysis:
            research = self.web_search.research_market(
                market_question=question,
                market_description=market.description,
                current_price=market.yes_price,
            )

            if research.success:
                news_context = research.content
                research_confidence = research.confidence
                self.research_cache[market_id] = research.to_dict()

                # Extract edge estimate from research (basic sentiment)
                research_edge = self._estimate_edge_from_research(
                    research.content,
                    market.yes_price
                )

        # 2. Order Book Analysis
        orderbook_healthy = True
        orderbook_spread = 0.0
        orderbook_liquidity = 0.5

        if self.use_orderbook and market.clobTokenIds:
            ob_summary = self.orderbook.get_market_summary(
                market.clobTokenIds,
                question
            )
            self.orderbook_cache[market_id] = ob_summary

            orderbook_healthy = ob_summary.get("healthy", True)
            orderbook_spread = ob_summary.get("average_spread_percent", 0)
            orderbook_liquidity = ob_summary.get("average_liquidity_score", 0.5)

            # Skip if order book is very unhealthy (unless edge is huge)
            if not orderbook_healthy and orderbook_spread > 15:
                if research_edge is None or abs(research_edge) < 20:
                    return None

        # 3. LLM Swarm Predictions
        swarm_prediction = None
        swarm_strength = 0.0
        swarm_vote_count = 0

        if self.use_llm_swarm and deep_analysis:
            swarm = self.llm_swarm.get_swarm_prediction(
                market_id=market_id,
                market_question=question,
                current_yes_price=market.yes_price,
                market_description=market.description,
                news_context=news_context[:1000] if news_context else None,
                key_factors=key_factors if key_factors else None,
            )

            self.swarm_cache[market_id] = swarm
            swarm_prediction = swarm.consensus_prediction
            swarm_strength = swarm.consensus_strength
            swarm_vote_count = swarm.total_responses

        # 4. Run Traditional Agents (for additional signals)
        agent_context = {
            "news_context": news_context,
            "sentiment_score": self._extract_sentiment(news_context) if news_context else 0,
            "orderbook_healthy": orderbook_healthy,
            "orderbook_spread": orderbook_spread,
        }

        # Get probability estimate from ProbabilityAnalyst
        prob_result = self.agents["probability_analyst"].analyze(market, agent_context)
        agent_edge = prob_result.edge_estimate if prob_result else 0

        # 5. Hybrid Scoring (combines all signals)
        if self.use_hybrid_scoring:
            # Get historical accuracy
            historical_accuracy = 0.5
            if self.tracker:
                metrics = self.tracker.calculate_metrics()
                historical_accuracy = metrics.accuracy_rate if metrics.accuracy_rate > 0 else 0.5

            # Use agent edge if we don't have research edge
            final_research_edge = research_edge if research_edge is not None else agent_edge

            hybrid = self.scorer.calculate_hybrid_score(
                market_id=market_id,
                market_question=question,
                current_yes_price=market.yes_price,
                research_edge=final_research_edge,
                research_confidence=research_confidence,
                swarm_prediction=swarm_prediction,
                swarm_strength=swarm_strength,
                swarm_vote_count=swarm_vote_count,
                orderbook_healthy=orderbook_healthy,
                orderbook_spread=orderbook_spread,
                orderbook_liquidity_score=orderbook_liquidity,
                news_sentiment=self._extract_sentiment(news_context) if news_context else 0,
                historical_accuracy=historical_accuracy,
            )

            self.score_cache[market_id] = hybrid

            # Use hybrid recommendation
            if hybrid.recommendation == "SKIP":
                return None

            if hybrid.edge_estimate < MIN_EDGE_PERCENT:
                return None

            recommendation = Recommendation.YES if hybrid.recommendation == "YES" else Recommendation.NO
            edge_percent = hybrid.edge_estimate
            confidence = hybrid.confidence
            fair_value = market.yes_price + (hybrid.raw_score * 0.3)

        else:
            # Fall back to traditional agent-based recommendation
            recommendation = prob_result.recommendation
            edge_percent = prob_result.edge_estimate
            confidence = prob_result.confidence
            fair_value = market.yes_price + (edge_percent / 100)

            if recommendation == Recommendation.SKIP:
                return None

            if edge_percent < MIN_EDGE_PERCENT:
                return None

            if confidence < MIN_CONFIDENCE:
                return None

        # 6. Build Opportunity
        opportunity = TradeOpportunity(
            market_id=market.id,
            market_question=question,
            condition_id=market.conditionId,
            clob_token_ids=market.clobTokenIds,
            recommendation=recommendation,
            confidence=confidence,
            edge_percent=edge_percent,
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            estimated_fair_yes_price=max(0.01, min(0.99, fair_value)),
            reasoning=self._build_reasoning(market_id),
            agent_votes=self._get_agent_votes(market_id),
            risk_factors=self._get_risk_factors(market_id),
            news_summary=news_context[:500] if news_context else None,
            volume=market.get_volume_safe(),
            liquidity=market.get_liquidity_safe(),
        )

        # 7. Record Prediction for Tracking
        if self.track_performance and self.tracker:
            self.tracker.record_prediction(
                market_id=market_id,
                market_question=question,
                our_prediction=recommendation.value,
                our_edge_percent=edge_percent,
                our_confidence=confidence,
                our_fair_value=fair_value,
                market_yes_price=market.yes_price,
                market_no_price=market.no_price,
                market_volume=market.get_volume_safe(),
                market_liquidity=market.get_liquidity_safe(),
                swarm_consensus=swarm_prediction,
                swarm_strength=swarm_strength,
                orderbook_healthy=orderbook_healthy,
                research_sources=["web_search"] if news_context else [],
            )

        return opportunity

    def _estimate_edge_from_research(self, content: str, current_price: float) -> float:
        """Estimate edge from research content using keyword analysis."""
        content_lower = content.lower()

        # Positive indicators (suggest YES probability should be higher)
        positive = [
            "likely", "expected", "confirmed", "agreement", "progress",
            "momentum", "breakthrough", "imminent", "certain", "scheduled"
        ]

        # Negative indicators (suggest YES probability should be lower)
        negative = [
            "unlikely", "stalled", "failed", "rejected", "collapsed",
            "delayed", "obstacles", "impossible", "doubt", "cancelled"
        ]

        pos_count = sum(1 for word in positive if word in content_lower)
        neg_count = sum(1 for word in negative if word in content_lower)

        # Calculate sentiment-based adjustment
        if pos_count + neg_count == 0:
            return 0.0

        sentiment = (pos_count - neg_count) / (pos_count + neg_count + 1)

        # Convert to edge percentage (max 20% adjustment)
        edge = sentiment * 20

        return edge

    def _extract_sentiment(self, text: str) -> float:
        """Extract sentiment score from text (-1 to +1)."""
        if not text:
            return 0.0

        text_lower = text.lower()

        positive_words = {"good", "great", "positive", "success", "win", "gain", "up", "likely", "progress"}
        negative_words = {"bad", "poor", "negative", "fail", "loss", "down", "unlikely", "decline", "concern"}

        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)

        if pos + neg == 0:
            return 0.0

        return (pos - neg) / (pos + neg)

    def _build_reasoning(self, market_id: str) -> str:
        """Build comprehensive reasoning from all signals."""
        parts = []

        # Hybrid score reasoning
        if market_id in self.score_cache:
            score = self.score_cache[market_id]
            parts.append(f"Hybrid score: {score.raw_score:+.2f} ({score.strength.value})")

        # Swarm consensus
        if market_id in self.swarm_cache:
            swarm = self.swarm_cache[market_id]
            parts.append(
                f"LLM consensus: {swarm.consensus_prediction} "
                f"({swarm.yes_votes}Y/{swarm.no_votes}N)"
            )

        # Order book health
        if market_id in self.orderbook_cache:
            ob = self.orderbook_cache[market_id]
            health = "healthy" if ob.get("healthy") else "thin"
            parts.append(f"Order book: {health} ({ob.get('average_spread_percent', 0):.1f}% spread)")

        return " | ".join(parts) if parts else "Standard analysis"

    def _get_agent_votes(self, market_id: str) -> Dict[str, Recommendation]:
        """Get voting summary from all analysis methods."""
        votes = {}

        # Swarm votes
        if market_id in self.swarm_cache:
            swarm = self.swarm_cache[market_id]
            for pred in swarm.predictions:
                if pred.success:
                    if pred.prediction == "YES":
                        votes[f"LLM_{pred.model_name}"] = Recommendation.YES
                    elif pred.prediction == "NO":
                        votes[f"LLM_{pred.model_name}"] = Recommendation.NO
                    else:
                        votes[f"LLM_{pred.model_name}"] = Recommendation.SKIP

        # Hybrid score vote
        if market_id in self.score_cache:
            score = self.score_cache[market_id]
            if score.recommendation == "YES":
                votes["Hybrid"] = Recommendation.YES
            elif score.recommendation == "NO":
                votes["Hybrid"] = Recommendation.NO
            else:
                votes["Hybrid"] = Recommendation.SKIP

        return votes

    def _get_risk_factors(self, market_id: str) -> List[str]:
        """Compile risk factors from all analyses."""
        risks = []

        # Order book risks
        if market_id in self.orderbook_cache:
            ob = self.orderbook_cache[market_id]
            if not ob.get("healthy"):
                risks.append("Order book unhealthy")
            if ob.get("average_spread_percent", 0) > 5:
                risks.append(f"Wide spread: {ob.get('average_spread_percent', 0):.1f}%")
            if ob.get("issues"):
                risks.extend(ob["issues"][:2])

        # Swarm disagreement
        if market_id in self.swarm_cache:
            swarm = self.swarm_cache[market_id]
            if swarm.consensus_strength < 0.6:
                risks.append(f"Split LLM consensus: {swarm.consensus_strength:.0%}")

        # Hybrid score risks
        if market_id in self.score_cache:
            score = self.score_cache[market_id]
            if score.signal_agreement < 0.5:
                risks.append("Conflicting signals")

        return risks

    def find_opportunities(
        self,
        max_markets: int = 50,
        min_edge: float = MIN_EDGE_PERCENT,
        min_confidence: float = MIN_CONFIDENCE,
        deep_analysis: bool = True,
    ) -> List[TradeOpportunity]:
        """Scan markets for trading opportunities.

        Args:
            max_markets: Maximum markets to analyze
            min_edge: Minimum edge percentage
            min_confidence: Minimum confidence score
            deep_analysis: Use full analysis (slower but better)

        Returns:
            List of trading opportunities sorted by edge
        """
        # Fetch markets (all categories except crypto/sports)
        markets = self.polymarket.fetch_all_categories(
            limit=max_markets,
            min_volume=5000,
            min_liquidity=2000,
        )

        print(f"\nAnalyzing {len(markets)} markets...")
        print(f"Mode: {'Deep Analysis' if deep_analysis else 'Quick Scan'}")
        print()

        opportunities = []

        for i, market in enumerate(markets):
            question = market.question[:50] if market.question else "Unknown"
            print(f"[{i+1}/{len(markets)}] {question}...")

            try:
                opportunity = self.analyze_market(market, deep_analysis=deep_analysis)

                if opportunity:
                    if opportunity.edge_percent >= min_edge and opportunity.confidence >= min_confidence:
                        opportunities.append(opportunity)
                        print(f"  -> FOUND: {opportunity.recommendation.value} ({opportunity.edge_percent:.1f}% edge)")
                    else:
                        print(f"  -> Below threshold")
                else:
                    print(f"  -> SKIP")

            except Exception as e:
                print(f"  -> Error: {e}")

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        return opportunities

    def generate_report(self, opportunities: List[TradeOpportunity]) -> str:
        """Generate comprehensive analysis report."""
        if not opportunities:
            return "No trading opportunities found."

        lines = [
            "=" * 70,
            "POLYMARKET SWARM TRADER v3.0 - COMPREHENSIVE REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "ANALYSIS METHODS:",
            f"  Web Search: {'ENABLED' if self.use_web_search else 'DISABLED'}",
            f"  LLM Swarm: {'ENABLED' if self.use_llm_swarm else 'DISABLED'}",
            f"  Order Book: {'ENABLED' if self.use_orderbook else 'DISABLED'}",
            f"  Hybrid Scoring: {'ENABLED' if self.use_hybrid_scoring else 'DISABLED'}",
            "",
            f"Found {len(opportunities)} opportunities:",
            "",
        ]

        for i, opp in enumerate(opportunities, 1):
            market_id = str(opp.market_id)

            lines.extend([
                "=" * 50,
                f"OPPORTUNITY #{i}",
                "=" * 50,
                f"Market: {opp.market_question}",
                f"Market ID: {opp.market_id}",
                "",
                f">>> RECOMMENDATION: {opp.recommendation.value} <<<",
                f"Edge: {opp.edge_percent:.1f}%",
                f"Confidence: {opp.confidence:.1%}",
                "",
                f"Prices:",
                f"  Current: YES {opp.current_yes_price:.1%} / NO {opp.current_no_price:.1%}",
                f"  Fair Value: {opp.estimated_fair_yes_price:.1%}",
                "",
                f"Volume: ${opp.volume:,.0f}",
                f"Liquidity: ${opp.liquidity:,.0f}",
                "",
            ])

            # Hybrid score details
            if market_id in self.score_cache:
                score = self.score_cache[market_id]
                lines.extend([
                    "Hybrid Score Analysis:",
                    f"  Raw Score: {score.raw_score:+.3f}",
                    f"  Strength: {score.strength.value}",
                    f"  Signal Agreement: {score.signal_agreement:.0%}",
                    f"  Dominant Signal: {score.dominant_signal}",
                    "",
                ])

            # LLM Swarm details
            if market_id in self.swarm_cache:
                swarm = self.swarm_cache[market_id]
                lines.extend([
                    "LLM Swarm Consensus:",
                    f"  Prediction: {swarm.consensus_prediction}",
                    f"  Strength: {swarm.consensus_strength:.0%}",
                    f"  Votes: YES={swarm.yes_votes} NO={swarm.no_votes} SKIP={swarm.skip_votes}",
                ])

                for pred in swarm.predictions:
                    if pred.success:
                        lines.append(f"    {pred.model_name}: {pred.prediction} ({pred.confidence:.0%})")

                lines.append("")

            # Order book details
            if market_id in self.orderbook_cache:
                ob = self.orderbook_cache[market_id]
                lines.extend([
                    "Order Book Analysis:",
                    f"  Healthy: {'YES' if ob.get('healthy') else 'NO'}",
                    f"  Spread: {ob.get('average_spread_percent', 0):.2f}%",
                    f"  Liquidity Score: {ob.get('average_liquidity_score', 0):.2f}",
                    f"  Recommendation: {ob.get('recommendation', 'N/A')}",
                    "",
                ])

            # Risk factors
            if opp.risk_factors:
                lines.append(f"Risk Factors: {', '.join(opp.risk_factors)}")
            else:
                lines.append("Risk Factors: None identified")

            lines.extend([
                "",
                f"Reasoning: {opp.reasoning}",
                "",
            ])

        # Performance summary
        if self.tracker:
            metrics = self.tracker.calculate_metrics()
            lines.extend([
                "=" * 50,
                "HISTORICAL PERFORMANCE",
                "=" * 50,
                f"Total Predictions: {metrics.total_predictions}",
                f"Resolved: {metrics.resolved_predictions}",
                f"Accuracy: {metrics.accuracy_rate:.1%}",
                f"Avg Edge Captured: {metrics.avg_edge_captured:.1f}%",
                "",
            ])

        return "\n".join(lines)

    def to_json(self, opportunities: List[TradeOpportunity]) -> str:
        """Export opportunities to JSON."""
        data = []
        for opp in opportunities:
            market_id = str(opp.market_id)
            opp_data = opp.model_dump(mode='json')

            # Add detailed analysis data
            if market_id in self.research_cache:
                opp_data["research"] = self.research_cache[market_id]

            if market_id in self.swarm_cache:
                opp_data["swarm"] = self.swarm_cache[market_id].to_dict()

            if market_id in self.orderbook_cache:
                opp_data["orderbook"] = self.orderbook_cache[market_id]

            if market_id in self.score_cache:
                opp_data["hybrid_score"] = self.score_cache[market_id].to_dict()

            data.append(opp_data)

        return json.dumps(data, indent=2, default=str)

    def close(self):
        """Close all clients."""
        self.polymarket.close()
        if self.web_search:
            self.web_search.close()
        if self.llm_swarm:
            self.llm_swarm.close()
        if self.orderbook:
            self.orderbook.close()
        if self.tracker:
            self.tracker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("ORCHESTRATOR V3 TEST")
    print("=" * 70)

    with EnhancedOrchestratorV3(
        use_web_search=True,
        use_llm_swarm=True,
        use_orderbook=True,
        use_hybrid_scoring=True,
        track_performance=True,
    ) as orch:
        # Quick test with 5 markets
        opportunities = orch.find_opportunities(
            max_markets=5,
            min_edge=5.0,
            min_confidence=0.5,
            deep_analysis=True,
        )

        print()
        print(orch.generate_report(opportunities))

        if opportunities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v3_opportunities_{timestamp}.json"
            with open(filename, "w") as f:
                f.write(orch.to_json(opportunities))
            print(f"\nResults saved to: {filename}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
