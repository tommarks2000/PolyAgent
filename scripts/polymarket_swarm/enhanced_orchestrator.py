"""Enhanced swarm orchestrator with Perplexity deep research.

Extends the base orchestrator with LLM-powered analysis for
better edge detection and probability estimation.
Now includes order book analysis for better trade execution.
"""
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from models import (
    Market, Article, AgentResult, TradeOpportunity,
    Recommendation, SwarmConsensus
)
from agents import (
    AGENT_REGISTRY,
    MarketScannerAgent,
    NewsResearcherAgent,
    ProbabilityAnalystAgent,
    RiskValidatorAgent,
    CoordinatorAgent,
)
from polymarket_client import PolymarketClient
from enhanced_research import EnhancedResearchClient, EnhancedResearchResult
from orderbook_client import OrderBookClient, OrderBookAnalysis
from market_rag import MarketRAG
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


class EnhancedSwarmOrchestrator:
    """Enhanced orchestrator with Perplexity deep research."""

    def __init__(self, use_rag: bool = True, use_perplexity: bool = True, use_orderbook: bool = True):
        """Initialize enhanced orchestrator.

        Args:
            use_rag: Whether to use RAG for market selection
            use_perplexity: Whether to use Perplexity for deep research
            use_orderbook: Whether to analyze order books for liquidity
        """
        # Initialize agents
        self.agents = {
            name: agent_class()
            for name, agent_class in AGENT_REGISTRY.items()
        }

        # Initialize clients
        self.polymarket = PolymarketClient()
        self.research = EnhancedResearchClient()
        self.orderbook = OrderBookClient() if use_orderbook else None
        self.use_perplexity = use_perplexity and self.research.has_perplexity
        self.use_orderbook = use_orderbook

        # Initialize RAG (optional)
        self.use_rag = use_rag
        self.rag = MarketRAG() if use_rag else None

        # Track research results for reporting
        self.research_cache: Dict[str, EnhancedResearchResult] = {}
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_market(
        self,
        market: Market,
        deep_research: bool = True,
        analyze_orderbook: bool = True,
    ) -> Optional[TradeOpportunity]:
        """Run enhanced swarm analysis on a single market.

        Args:
            market: Market data from Polymarket API
            deep_research: Whether to use Perplexity deep research
            analyze_orderbook: Whether to analyze order book liquidity

        Returns:
            TradeOpportunity if opportunity found, None otherwise
        """
        # Perform enhanced research
        research_result = self.research.research_market(
            market,
            use_perplexity=deep_research and self.use_perplexity,
            use_news_api=True,
        )

        # Cache for reporting
        self.research_cache[str(market.id)] = research_result

        # Analyze order book if enabled
        orderbook_summary = None
        if analyze_orderbook and self.orderbook and market.clobTokenIds:
            orderbook_summary = self.orderbook.get_market_summary(
                market.clobTokenIds,
                market.question or ""
            )
            self.orderbook_cache[str(market.id)] = orderbook_summary

            # Skip if order book is unhealthy (unless we have strong edge)
            if not orderbook_summary.get("healthy", True):
                # Allow if edge is very strong (>15%)
                if abs(research_result.edge_estimate) < 15:
                    return None

        # Build context for agents with enhanced data
        context = {
            "news_articles": research_result.news_articles,
            "sentiment_score": research_result.combined_sentiment,  # Use combined sentiment
            # Enhanced context from Perplexity
            "perplexity_analysis": research_result.perplexity_analysis,
            "perplexity_citations": research_result.perplexity_citations,
            "perplexity_confidence": research_result.perplexity_confidence,
            "key_factors": research_result.key_factors,
            "estimated_probability": research_result.estimated_probability,
            "research_edge": research_result.edge_estimate,
            # Order book context
            "orderbook_summary": orderbook_summary,
            "orderbook_healthy": orderbook_summary.get("healthy", True) if orderbook_summary else True,
            "orderbook_spread": orderbook_summary.get("average_spread_percent", 0) if orderbook_summary else 0,
        }

        # Run agents in sequence
        results: List[AgentResult] = []

        # 1. Market Scanner - analyzes market structure
        scanner_result = self.agents["market_scanner"].analyze(market, context)
        results.append(scanner_result)
        context["market_quality"] = scanner_result.data_points.get("market_quality", 0.5)

        # 2. News Researcher - now with enhanced sentiment
        news_result = self.agents["news_researcher"].analyze(market, context)
        results.append(news_result)

        # 3. Probability Analyst - use research-derived probability if available
        if research_result.estimated_probability is not None:
            # Override with research-based estimate
            context["research_probability"] = research_result.estimated_probability

        prob_result = self._enhanced_probability_analysis(market, context, research_result)
        results.append(prob_result)

        # Update context for risk validator
        context["other_recommendations"] = [r.recommendation for r in results]
        context["probability_recommendation"] = prob_result.recommendation
        context["probability_edge"] = prob_result.edge_estimate

        # 4. Risk Validator
        risk_result = self.agents["risk_validator"].analyze(market, context)
        results.append(risk_result)

        # 5. Coordinator
        context["agent_results"] = results
        final_result = self.agents["coordinator"].analyze(market, context)

        # Check thresholds
        if final_result.recommendation == Recommendation.SKIP:
            return None

        if final_result.confidence < MIN_CONFIDENCE:
            return None

        if final_result.edge_estimate < MIN_EDGE_PERCENT:
            return None

        # Build opportunity with enhanced data
        agent_votes = {r.agent_name: r.recommendation for r in results}
        risk_factors = risk_result.data_points.get("risks", [])

        # Use research-derived fair price
        estimated_fair_yes = research_result.estimated_probability or market.yes_price

        # Build comprehensive news summary
        news_summary = self._build_news_summary(research_result)

        return TradeOpportunity(
            market_id=market.id,
            market_question=market.question or "Unknown",
            condition_id=market.conditionId,
            clob_token_ids=market.clobTokenIds,
            recommendation=final_result.recommendation,
            confidence=final_result.confidence,
            edge_percent=final_result.edge_estimate,
            current_yes_price=market.yes_price,
            current_no_price=market.no_price,
            estimated_fair_yes_price=estimated_fair_yes,
            reasoning=final_result.reasoning,
            agent_votes=agent_votes,
            risk_factors=risk_factors,
            news_summary=news_summary,
            volume=market.get_volume_safe(),
            liquidity=market.get_liquidity_safe(),
        )

    def _enhanced_probability_analysis(
        self,
        market: Market,
        context: Dict[str, Any],
        research: EnhancedResearchResult,
    ) -> AgentResult:
        """Enhanced probability analysis using Perplexity research."""

        # If we have high-confidence Perplexity research, use it
        if research.perplexity_confidence > 0.7 and research.estimated_probability:
            market_prob = market.yes_price
            estimated_prob = research.estimated_probability
            edge = (estimated_prob - market_prob) * 100

            # Higher confidence from deep research
            confidence = min(0.6 + research.perplexity_confidence * 0.3, 0.95)

            if abs(edge) >= MIN_EDGE_PERCENT:
                recommendation = Recommendation.YES if edge > 0 else Recommendation.NO

                return AgentResult(
                    agent_name="ProbabilityAnalyst",
                    cognitive_pattern=self.agents["probability_analyst"].cognitive_pattern,
                    confidence=confidence,
                    recommendation=recommendation,
                    edge_estimate=abs(edge),
                    reasoning=f"Deep research: Market {market_prob:.1%}, Estimated {estimated_prob:.1%}, Edge {edge:+.1f}% (Perplexity confidence: {research.perplexity_confidence:.1%})",
                    data_points={
                        "market_probability": market_prob,
                        "estimated_probability": estimated_prob,
                        "edge_percent": edge,
                        "perplexity_confidence": research.perplexity_confidence,
                        "agent_weight": self.agents["probability_analyst"].weight,
                        "research_enhanced": True,
                    }
                )

        # Fall back to standard probability analysis
        return self.agents["probability_analyst"].analyze(market, context)

    def _build_news_summary(self, research: EnhancedResearchResult) -> str:
        """Build comprehensive news summary from research."""
        parts = []

        # Key factors from Perplexity
        if research.key_factors:
            parts.append("Key factors: " + "; ".join(research.key_factors[:3]))

        # Top news articles
        if research.news_articles:
            top_articles = sorted(
                research.news_articles,
                key=lambda a: a.relevance_score,
                reverse=True
            )[:2]
            titles = [a.title for a in top_articles if a.title]
            if titles:
                parts.append("News: " + "; ".join(titles))

        # Perplexity citations
        if research.perplexity_citations:
            parts.append(f"Sources: {len(research.perplexity_citations)} citations")

        return " | ".join(parts) if parts else "No news data"

    def find_opportunities(
        self,
        max_markets: int = 50,
        search_query: Optional[str] = None,
        min_edge: float = MIN_EDGE_PERCENT,
        min_confidence: float = MIN_CONFIDENCE,
        deep_research: bool = True,
    ) -> List[TradeOpportunity]:
        """Scan markets with enhanced research.

        Args:
            max_markets: Maximum markets to analyze
            search_query: Optional semantic search query
            min_edge: Minimum edge percentage
            min_confidence: Minimum confidence score
            deep_research: Use Perplexity for top candidates

        Returns:
            List of trading opportunities
        """
        # Get markets
        if self.use_rag and self.rag and search_query:
            all_markets = self.polymarket.fetch_filtered_markets(limit=200)
            self.rag.index_markets(all_markets)
            markets = self.rag.search(search_query, top_k=max_markets)
            print(f"RAG found {len(markets)} relevant markets for '{search_query}'")
        else:
            markets = self.polymarket.fetch_filtered_markets(limit=max_markets)

        opportunities = []

        print(f"\nPerplexity deep research: {'ENABLED' if self.use_perplexity else 'DISABLED'}")
        print(f"Analyzing {len(markets)} markets...\n")

        for i, market in enumerate(markets):
            question = market.question[:50] if market.question else "Unknown"
            print(f"[{i+1}/{len(markets)}] {question}...")

            try:
                # Use deep research for all markets if available
                opportunity = self.analyze_market(
                    market,
                    deep_research=deep_research,
                )

                if opportunity:
                    if opportunity.edge_percent >= min_edge and opportunity.confidence >= min_confidence:
                        opportunities.append(opportunity)
                        print(f"  -> FOUND: {opportunity.recommendation.value} ({opportunity.edge_percent:.1f}% edge, {opportunity.confidence:.1%} conf)")
                    else:
                        print(f"  -> Below threshold (edge: {opportunity.edge_percent:.1f}%, conf: {opportunity.confidence:.1%})")
                else:
                    print(f"  -> SKIP")

            except Exception as e:
                print(f"  -> Error: {e}")
                continue

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        return opportunities

    def generate_report(
        self,
        opportunities: List[TradeOpportunity],
        include_research: bool = True,
        include_orderbook: bool = True,
    ) -> str:
        """Generate enhanced report with research and order book details."""
        if not opportunities:
            return "No trading opportunities found matching criteria."

        lines = [
            "=" * 70,
            "POLYMARKET SWARM TRADER v2.1 - ENHANCED REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Deep Research: {'ENABLED' if self.use_perplexity else 'DISABLED'}",
            f"Order Book Analysis: {'ENABLED' if self.use_orderbook else 'DISABLED'}",
            "=" * 70,
            "",
            f"Found {len(opportunities)} opportunities:",
            ""
        ]

        for i, opp in enumerate(opportunities, 1):
            lines.extend([
                f"{'='*50}",
                f"OPPORTUNITY #{i}",
                f"{'='*50}",
                f"Market: {opp.market_question}",
                f"Market ID: {opp.market_id}",
                f"",
                f">>> RECOMMENDATION: {opp.recommendation.value} <<<",
                f"Confidence: {opp.confidence:.1%}",
                f"Edge: {opp.edge_percent:.1f}%",
                f"",
                f"Prices:",
                f"  Current YES: {opp.current_yes_price:.1%}",
                f"  Current NO: {opp.current_no_price:.1%}",
                f"  Fair Value: {opp.estimated_fair_yes_price:.1%}",
                f"",
                f"Market Metrics:",
                f"  Volume: ${opp.volume:,.0f}",
                f"  Liquidity: ${opp.liquidity:,.0f}",
                f"",
                f"Analysis: {opp.reasoning}",
                f"",
                f"Agent Votes:",
            ])

            for agent, vote in opp.agent_votes.items():
                vote_str = vote.value if hasattr(vote, 'value') else str(vote)
                lines.append(f"  - {agent}: {vote_str}")

            lines.append("")

            if opp.risk_factors:
                lines.append(f"Risk Factors: {', '.join(opp.risk_factors)}")
            else:
                lines.append("Risk Factors: None identified")

            lines.append("")

            # Include research details if available
            if include_research and str(opp.market_id) in self.research_cache:
                research = self.research_cache[str(opp.market_id)]
                lines.extend([
                    "Research Summary:",
                    f"  Sources: {', '.join(research.sources_used)}",
                    f"  Combined Sentiment: {research.combined_sentiment:.2f}",
                ])

                if research.perplexity_citations:
                    lines.append(f"  Citations: {len(research.perplexity_citations)}")
                    for cite in research.perplexity_citations[:3]:
                        lines.append(f"    - {cite[:80]}...")

                if research.key_factors:
                    lines.append("  Key Factors:")
                    for factor in research.key_factors[:3]:
                        lines.append(f"    - {factor[:80]}...")

            # Include order book data if available
            if include_orderbook and str(opp.market_id) in self.orderbook_cache:
                ob = self.orderbook_cache[str(opp.market_id)]
                lines.extend([
                    "",
                    "Order Book Analysis:",
                    f"  Healthy: {'YES' if ob.get('healthy') else 'NO'}",
                    f"  Avg Spread: {ob.get('average_spread_percent', 0):.2f}%",
                    f"  Liquidity Score: {ob.get('average_liquidity_score', 0):.2f}",
                    f"  Trading Rec: {ob.get('recommendation', 'N/A')}",
                ])

                if ob.get('yes_analysis'):
                    yes = ob['yes_analysis']
                    lines.append(f"  YES Book: Bid {yes.get('best_bid', 0):.3f} / Ask {yes.get('best_ask', 0):.3f} (spread {yes.get('spread_percent', 0):.1f}%)")
                    lines.append(f"    Depth: {yes.get('bid_depth_5', 0):.0f} bid / {yes.get('ask_depth_5', 0):.0f} ask")
                    lines.append(f"    Slippage (500 shares): {yes.get('slippage_500', 0):.2f}%")

                if ob.get('issues'):
                    lines.append(f"  Issues: {'; '.join(ob['issues'])}")

            lines.extend([
                "",
                f"Executable: {'YES' if opp.is_executable else 'NO'}",
                f"CLOB Tokens: {opp.clob_token_ids}",
                "",
            ])

            if opp.news_summary:
                lines.append(f"News: {opp.news_summary[:300]}...")
                lines.append("")

        return "\n".join(lines)

    def to_json(self, opportunities: List[TradeOpportunity]) -> str:
        """Convert opportunities to JSON with research and order book data."""
        data = []
        for opp in opportunities:
            opp_data = opp.model_dump(mode='json')

            # Add research data if available
            if str(opp.market_id) in self.research_cache:
                research = self.research_cache[str(opp.market_id)]
                opp_data["research"] = research.to_dict()

            # Add order book data if available
            if str(opp.market_id) in self.orderbook_cache:
                opp_data["orderbook"] = self.orderbook_cache[str(opp.market_id)]

            data.append(opp_data)

        return json.dumps(data, indent=2, default=str)

    def close(self):
        """Close all clients."""
        self.polymarket.close()
        self.research.close()
        if self.orderbook:
            self.orderbook.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("POLYMARKET SWARM TRADER v2.0 - ENHANCED RESEARCH MODE")
    print("=" * 70)
    print()

    with EnhancedSwarmOrchestrator(use_rag=False, use_perplexity=True) as orchestrator:
        print(f"Perplexity API: {'AVAILABLE' if orchestrator.use_perplexity else 'NOT CONFIGURED'}")
        print()

        # Find opportunities with deep research
        opportunities = orchestrator.find_opportunities(
            max_markets=30,
            min_edge=8,  # Lower threshold to catch more
            min_confidence=0.6,
            deep_research=True,
        )

        # Generate report
        print()
        print(orchestrator.generate_report(opportunities))

        # Save to file
        if opportunities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"polymarket_enhanced_{timestamp}.json"
            with open(filename, "w") as f:
                f.write(orchestrator.to_json(opportunities))
            print(f"\nResults saved to: {filename}")
