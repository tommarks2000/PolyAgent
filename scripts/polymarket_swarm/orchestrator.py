"""Swarm orchestrator for coordinating market analysis.

Coordinates all agents to analyze markets and identify opportunities.
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
from news_client import NewsClient
from market_rag import MarketRAG
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


class SwarmOrchestrator:
    """Orchestrates the swarm of agents to analyze Polymarket opportunities."""

    def __init__(self, use_rag: bool = True):
        """Initialize orchestrator with all components.

        Args:
            use_rag: Whether to use RAG for market selection
        """
        # Initialize agents
        self.agents = {
            name: agent_class()
            for name, agent_class in AGENT_REGISTRY.items()
        }

        # Initialize clients
        self.polymarket = PolymarketClient()
        self.news = NewsClient()

        # Initialize RAG (optional)
        self.use_rag = use_rag
        self.rag = MarketRAG() if use_rag else None

    def analyze_market(self, market: Market) -> Optional[TradeOpportunity]:
        """Run full swarm analysis on a single market.

        Args:
            market: Market data from Polymarket API

        Returns:
            TradeOpportunity if opportunity found, None otherwise
        """
        # Get relevant news
        articles = self.news.get_news_for_market(market)
        sentiment = self.news.calculate_sentiment(articles)

        # Build context for agents
        context = {
            "news_articles": articles,
            "sentiment_score": sentiment,
        }

        # Run agents in sequence (dependencies between them)
        results: List[AgentResult] = []

        # 1. Market Scanner - analyzes market structure
        scanner_result = self.agents["market_scanner"].analyze(market, context)
        results.append(scanner_result)
        context["market_quality"] = scanner_result.data_points.get("market_quality", 0.5)

        # 2. News Researcher - analyzes news
        news_result = self.agents["news_researcher"].analyze(market, context)
        results.append(news_result)

        # 3. Probability Analyst - calculates edge
        prob_result = self.agents["probability_analyst"].analyze(market, context)
        results.append(prob_result)

        # Update context for risk validator
        context["other_recommendations"] = [r.recommendation for r in results]
        context["probability_recommendation"] = prob_result.recommendation
        context["probability_edge"] = prob_result.edge_estimate

        # 4. Risk Validator - validates the trade
        risk_result = self.agents["risk_validator"].analyze(market, context)
        results.append(risk_result)

        # 5. Coordinator - synthesizes final decision
        context["agent_results"] = results
        final_result = self.agents["coordinator"].analyze(market, context)

        # Check if we have an opportunity
        if final_result.recommendation == Recommendation.SKIP:
            return None

        if final_result.confidence < MIN_CONFIDENCE:
            return None

        if final_result.edge_estimate < MIN_EDGE_PERCENT:
            return None

        # Build opportunity
        agent_votes = {r.agent_name: r.recommendation for r in results}

        risk_factors = risk_result.data_points.get("risks", [])

        # Calculate estimated fair price
        prob_data = prob_result.data_points
        estimated_fair_yes = prob_data.get("estimated_probability", market.yes_price)

        # Build news summary
        news_summary = None
        if articles:
            top_articles = sorted(articles, key=lambda a: a.relevance_score, reverse=True)[:3]
            news_summary = "; ".join([a.title for a in top_articles if a.title])

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

    def find_opportunities(
        self,
        max_markets: int = 50,
        search_query: Optional[str] = None,
        min_edge: float = MIN_EDGE_PERCENT,
        min_confidence: float = MIN_CONFIDENCE
    ) -> List[TradeOpportunity]:
        """Scan markets and find trading opportunities.

        Args:
            max_markets: Maximum markets to analyze
            search_query: Optional semantic search query for RAG
            min_edge: Minimum edge percentage to include
            min_confidence: Minimum confidence score

        Returns:
            List of trading opportunities, sorted by edge
        """
        # Get markets
        if self.use_rag and self.rag and search_query:
            # Use RAG for semantic search
            all_markets = self.polymarket.fetch_filtered_markets(limit=200)
            self.rag.index_markets(all_markets)
            markets = self.rag.search(search_query, top_k=max_markets)
            print(f"RAG found {len(markets)} relevant markets for '{search_query}'")
        else:
            # Direct filtered fetch
            markets = self.polymarket.fetch_filtered_markets(limit=max_markets)

        opportunities = []

        for i, market in enumerate(markets):
            question = market.question[:50] if market.question else "Unknown"
            print(f"Analyzing [{i+1}/{len(markets)}]: {question}...")

            try:
                opportunity = self.analyze_market(market)

                if opportunity:
                    if opportunity.edge_percent >= min_edge and opportunity.confidence >= min_confidence:
                        opportunities.append(opportunity)
                        print(f"  -> FOUND: {opportunity.recommendation.value} ({opportunity.edge_percent:.1f}% edge)")
                    else:
                        print(f"  -> Below threshold (edge: {opportunity.edge_percent:.1f}%, conf: {opportunity.confidence:.1%})")
                else:
                    print(f"  -> SKIP")

            except Exception as e:
                print(f"  -> Error: {e}")
                continue

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        return opportunities

    def generate_report(self, opportunities: List[TradeOpportunity]) -> str:
        """Generate a text report of opportunities."""
        if not opportunities:
            return "No trading opportunities found matching criteria."

        lines = [
            "=" * 70,
            "POLYMARKET SWARM TRADER v2.0 - OPPORTUNITY REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            f"Found {len(opportunities)} opportunities:",
            ""
        ]

        for i, opp in enumerate(opportunities, 1):
            lines.extend([
                f"--- Opportunity #{i} ---",
                f"Market: {opp.market_question}",
                f"Market ID: {opp.market_id}",
                f"",
                f"Recommendation: {opp.recommendation.value}",
                f"Confidence: {opp.confidence:.1%}",
                f"Edge: {opp.edge_percent:.1f}%",
                f"",
                f"Current YES Price: {opp.current_yes_price:.1%}",
                f"Current NO Price: {opp.current_no_price:.1%}",
                f"Estimated Fair YES: {opp.estimated_fair_yes_price:.1%}",
                f"",
                f"Volume: ${opp.volume:,.0f}",
                f"Liquidity: ${opp.liquidity:,.0f}",
                f"",
                f"Reasoning: {opp.reasoning}",
                f"Agent Votes: {dict((k, v.value) for k, v in opp.agent_votes.items())}",
                f"Risk Factors: {', '.join(opp.risk_factors) if opp.risk_factors else 'None'}",
                f"",
                f"Executable: {'YES' if opp.is_executable else 'NO'}",
                f"CLOB Tokens: {opp.clob_token_ids}",
                "",
            ])

            if opp.news_summary:
                lines.append(f"News: {opp.news_summary[:200]}...")
                lines.append("")

        return "\n".join(lines)

    def to_json(self, opportunities: List[TradeOpportunity]) -> str:
        """Convert opportunities to JSON."""
        data = [opp.model_dump(mode='json') for opp in opportunities]
        return json.dumps(data, indent=2, default=str)

    def close(self):
        """Close all clients."""
        self.polymarket.close()
        self.news.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("Starting Polymarket Swarm Trader v2.0...")
    print()

    with SwarmOrchestrator(use_rag=False) as orchestrator:
        # Find opportunities
        opportunities = orchestrator.find_opportunities(
            max_markets=20,
            min_edge=10,
            min_confidence=0.5
        )

        # Generate report
        print()
        print(orchestrator.generate_report(opportunities))

        # Save to file
        if opportunities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"polymarket_opportunities_{timestamp}.json"
            with open(filename, "w") as f:
                f.write(orchestrator.to_json(opportunities))
            print(f"\nResults saved to: {filename}")
