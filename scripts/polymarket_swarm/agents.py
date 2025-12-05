"""Swarm agent definitions for Polymarket analysis.

Each agent uses a different cognitive pattern for diverse analysis.
Based on research showing cognitive diversity improves group decisions.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from models import (
    Market, Article, AgentResult,
    CognitivePattern, Recommendation
)
from config import (
    AGENT_WEIGHTS,
    MIN_VOLUME_USD,
    MIN_LIQUIDITY_USD,
    MIN_EDGE_PERCENT,
)


class BaseAgent(ABC):
    """Base class for all swarm agents."""

    def __init__(self):
        self.name: str = "BaseAgent"
        self.cognitive_pattern: CognitivePattern = CognitivePattern.CONVERGENT
        self.weight: float = 1.0

    @abstractmethod
    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze a market and return recommendation.

        Args:
            market: Market data from Polymarket API
            context: Additional context (news, prices, etc.)

        Returns:
            AgentResult with recommendation
        """
        pass


class MarketScannerAgent(BaseAgent):
    """Scans markets for opportunities using convergent thinking.

    Focuses on: volume, liquidity, price movements, market structure.
    Convergent thinking: narrows down to specific data points.
    """

    def __init__(self):
        super().__init__()
        self.name = "MarketScanner"
        self.cognitive_pattern = CognitivePattern.CONVERGENT
        self.weight = AGENT_WEIGHTS.get("market_scanner", 1.0)

    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze market structure and trading metrics."""

        volume = market.get_volume_safe()
        liquidity = market.get_liquidity_safe()
        yes_price = market.yes_price

        # Calculate market quality score
        volume_score = min(volume / 100000, 1.0)  # Normalize to 100k
        liquidity_score = min(liquidity / 50000, 1.0)  # Normalize to 50k
        market_quality = (volume_score + liquidity_score) / 2

        # Check for extreme prices (potential opportunities)
        price_extremity = abs(yes_price - 0.5) * 2  # 0 at 50%, 1 at extremes

        # Data points for other agents
        data_points = {
            "volume": volume,
            "liquidity": liquidity,
            "yes_price": yes_price,
            "market_quality": market_quality,
            "price_extremity": price_extremity,
            "agent_weight": self.weight,
        }

        # Low quality markets
        if market_quality < 0.1:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=0.3,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning=f"Insufficient volume (${volume:,.0f}) or liquidity (${liquidity:,.0f})",
                data_points=data_points
            )

        # Extreme prices might indicate opportunity
        if price_extremity > 0.6:  # Price < 0.2 or > 0.8
            # Recommend the underpriced side
            recommendation = Recommendation.YES if yes_price < 0.5 else Recommendation.NO
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=market_quality,
                recommendation=recommendation,
                edge_estimate=price_extremity * 20,  # Up to 20%
                reasoning=f"Extreme price ({yes_price:.1%}) with good liquidity - potential mispricing",
                data_points=data_points
            )

        return AgentResult(
            agent_name=self.name,
            cognitive_pattern=self.cognitive_pattern,
            confidence=market_quality,
            recommendation=Recommendation.SKIP,
            edge_estimate=0,
            reasoning="No clear market structure opportunity",
            data_points=data_points
        )


class NewsResearcherAgent(BaseAgent):
    """Researches news and events using divergent thinking.

    Explores multiple sources and perspectives.
    Divergent thinking: expands to find diverse information.
    """

    def __init__(self):
        super().__init__()
        self.name = "NewsResearcher"
        self.cognitive_pattern = CognitivePattern.DIVERGENT
        self.weight = AGENT_WEIGHTS.get("news_researcher", 1.2)

    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze news and real-world events."""

        articles: List[Article] = context.get("news_articles", [])
        sentiment = context.get("sentiment_score", 0)

        data_points = {
            "article_count": len(articles),
            "sentiment_score": sentiment,
            "agent_weight": self.weight,
        }

        if not articles:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=0.2,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning="No relevant news found - insufficient information",
                data_points=data_points
            )

        # Count high-relevance articles
        high_relevance = [a for a in articles if a.relevance_score > 0.5]
        data_points["high_relevance_count"] = len(high_relevance)

        # Strong sentiment signals direction
        if abs(sentiment) > 0.3:
            recommendation = Recommendation.YES if sentiment > 0 else Recommendation.NO
            confidence = min(0.5 + abs(sentiment), 0.9)
            edge = abs(sentiment) * 15  # Up to 15% edge

            sentiment_desc = "positive" if sentiment > 0 else "negative"

            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=confidence,
                recommendation=recommendation,
                edge_estimate=edge,
                reasoning=f"News sentiment strongly {sentiment_desc} ({sentiment:.2f}) across {len(articles)} articles",
                data_points=data_points
            )

        return AgentResult(
            agent_name=self.name,
            cognitive_pattern=self.cognitive_pattern,
            confidence=0.4,
            recommendation=Recommendation.SKIP,
            edge_estimate=0,
            reasoning=f"Neutral news sentiment ({sentiment:.2f}) - no clear directional signal",
            data_points=data_points
        )


class ProbabilityAnalystAgent(BaseAgent):
    """Analyzes true probability using critical thinking.

    Compares market price to estimated fair value.
    Uses Superforecaster methodology from Polymarket/agents.
    Critical thinking: questions assumptions, seeks evidence.
    """

    def __init__(self):
        super().__init__()
        self.name = "ProbabilityAnalyst"
        self.cognitive_pattern = CognitivePattern.CRITICAL
        self.weight = AGENT_WEIGHTS.get("probability_analyst", 1.5)

    def _superforecast_prompt(
        self,
        question: str,
        description: str,
        outcome: str
    ) -> str:
        """Generate Superforecaster analysis prompt.

        Based on Polymarket/agents prompts.py methodology.
        """
        return f"""
        You are a Superforecaster tasked with correctly predicting the likelihood of events.
        Use the following systematic process:

        1. Breaking Down the Question:
           - Decompose into smaller, manageable parts
           - Identify key components that need to be addressed

        2. Consider Base Rates:
           - Use statistical baselines or historical averages
           - Compare to similar past events

        3. Identify and Evaluate Factors:
           - List factors that could influence the outcome
           - Assess impact of each factor with evidence
           - Avoid over-reliance on any single piece of information

        4. Think Probabilistically:
           - Express predictions as probabilities, not certainties
           - Embrace uncertainty

        Question: {question}
        Description: {description}

        Estimate probability for outcome: {outcome}
        """

    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Calculate fair probability and identify mispricing."""

        market_prob = market.yes_price
        news_sentiment = context.get("sentiment_score", 0)
        market_quality = context.get("market_quality", 0.5)

        # Base probability estimation
        # Start with market price as base (wisdom of crowds)
        estimated_prob = market_prob

        # Adjust based on news sentiment
        # Strong sentiment suggests market may not have incorporated news
        sentiment_adjustment = news_sentiment * 0.15  # Up to 15% adjustment
        estimated_prob += sentiment_adjustment

        # Clamp to valid range
        estimated_prob = max(0.05, min(0.95, estimated_prob))

        # Calculate edge
        edge = (estimated_prob - market_prob) * 100  # As percentage

        data_points = {
            "market_probability": market_prob,
            "estimated_probability": estimated_prob,
            "edge_percent": edge,
            "sentiment_adjustment": sentiment_adjustment,
            "agent_weight": self.weight,
        }

        if abs(edge) >= MIN_EDGE_PERCENT:
            recommendation = Recommendation.YES if edge > 0 else Recommendation.NO
            confidence = min(0.5 + abs(edge) / 50, 0.95)

            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=confidence,
                recommendation=recommendation,
                edge_estimate=abs(edge),
                reasoning=f"Market: {market_prob:.1%}, Estimated: {estimated_prob:.1%}, Edge: {edge:+.1f}%",
                data_points=data_points
            )

        return AgentResult(
            agent_name=self.name,
            cognitive_pattern=self.cognitive_pattern,
            confidence=0.5,
            recommendation=Recommendation.SKIP,
            edge_estimate=abs(edge),
            reasoning=f"Insufficient edge ({edge:+.1f}%) - market fairly priced",
            data_points=data_points
        )


class RiskValidatorAgent(BaseAgent):
    """Validates trades using systems thinking.

    Considers market risks, liquidity, timing, position sizing.
    Systems thinking: sees interconnections and feedback loops.
    """

    def __init__(self):
        super().__init__()
        self.name = "RiskValidator"
        self.cognitive_pattern = CognitivePattern.SYSTEMS
        self.weight = AGENT_WEIGHTS.get("risk_validator", 1.3)

    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Validate trade from risk perspective."""

        volume = market.get_volume_safe()
        liquidity = market.get_liquidity_safe()

        # Collect risk factors
        risks = []
        risk_score = 0

        # Low liquidity risk - slippage concern
        if liquidity < MIN_LIQUIDITY_USD:
            risks.append(f"Low liquidity (${liquidity:,.0f}) - slippage risk")
            risk_score += 0.3
        elif liquidity < MIN_LIQUIDITY_USD * 2:
            risks.append(f"Moderate liquidity (${liquidity:,.0f})")
            risk_score += 0.1

        # Low volume risk - price discovery concern
        if volume < MIN_VOLUME_USD:
            risks.append(f"Low volume (${volume:,.0f}) - limited price discovery")
            risk_score += 0.2

        # Negative risk markets have complex resolution
        if market.negRisk:
            risks.append("Negative risk market - complex resolution rules")
            risk_score += 0.15

        # Check if market is actually tradeable
        if not market.is_tradeable:
            risks.append("Market not tradeable - no order book")
            risk_score += 0.5

        # Check CLOB token IDs exist
        if not market.clobTokenIds:
            risks.append("Missing CLOB token IDs - cannot execute")
            risk_score += 0.5

        # Get other agents' recommendations
        other_recommendations = context.get("other_recommendations", [])
        positive_votes = sum(
            1 for r in other_recommendations
            if r in [Recommendation.YES, Recommendation.NO]
        )

        prob_recommendation = context.get("probability_recommendation", Recommendation.SKIP)
        prob_edge = context.get("probability_edge", 0)

        data_points = {
            "risks": risks,
            "risk_score": risk_score,
            "positive_votes": positive_votes,
            "agent_weight": self.weight,
        }

        # No positive signals from other agents
        if positive_votes == 0:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=0.3,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning="No positive signals from other agents",
                data_points=data_points
            )

        # High risk - reject
        if risk_score > 0.5:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=1 - risk_score,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning=f"High risk ({risk_score:.1%}): {'; '.join(risks)}",
                data_points=data_points
            )

        # Pass through probability recommendation with risk-adjusted edge
        adjusted_edge = prob_edge * (1 - risk_score)

        return AgentResult(
            agent_name=self.name,
            cognitive_pattern=self.cognitive_pattern,
            confidence=0.7 - risk_score,
            recommendation=prob_recommendation,
            edge_estimate=adjusted_edge,
            reasoning=f"Risk validated (score: {risk_score:.1%}). Risks: {'; '.join(risks) if risks else 'None'}",
            data_points=data_points
        )


class CoordinatorAgent(BaseAgent):
    """Coordinates all agents using adaptive thinking.

    Synthesizes results and makes final recommendation.
    Adaptive thinking: adjusts approach based on inputs.
    """

    def __init__(self):
        super().__init__()
        self.name = "Coordinator"
        self.cognitive_pattern = CognitivePattern.ADAPTIVE
        self.weight = AGENT_WEIGHTS.get("coordinator", 1.0)

    def analyze(
        self,
        market: Market,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Synthesize all agent results into final recommendation."""

        agent_results: List[AgentResult] = context.get("agent_results", [])

        if not agent_results:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=0,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning="No agent results to coordinate",
                data_points={}
            )

        # Weighted voting
        yes_score = 0
        no_score = 0
        skip_score = 0
        total_weight = 0
        edges = []

        for result in agent_results:
            weight = result.data_points.get("agent_weight", 1.0)

            if result.recommendation == Recommendation.YES:
                yes_score += result.confidence * weight
                edges.append(result.edge_estimate)
            elif result.recommendation == Recommendation.NO:
                no_score += result.confidence * weight
                edges.append(result.edge_estimate)
            else:
                skip_score += result.confidence * weight

            total_weight += weight

        # Normalize
        if total_weight > 0:
            yes_score /= total_weight
            no_score /= total_weight
            skip_score /= total_weight

        data_points = {
            "yes_score": yes_score,
            "no_score": no_score,
            "skip_score": skip_score,
            "agent_count": len(agent_results),
            "agent_weight": self.weight,
        }

        # Determine final recommendation
        max_score = max(yes_score, no_score, skip_score)

        # Insufficient consensus
        if max_score == skip_score or max_score < 0.4:
            return AgentResult(
                agent_name=self.name,
                cognitive_pattern=self.cognitive_pattern,
                confidence=max_score,
                recommendation=Recommendation.SKIP,
                edge_estimate=0,
                reasoning=f"Insufficient consensus (YES: {yes_score:.1%}, NO: {no_score:.1%}, SKIP: {skip_score:.1%})",
                data_points=data_points
            )

        recommendation = Recommendation.YES if yes_score > no_score else Recommendation.NO
        avg_edge = sum(edges) / len(edges) if edges else 0

        # Count dissent
        dissent = sum(
            1 for r in agent_results
            if r.recommendation != recommendation and r.recommendation != Recommendation.SKIP
        )
        data_points["dissent_count"] = dissent

        return AgentResult(
            agent_name=self.name,
            cognitive_pattern=self.cognitive_pattern,
            confidence=max_score,
            recommendation=recommendation,
            edge_estimate=avg_edge,
            reasoning=f"Swarm consensus: {recommendation.value} (score: {max_score:.1%}, dissent: {dissent}/{len(agent_results)})",
            data_points=data_points
        )


# Agent registry
AGENT_REGISTRY = {
    "market_scanner": MarketScannerAgent,
    "news_researcher": NewsResearcherAgent,
    "probability_analyst": ProbabilityAnalystAgent,
    "risk_validator": RiskValidatorAgent,
    "coordinator": CoordinatorAgent,
}
