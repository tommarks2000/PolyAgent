"""Enhanced research module combining multiple sources.

Integrates Perplexity deep research with traditional news APIs
for comprehensive market analysis.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from models import Market, Article
from news_client import NewsClient
from perplexity_client import PerplexityClient, ResearchResult


@dataclass
class EnhancedResearchResult:
    """Combined research result from all sources."""
    market_id: str
    market_question: str

    # Perplexity deep research
    perplexity_analysis: Optional[str] = None
    perplexity_citations: List[str] = field(default_factory=list)
    perplexity_confidence: float = 0.0

    # Traditional news
    news_articles: List[Article] = field(default_factory=list)
    news_sentiment: float = 0.0

    # Combined analysis
    combined_sentiment: float = 0.0
    estimated_probability: Optional[float] = None
    edge_estimate: float = 0.0
    key_factors: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # Metadata
    research_timestamp: str = ""
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "perplexity_analysis": self.perplexity_analysis,
            "perplexity_citations": self.perplexity_citations,
            "perplexity_confidence": self.perplexity_confidence,
            "news_article_count": len(self.news_articles),
            "news_sentiment": self.news_sentiment,
            "combined_sentiment": self.combined_sentiment,
            "estimated_probability": self.estimated_probability,
            "edge_estimate": self.edge_estimate,
            "key_factors": self.key_factors,
            "risks": self.risks,
            "sources_used": self.sources_used,
            "research_timestamp": self.research_timestamp,
        }


class EnhancedResearchClient:
    """Combined research client using multiple sources."""

    def __init__(self):
        self.news_client = NewsClient()
        self.perplexity_client = PerplexityClient()

    @property
    def has_perplexity(self) -> bool:
        """Check if Perplexity API is available."""
        return self.perplexity_client.is_available

    def research_market(
        self,
        market: Market,
        use_perplexity: bool = True,
        use_news_api: bool = True,
    ) -> EnhancedResearchResult:
        """Perform comprehensive research on a market.

        Args:
            market: Market to research
            use_perplexity: Whether to use Perplexity deep research
            use_news_api: Whether to use traditional news APIs

        Returns:
            EnhancedResearchResult with combined analysis
        """
        result = EnhancedResearchResult(
            market_id=str(market.id),
            market_question=market.question or "",
            research_timestamp=datetime.now().isoformat(),
        )

        # Get traditional news
        if use_news_api:
            articles = self.news_client.get_news_for_market(market)
            result.news_articles = articles
            result.news_sentiment = self.news_client.calculate_sentiment(articles)
            result.sources_used.append("NewsAPI/Finnhub")

        # Get Perplexity deep research
        if use_perplexity and self.has_perplexity:
            perplexity_result = self.perplexity_client.research_market(
                market_question=market.question or "",
                market_description=market.description,
                current_price=market.yes_price,
            )

            if perplexity_result:
                result.perplexity_analysis = perplexity_result.content
                result.perplexity_citations = perplexity_result.citations
                result.perplexity_confidence = perplexity_result.confidence
                result.sources_used.append("Perplexity")

                # Extract key factors from Perplexity analysis
                result.key_factors = self._extract_key_factors(
                    perplexity_result.content
                )

        # Combine sentiment signals
        result.combined_sentiment = self._combine_sentiments(result)

        # Estimate probability
        result.estimated_probability = self._estimate_probability(
            market, result
        )

        # Calculate edge
        if result.estimated_probability is not None:
            result.edge_estimate = (
                result.estimated_probability - market.yes_price
            ) * 100

        return result

    def _combine_sentiments(
        self,
        result: EnhancedResearchResult
    ) -> float:
        """Combine sentiment signals from multiple sources.

        Perplexity gets higher weight due to deeper analysis.
        """
        sentiments = []
        weights = []

        # News sentiment (lower weight)
        if result.news_articles:
            sentiments.append(result.news_sentiment)
            weights.append(1.0)

        # Perplexity sentiment (higher weight if available)
        if result.perplexity_analysis:
            # Extract sentiment from Perplexity analysis
            perplexity_sentiment = self._extract_sentiment_from_analysis(
                result.perplexity_analysis
            )
            if perplexity_sentiment != 0:
                sentiments.append(perplexity_sentiment)
                weights.append(2.0)  # Double weight for deep research

        if not sentiments:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
        return weighted_sum / total_weight

    def _extract_sentiment_from_analysis(self, analysis: str) -> float:
        """Extract sentiment score from Perplexity analysis text."""
        analysis_lower = analysis.lower()

        # Strong positive signals
        strong_positive = [
            "very likely", "highly probable", "strong evidence",
            "expected to", "will likely", "almost certain",
            "significant progress", "breakthrough", "confirmed"
        ]

        # Moderate positive
        moderate_positive = [
            "likely", "probable", "possible", "trending toward",
            "momentum", "progress", "improving", "positive signs"
        ]

        # Strong negative
        strong_negative = [
            "very unlikely", "highly improbable", "no evidence",
            "failed", "rejected", "impossible", "no chance",
            "collapsed", "abandoned"
        ]

        # Moderate negative
        moderate_negative = [
            "unlikely", "improbable", "stalled", "delayed",
            "obstacles", "challenges", "concerns", "doubts"
        ]

        score = 0.0

        for phrase in strong_positive:
            if phrase in analysis_lower:
                score += 0.4

        for phrase in moderate_positive:
            if phrase in analysis_lower:
                score += 0.15

        for phrase in strong_negative:
            if phrase in analysis_lower:
                score -= 0.4

        for phrase in moderate_negative:
            if phrase in analysis_lower:
                score -= 0.15

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, score))

    def _extract_key_factors(self, analysis: str) -> List[str]:
        """Extract key factors from Perplexity analysis."""
        factors = []

        # Look for numbered lists or bullet points
        lines = analysis.split("\n")
        for line in lines:
            line = line.strip()
            # Check for numbered items or bullets
            if (line and
                (line[0].isdigit() or line.startswith("-") or line.startswith("•"))):
                # Clean up the line
                clean = line.lstrip("0123456789.-•) ").strip()
                if len(clean) > 10 and len(clean) < 200:
                    factors.append(clean)

        return factors[:5]  # Top 5 factors

    def _estimate_probability(
        self,
        market: Market,
        result: EnhancedResearchResult
    ) -> Optional[float]:
        """Estimate fair probability using research results.

        Uses a Bayesian-style update from market price based on evidence.
        """
        # Start with market price as base (wisdom of crowds)
        base_prob = market.yes_price

        # Adjustment factors
        adjustment = 0.0

        # Sentiment-based adjustment (up to 20%)
        sentiment_adjustment = result.combined_sentiment * 0.20
        adjustment += sentiment_adjustment

        # Perplexity confidence boost
        if result.perplexity_confidence > 0.7:
            # High confidence Perplexity research amplifies adjustment
            adjustment *= (1 + (result.perplexity_confidence - 0.7))

        # Apply adjustment
        estimated = base_prob + adjustment

        # Clamp to reasonable range
        estimated = max(0.02, min(0.98, estimated))

        return estimated

    def get_sentiment_analysis(
        self,
        topic: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get sentiment analysis for a topic.

        Args:
            topic: Topic to analyze
            context: Optional additional context

        Returns:
            Dict with sentiment analysis
        """
        if self.has_perplexity:
            return self.perplexity_client.analyze_sentiment(topic, context) or {
                "sentiment_score": 0.0,
                "direction": "neutral",
                "analysis": "Perplexity analysis unavailable",
                "citations": [],
                "confidence": 0.0,
            }

        # Fallback to basic news sentiment
        articles = self.news_client.search_news(topic, limit=10)
        sentiment = self.news_client.calculate_sentiment(articles)

        return {
            "sentiment_score": sentiment,
            "direction": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral",
            "analysis": f"Based on {len(articles)} news articles",
            "citations": [a.url for a in articles if a.url],
            "confidence": 0.5,
        }

    def fact_check_market(
        self,
        market: Market,
    ) -> Dict[str, Any]:
        """Fact-check a market's question.

        Args:
            market: Market to fact-check

        Returns:
            Dict with verification results
        """
        if not self.has_perplexity:
            return {
                "verdict": "UNVERIFIED",
                "probability": market.yes_price,
                "analysis": "Deep fact-checking requires Perplexity API",
                "citations": [],
                "confidence": 0.0,
            }

        return self.perplexity_client.fact_check(
            claim=market.question or "",
            deadline=market.endDate,
        ) or {
            "verdict": "UNVERIFIED",
            "probability": market.yes_price,
            "analysis": "Fact-check failed",
            "citations": [],
            "confidence": 0.0,
        }

    def close(self):
        """Close all clients."""
        self.news_client.close()
        self.perplexity_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    from polymarket_client import PolymarketClient

    print("Testing Enhanced Research Client...")

    # Fetch a market to test
    with PolymarketClient() as pm_client:
        markets = pm_client.fetch_tradeable_markets(limit=10)

    if not markets:
        print("No markets found")
        exit(1)

    # Test with first non-crypto market
    test_market = None
    for m in markets:
        q = (m.question or "").lower()
        if not any(x in q for x in ["bitcoin", "ethereum", "crypto"]):
            test_market = m
            break

    if not test_market:
        test_market = markets[0]

    print(f"\nTesting with market: {test_market.question}")
    print(f"Current price: {test_market.yes_price:.1%}")

    with EnhancedResearchClient() as client:
        print(f"\nPerplexity available: {client.has_perplexity}")

        result = client.research_market(test_market)

        print(f"\n{'='*60}")
        print("ENHANCED RESEARCH RESULT")
        print(f"{'='*60}")
        print(f"Sources used: {result.sources_used}")
        print(f"News articles: {len(result.news_articles)}")
        print(f"News sentiment: {result.news_sentiment:.2f}")
        print(f"Combined sentiment: {result.combined_sentiment:.2f}")
        print(f"Estimated probability: {result.estimated_probability:.1%}" if result.estimated_probability else "N/A")
        print(f"Edge estimate: {result.edge_estimate:+.1f}%")

        if result.perplexity_analysis:
            print(f"\nPerplexity Analysis Preview:")
            print(result.perplexity_analysis[:500] + "...")
            print(f"\nCitations: {len(result.perplexity_citations)}")

        if result.key_factors:
            print(f"\nKey Factors:")
            for factor in result.key_factors:
                print(f"  - {factor}")
