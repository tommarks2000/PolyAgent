"""Perplexity Sonar API client for deep research.

Provides real-time web search with citations for market analysis.
Uses OpenAI-compatible API structure for easy integration.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx

from config import OPENAI_API_KEY


# Get Perplexity API key from environment
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")


@dataclass
class ResearchResult:
    """Result from Perplexity research query."""
    content: str
    citations: List[str]
    model: str
    query: str
    timestamp: str
    confidence: float  # Derived from citation count and relevance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "citations": self.citations,
            "model": self.model,
            "query": self.query,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


class PerplexityClient:
    """Client for Perplexity Sonar API with real-time web search."""

    API_BASE = "https://api.perplexity.ai"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Perplexity client.

        Args:
            api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
        """
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.client = httpx.Client(timeout=60.0)

        if not self.api_key:
            print("Warning: No Perplexity API key. Deep research disabled.")

    @property
    def is_available(self) -> bool:
        """Check if Perplexity API is configured."""
        return bool(self.api_key)

    def research(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        model: str = "sonar",
        recency_filter: str = "week",
        search_context_size: str = "high",
        max_tokens: int = 1024,
    ) -> Optional[ResearchResult]:
        """Perform deep research query with web search.

        Args:
            query: The research question
            system_prompt: Optional system instructions
            model: Perplexity model (sonar, sonar-pro)
            recency_filter: Time filter (hour, day, week, month)
            search_context_size: Amount of search context (low, medium, high)
            max_tokens: Maximum response tokens

        Returns:
            ResearchResult with content and citations, or None if failed
        """
        if not self.is_available:
            return None

        if system_prompt is None:
            system_prompt = (
                "You are a research analyst providing factual, well-sourced information. "
                "Focus on recent news, verified facts, and credible sources. "
                "Be precise and cite specific evidence for claims."
            )

        try:
            response = self.client.post(
                f"{self.API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,  # Low for factual responses
                    "search_recency_filter": recency_filter,
                    "web_search_options": {
                        "search_context_size": search_context_size,
                    },
                    "return_citations": True,
                },
            )

            if response.status_code != 200:
                print(f"Perplexity API error: {response.status_code} - {response.text}")
                return None

            data = response.json()

            # Extract content
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            # Extract citations
            citations = data.get("citations", [])

            # Calculate confidence based on citation count
            confidence = min(0.5 + len(citations) * 0.1, 0.95)

            return ResearchResult(
                content=content,
                citations=citations,
                model=model,
                query=query,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
            )

        except Exception as e:
            print(f"Perplexity research error: {e}")
            return None

    def research_market(
        self,
        market_question: str,
        market_description: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> Optional[ResearchResult]:
        """Research a specific prediction market question.

        Args:
            market_question: The market's question
            market_description: Optional additional context
            current_price: Current YES price for context

        Returns:
            ResearchResult with analysis and citations
        """
        # Build research query
        query_parts = [
            f"Research the following prediction market question: {market_question}",
        ]

        if market_description:
            query_parts.append(f"Context: {market_description}")

        if current_price is not None:
            query_parts.append(
                f"The current market probability is {current_price:.1%}. "
                "Is this price accurate based on recent news and evidence?"
            )

        query_parts.extend([
            "",
            "Provide:",
            "1. Recent relevant news and events",
            "2. Key factors that could influence the outcome",
            "3. Your assessment of whether the market price is accurate",
            "4. Any information the market might be missing",
        ])

        system_prompt = (
            "You are a prediction market analyst specializing in identifying mispriced markets. "
            "Focus on recent news, verified facts, and events that could move the market. "
            "Be specific about dates, sources, and confidence levels."
        )

        return self.research(
            query="\n".join(query_parts),
            system_prompt=system_prompt,
            recency_filter="week",
            search_context_size="high",
        )

    def analyze_sentiment(
        self,
        topic: str,
        context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze sentiment around a topic using web research.

        Args:
            topic: Topic to analyze sentiment for
            context: Optional additional context

        Returns:
            Dict with sentiment score, reasoning, and citations
        """
        query = f"""
        Analyze the current sentiment and momentum around: {topic}

        {f"Context: {context}" if context else ""}

        Provide:
        1. Overall sentiment (strongly negative, negative, neutral, positive, strongly positive)
        2. Sentiment score from -1.0 (very negative) to +1.0 (very positive)
        3. Key positive factors
        4. Key negative factors
        5. Recent events driving sentiment

        Format your response as:
        SENTIMENT: [score from -1.0 to 1.0]
        DIRECTION: [negative/neutral/positive]
        REASONING: [brief explanation]
        POSITIVE_FACTORS: [list]
        NEGATIVE_FACTORS: [list]
        """

        result = self.research(
            query=query,
            recency_filter="week",
            max_tokens=800,
        )

        if not result:
            return None

        # Parse sentiment from response
        content = result.content
        sentiment_score = 0.0
        direction = "neutral"

        # Extract sentiment score
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("SENTIMENT:"):
                try:
                    score_str = line.replace("SENTIMENT:", "").strip()
                    sentiment_score = float(score_str)
                    sentiment_score = max(-1.0, min(1.0, sentiment_score))
                except:
                    pass
            elif line.startswith("DIRECTION:"):
                direction = line.replace("DIRECTION:", "").strip().lower()

        return {
            "sentiment_score": sentiment_score,
            "direction": direction,
            "analysis": result.content,
            "citations": result.citations,
            "confidence": result.confidence,
        }

    def fact_check(
        self,
        claim: str,
        deadline: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fact-check a specific claim relevant to a market.

        Args:
            claim: The claim to verify
            deadline: Market resolution deadline for temporal context

        Returns:
            Dict with verification status and evidence
        """
        query = f"""
        Fact-check the following claim: {claim}

        {f"Timeline: This needs to happen by {deadline}" if deadline else ""}

        Provide:
        1. VERDICT: TRUE / FALSE / UNVERIFIED / PARTIALLY_TRUE
        2. PROBABILITY: Estimated probability this claim is/will be true (0-100%)
        3. EVIDENCE: Key supporting or contradicting evidence
        4. SOURCES: Most reliable sources on this topic
        5. UNCERTAINTY: What could change the outcome
        """

        result = self.research(
            query=query,
            recency_filter="day",  # Most recent for fact-checking
            search_context_size="high",
        )

        if not result:
            return None

        # Parse verdict and probability
        content = result.content
        verdict = "UNVERIFIED"
        probability = 0.5

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip().upper()
            elif line.startswith("PROBABILITY:"):
                try:
                    prob_str = line.replace("PROBABILITY:", "").strip()
                    prob_str = prob_str.replace("%", "")
                    probability = float(prob_str) / 100
                except:
                    pass

        return {
            "verdict": verdict,
            "probability": probability,
            "analysis": result.content,
            "citations": result.citations,
            "confidence": result.confidence,
        }

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Test the client
    client = PerplexityClient()

    if not client.is_available:
        print("Perplexity API key not set. Set PERPLEXITY_API_KEY environment variable.")
        print("Get your key at: https://www.perplexity.ai/settings/api")
        exit(1)

    print("Testing Perplexity research client...")

    # Test market research
    result = client.research_market(
        market_question="Will there be a Russia-Ukraine ceasefire in 2025?",
        current_price=0.045,
    )

    if result:
        print(f"\n{'='*60}")
        print("RESEARCH RESULT")
        print(f"{'='*60}")
        print(f"Content:\n{result.content[:500]}...")
        print(f"\nCitations ({len(result.citations)}):")
        for cite in result.citations[:5]:
            print(f"  - {cite}")
        print(f"\nConfidence: {result.confidence:.1%}")
    else:
        print("Research failed")
