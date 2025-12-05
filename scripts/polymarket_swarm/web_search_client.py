"""Web Search Client with OpenAI Search Fallback.

Provides real-time web search using:
1. Perplexity Sonar (primary) - best for deep research
2. OpenAI Search (backup) - gpt-4o-mini-search-preview

This ensures we always have web search capability even if one API is down.
"""
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import httpx

from config import OPENAI_API_KEY


# API Keys
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)


@dataclass
class WebSearchResult:
    """Result from web search query."""
    content: str
    citations: List[str] = field(default_factory=list)
    provider: str = ""  # "perplexity" or "openai"
    model: str = ""
    query: str = ""
    timestamp: str = ""
    confidence: float = 0.5
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "citations": self.citations,
            "provider": self.provider,
            "model": self.model,
            "query": self.query,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "success": self.success,
            "error": self.error,
        }


class WebSearchClient:
    """Unified web search client with automatic fallback.

    Primary: Perplexity Sonar (deep research with citations)
    Backup: OpenAI gpt-4o-mini-search-preview (web search enabled)
    """

    PERPLEXITY_API_BASE = "https://api.perplexity.ai"
    OPENAI_API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        perplexity_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        timeout: float = 60.0,
        prefer_perplexity: bool = True,
    ):
        """Initialize web search client.

        Args:
            perplexity_key: Perplexity API key (optional)
            openai_key: OpenAI API key (optional)
            timeout: Request timeout in seconds
            prefer_perplexity: Use Perplexity as primary if available
        """
        self.perplexity_key = perplexity_key or PERPLEXITY_API_KEY
        self.openai_key = openai_key or OPENAI_KEY
        self.prefer_perplexity = prefer_perplexity
        self.client = httpx.Client(timeout=timeout)

        # Track provider status
        self.perplexity_available = bool(self.perplexity_key)
        self.openai_available = bool(self.openai_key)

        providers = []
        if self.perplexity_available:
            providers.append("Perplexity")
        if self.openai_available:
            providers.append("OpenAI")

        print(f"WebSearchClient initialized: {', '.join(providers) or 'No providers available'}")

    @property
    def is_available(self) -> bool:
        """Check if any search provider is available."""
        return self.perplexity_available or self.openai_available

    def _search_perplexity(
        self,
        query: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> WebSearchResult:
        """Execute search using Perplexity Sonar API."""
        try:
            response = self.client.post(
                f"{self.PERPLEXITY_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "search_recency_filter": "week",
                    "web_search_options": {"search_context_size": "high"},
                    "return_citations": True,
                },
            )

            if response.status_code != 200:
                return WebSearchResult(
                    content="",
                    provider="perplexity",
                    model="sonar",
                    query=query,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error=f"API error: {response.status_code}",
                )

            data = response.json()

            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            citations = data.get("citations", [])
            confidence = min(0.5 + len(citations) * 0.1, 0.95)

            return WebSearchResult(
                content=content,
                citations=citations,
                provider="perplexity",
                model="sonar",
                query=query,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                success=True,
            )

        except Exception as e:
            return WebSearchResult(
                content="",
                provider="perplexity",
                model="sonar",
                query=query,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e),
            )

    def _search_openai(
        self,
        query: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> WebSearchResult:
        """Execute search using OpenAI's search-enabled model.

        Uses gpt-4o-mini-search-preview which has built-in web search.
        """
        try:
            # OpenAI search model - includes web search capability
            response = self.client.post(
                f"{self.OPENAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini-search-preview",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    # Enable web search
                    "web_search_options": {
                        "search_context_size": "medium",
                    },
                },
            )

            if response.status_code != 200:
                # Try fallback to regular gpt-4o-mini without search
                return self._search_openai_fallback(query, system_prompt, max_tokens)

            data = response.json()

            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            # OpenAI search may include inline citations in the response
            citations = self._extract_citations_from_text(content)

            return WebSearchResult(
                content=content,
                citations=citations,
                provider="openai",
                model="gpt-4o-mini-search-preview",
                query=query,
                timestamp=datetime.now().isoformat(),
                confidence=0.7 if citations else 0.5,
                success=True,
            )

        except Exception as e:
            return WebSearchResult(
                content="",
                provider="openai",
                model="gpt-4o-mini-search-preview",
                query=query,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e),
            )

    def _search_openai_fallback(
        self,
        query: str,
        system_prompt: str,
        max_tokens: int = 1024,
    ) -> WebSearchResult:
        """Fallback to regular GPT-4o-mini without web search."""
        try:
            response = self.client.post(
                f"{self.OPENAI_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nNote: You don't have access to real-time web search. Provide analysis based on your training data and clearly state when information may be outdated."},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )

            if response.status_code != 200:
                return WebSearchResult(
                    content="",
                    provider="openai",
                    model="gpt-4o-mini",
                    query=query,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error=f"API error: {response.status_code}",
                )

            data = response.json()

            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            return WebSearchResult(
                content=content,
                citations=[],
                provider="openai",
                model="gpt-4o-mini (no web search)",
                query=query,
                timestamp=datetime.now().isoformat(),
                confidence=0.4,  # Lower confidence without web search
                success=True,
            )

        except Exception as e:
            return WebSearchResult(
                content="",
                provider="openai",
                model="gpt-4o-mini",
                query=query,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e),
            )

    def _extract_citations_from_text(self, text: str) -> List[str]:
        """Extract URLs/citations from text response."""
        import re
        urls = re.findall(r'https?://[^\s\)\]>]+', text)
        return list(set(urls))[:10]  # Dedupe and limit

    def search(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> WebSearchResult:
        """Execute web search with automatic fallback.

        Args:
            query: Search query
            system_prompt: Optional system instructions
            max_tokens: Maximum response tokens

        Returns:
            WebSearchResult from best available provider
        """
        if system_prompt is None:
            system_prompt = (
                "You are a research analyst. Provide factual, well-sourced information. "
                "Focus on recent news and verified facts. Be precise and cite sources."
            )

        # Try primary provider first
        if self.prefer_perplexity and self.perplexity_available:
            result = self._search_perplexity(query, system_prompt, max_tokens)
            if result.success:
                return result
            print(f"Perplexity failed: {result.error}, trying OpenAI...")

        # Try OpenAI
        if self.openai_available:
            result = self._search_openai(query, system_prompt, max_tokens)
            if result.success:
                return result
            print(f"OpenAI search failed: {result.error}")

        # If OpenAI was not preferred, try Perplexity as backup
        if not self.prefer_perplexity and self.perplexity_available:
            result = self._search_perplexity(query, system_prompt, max_tokens)
            if result.success:
                return result

        # All providers failed
        return WebSearchResult(
            content="",
            provider="none",
            model="none",
            query=query,
            timestamp=datetime.now().isoformat(),
            success=False,
            error="All search providers failed",
        )

    def research_market(
        self,
        market_question: str,
        market_description: Optional[str] = None,
        current_price: Optional[float] = None,
    ) -> WebSearchResult:
        """Research a specific prediction market.

        Args:
            market_question: The market question
            market_description: Optional description
            current_price: Current YES price

        Returns:
            WebSearchResult with research findings
        """
        query_parts = [
            f"Research the following prediction market: {market_question}",
        ]

        if market_description:
            query_parts.append(f"Context: {market_description[:500]}")

        if current_price is not None:
            query_parts.append(
                f"Current market probability: {current_price:.1%}. "
                "Is this price accurate based on recent news?"
            )

        query_parts.extend([
            "",
            "Find and analyze:",
            "1. Recent news articles and official announcements",
            "2. Expert opinions and analysis",
            "3. Key factors that could affect the outcome",
            "4. Any information the market might be missing",
            "",
            "Provide specific dates and sources for claims.",
        ])

        system_prompt = (
            "You are a prediction market analyst. Focus on recent news and events "
            "that could move the market. Be specific about sources and confidence levels. "
            "Identify any mispricing or information gaps."
        )

        return self.search(
            query="\n".join(query_parts),
            system_prompt=system_prompt,
            max_tokens=1200,
        )

    def get_news_context(
        self,
        topic: str,
        hours: int = 48,
    ) -> WebSearchResult:
        """Get recent news context for a topic.

        Args:
            topic: Topic to search for
            hours: How recent (for context, actual filtering depends on provider)

        Returns:
            WebSearchResult with news summary
        """
        query = f"""
        Find the most recent news and developments about: {topic}

        Focus on:
        - Breaking news and announcements from the last {hours} hours
        - Official statements and press releases
        - Expert analysis and predictions
        - Market-moving events

        Summarize the key points with dates and sources.
        """

        system_prompt = (
            "You are a news aggregator. Summarize recent developments concisely. "
            "Include specific dates and sources for all claims."
        )

        return self.search(query=query, system_prompt=system_prompt, max_tokens=800)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("WEB SEARCH CLIENT TEST")
    print("=" * 70)

    with WebSearchClient() as client:
        if not client.is_available:
            print("No search providers available!")
            print("Set PERPLEXITY_API_KEY or OPENAI_API_KEY environment variables.")
            exit(1)

        print(f"\nPerplexity available: {client.perplexity_available}")
        print(f"OpenAI available: {client.openai_available}")

        # Test market research
        print("\n" + "-" * 50)
        print("Testing market research...")

        result = client.research_market(
            market_question="Will there be a Russia-Ukraine ceasefire in 2025?",
            current_price=0.045,
        )

        print(f"\nProvider: {result.provider} ({result.model})")
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Citations: {len(result.citations)}")

        if result.success:
            print(f"\nContent preview:\n{result.content[:500]}...")

            if result.citations:
                print(f"\nCitations:")
                for cite in result.citations[:3]:
                    print(f"  - {cite[:70]}...")
        else:
            print(f"Error: {result.error}")

        # Test news context
        print("\n" + "-" * 50)
        print("Testing news context...")

        news = client.get_news_context("Federal Reserve interest rate decision December 2025")

        print(f"\nProvider: {news.provider} ({news.model})")
        print(f"Success: {news.success}")

        if news.success:
            print(f"\nNews preview:\n{news.content[:400]}...")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
