# Polymarket Swarm Trader Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a swarm-based trading intelligence system that identifies high-probability trades on Polymarket for non-crypto, non-sports markets by combining real-time market data with news and world events.

**Architecture:** A 5-agent swarm with specialized cognitive patterns: Market Scanner (convergent), News Researcher (divergent), Probability Analyst (critical), Risk Validator (systems), and Coordinator (adaptive). Each agent contributes to a weighted consensus that identifies mispriced markets with >10% edge.

**Tech Stack:** Python 3.9+, ruv-swarm MCP, Polymarket Gamma API, Finnhub News API, requests, pandas

---

## Data Sources

| Source | Endpoint | Purpose | Rate Limit |
|--------|----------|---------|------------|
| Polymarket Gamma | `https://gamma-api.polymarket.com/markets` | Market data, prices, volumes | Public/Unlimited |
| Polymarket Events | `https://gamma-api.polymarket.com/events` | Event metadata, categories | Public/Unlimited |
| Finnhub News | `https://finnhub.io/api/v1/news` | Real-time news, sentiment | 30/sec (free) |
| Web Search | WebSearch tool | Breaking news, verification | As needed |

---

## Task 1: Project Setup and Dependencies

**Files:**
- Create: `scripts/polymarket_swarm/__init__.py`
- Create: `scripts/polymarket_swarm/config.py`
- Create: `requirements-polymarket.txt`

**Step 1: Create project directory structure**

```bash
mkdir -p scripts/polymarket_swarm
```

**Step 2: Create requirements file**

Create `requirements-polymarket.txt`:
```
requests>=2.31.0
pandas>=2.0.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
```

**Step 3: Create config module**

Create `scripts/polymarket_swarm/config.py`:
```python
"""Configuration for Polymarket Swarm Trader."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Endpoints
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_MARKETS_ENDPOINT = f"{GAMMA_API_BASE}/markets"
GAMMA_EVENTS_ENDPOINT = f"{GAMMA_API_BASE}/events"

# Finnhub API (free tier)
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_NEWS_ENDPOINT = "https://finnhub.io/api/v1/news"

# Filter categories (exclude these)
EXCLUDED_CATEGORIES = [
    "crypto", "cryptocurrency", "bitcoin", "ethereum", "defi",
    "sports", "nfl", "nba", "mlb", "soccer", "football", "basketball",
    "baseball", "hockey", "tennis", "golf", "mma", "ufc", "boxing"
]

# Target categories (include these)
TARGET_CATEGORIES = [
    "politics", "elections", "economy", "science", "technology",
    "entertainment", "culture", "world", "business", "climate"
]

# Trading thresholds
MIN_EDGE_PERCENT = 10  # Minimum edge to recommend trade
MIN_VOLUME_USD = 10000  # Minimum market volume
MIN_CONFIDENCE = 0.7   # Minimum swarm confidence score
```

**Step 4: Create package init**

Create `scripts/polymarket_swarm/__init__.py`:
```python
"""Polymarket Swarm Trader - AI-powered prediction market analysis."""
__version__ = "0.1.0"
```

**Step 5: Install dependencies**

```bash
pip install -r requirements-polymarket.txt
```

**Step 6: Commit**

```bash
git add scripts/polymarket_swarm/ requirements-polymarket.txt
git commit -m "feat: initialize polymarket swarm trader project structure"
```

---

## Task 2: Polymarket API Client

**Files:**
- Create: `scripts/polymarket_swarm/polymarket_client.py`
- Test: `scripts/polymarket_swarm/test_polymarket_client.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_polymarket_client.py`:
```python
"""Tests for Polymarket API client."""
import pytest
from polymarket_client import PolymarketClient

def test_client_initialization():
    """Test client initializes with correct endpoints."""
    client = PolymarketClient()
    assert client.markets_endpoint == "https://gamma-api.polymarket.com/markets"
    assert client.events_endpoint == "https://gamma-api.polymarket.com/events"

def test_fetch_markets_returns_list():
    """Test that fetch_markets returns a list of market dicts."""
    client = PolymarketClient()
    markets = client.fetch_markets(limit=5)
    assert isinstance(markets, list)
    assert len(markets) <= 5

def test_filter_non_crypto_non_sports():
    """Test filtering excludes crypto and sports markets."""
    client = PolymarketClient()
    markets = client.fetch_filtered_markets(limit=20)
    for market in markets:
        question = market.get("question", "").lower()
        tags = [t.get("label", "").lower() for t in market.get("tags", [])]
        all_text = question + " ".join(tags)
        assert "bitcoin" not in all_text
        assert "nfl" not in all_text
        assert "crypto" not in all_text
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_polymarket_client.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'polymarket_client'"

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/polymarket_client.py`:
```python
"""Polymarket Gamma API client for fetching market data."""
import requests
from typing import List, Dict, Optional, Any
from config import (
    GAMMA_MARKETS_ENDPOINT,
    GAMMA_EVENTS_ENDPOINT,
    EXCLUDED_CATEGORIES,
    TARGET_CATEGORIES,
    MIN_VOLUME_USD
)

class PolymarketClient:
    """Client for interacting with Polymarket's Gamma API."""

    def __init__(self):
        self.markets_endpoint = GAMMA_MARKETS_ENDPOINT
        self.events_endpoint = GAMMA_EVENTS_ENDPOINT
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketSwarmTrader/0.1"
        })

    def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Polymarket Gamma API.

        Args:
            limit: Maximum number of markets to return
            offset: Pagination offset
            active: Only return active markets
            closed: Include closed markets

        Returns:
            List of market dictionaries
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower()
        }

        try:
            response = self.session.get(self.markets_endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching markets: {e}")
            return []

    def fetch_events(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch events from Polymarket Gamma API."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower()
        }

        try:
            response = self.session.get(self.events_endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching events: {e}")
            return []

    def _is_excluded_market(self, market: Dict[str, Any]) -> bool:
        """Check if market should be excluded (crypto/sports)."""
        question = market.get("question", "").lower()
        description = market.get("description", "").lower()
        tags = [t.get("label", "").lower() for t in market.get("tags", [])]

        all_text = f"{question} {description} {' '.join(tags)}"

        for excluded in EXCLUDED_CATEGORIES:
            if excluded in all_text:
                return True
        return False

    def _is_target_market(self, market: Dict[str, Any]) -> bool:
        """Check if market is in target categories."""
        tags = [t.get("label", "").lower() for t in market.get("tags", [])]

        for target in TARGET_CATEGORIES:
            if target in tags:
                return True
        return False

    def fetch_filtered_markets(
        self,
        limit: int = 100,
        min_volume: float = MIN_VOLUME_USD
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets filtered for non-crypto, non-sports categories.

        Args:
            limit: Maximum markets to return after filtering
            min_volume: Minimum volume in USD

        Returns:
            Filtered list of market dictionaries
        """
        filtered = []
        offset = 0
        batch_size = 100

        while len(filtered) < limit:
            markets = self.fetch_markets(limit=batch_size, offset=offset)
            if not markets:
                break

            for market in markets:
                if self._is_excluded_market(market):
                    continue

                # Check volume threshold
                volume = float(market.get("volume", 0) or 0)
                if volume < min_volume:
                    continue

                filtered.append(market)

                if len(filtered) >= limit:
                    break

            offset += batch_size

            # Safety limit to prevent infinite loops
            if offset > 1000:
                break

        return filtered[:limit]

    def get_market_by_id(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific market by ID."""
        try:
            url = f"{self.markets_endpoint}/{market_id}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching market {market_id}: {e}")
            return None

    def get_market_prices(self, market: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract current prices from market data.

        Returns:
            Dict with 'yes_price' and 'no_price' (0.0-1.0)
        """
        try:
            outcome_prices = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                import json
                prices = json.loads(outcome_prices)
            else:
                prices = outcome_prices

            return {
                "yes_price": float(prices[0]) if len(prices) > 0 else 0.5,
                "no_price": float(prices[1]) if len(prices) > 1 else 0.5
            }
        except (json.JSONDecodeError, IndexError, TypeError):
            return {"yes_price": 0.5, "no_price": 0.5}


if __name__ == "__main__":
    # Quick test
    client = PolymarketClient()
    markets = client.fetch_filtered_markets(limit=10)
    print(f"Found {len(markets)} filtered markets")
    for m in markets[:3]:
        print(f"  - {m.get('question', 'Unknown')[:60]}...")
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_polymarket_client.py -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/polymarket_client.py scripts/polymarket_swarm/test_polymarket_client.py
git commit -m "feat: add Polymarket Gamma API client with filtering"
```

---

## Task 3: News Research Client

**Files:**
- Create: `scripts/polymarket_swarm/news_client.py`
- Test: `scripts/polymarket_swarm/test_news_client.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_news_client.py`:
```python
"""Tests for news research client."""
import pytest
from news_client import NewsClient

def test_client_initialization():
    """Test news client initializes correctly."""
    client = NewsClient()
    assert client.finnhub_endpoint is not None

def test_search_news_returns_articles():
    """Test searching news returns article list."""
    client = NewsClient()
    articles = client.search_news("politics", limit=5)
    assert isinstance(articles, list)

def test_extract_keywords_from_question():
    """Test keyword extraction from market question."""
    client = NewsClient()
    question = "Will Biden win the 2024 presidential election?"
    keywords = client.extract_keywords(question)
    assert "biden" in [k.lower() for k in keywords]
    assert "election" in [k.lower() for k in keywords]
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_news_client.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/news_client.py`:
```python
"""News research client for gathering real-world data."""
import requests
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from config import FINNHUB_API_KEY, FINNHUB_NEWS_ENDPOINT

class NewsClient:
    """Client for fetching news and real-world event data."""

    def __init__(self, api_key: str = FINNHUB_API_KEY):
        self.api_key = api_key
        self.finnhub_endpoint = FINNHUB_NEWS_ENDPOINT
        self.session = requests.Session()

        # Common stop words to exclude from keywords
        self.stop_words = {
            "will", "the", "be", "is", "are", "was", "were", "been",
            "being", "have", "has", "had", "do", "does", "did", "a",
            "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "can", "just",
            "should", "now", "what", "who", "which", "this", "that"
        }

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract relevant keywords from text for news search.

        Args:
            text: Input text (e.g., market question)
            max_keywords: Maximum keywords to return

        Returns:
            List of keywords
        """
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter stop words and short words
        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) > 2
        ]

        # Prioritize capitalized words (names, places) from original
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Combine: capitalized first, then other keywords
        result = []
        for word in capitalized:
            if word.lower() not in [r.lower() for r in result]:
                result.append(word)

        for word in keywords:
            if word not in [r.lower() for r in result]:
                result.append(word)

        return result[:max_keywords]

    def search_news(
        self,
        query: str,
        category: str = "general",
        limit: int = 10,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles related to query.

        Args:
            query: Search query or keywords
            category: News category (general, forex, crypto, merger)
            limit: Maximum articles to return
            days_back: How many days back to search

        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            # Return mock data if no API key
            return self._get_mock_news(query, limit)

        params = {
            "category": category,
            "token": self.api_key
        }

        try:
            response = self.session.get(self.finnhub_endpoint, params=params)
            response.raise_for_status()
            articles = response.json()

            # Filter by date and relevance
            cutoff = datetime.now() - timedelta(days=days_back)
            cutoff_ts = cutoff.timestamp()

            query_words = set(query.lower().split())

            filtered = []
            for article in articles:
                # Check date
                if article.get("datetime", 0) < cutoff_ts:
                    continue

                # Check relevance
                headline = article.get("headline", "").lower()
                summary = article.get("summary", "").lower()
                text = f"{headline} {summary}"

                # Score by keyword matches
                matches = sum(1 for w in query_words if w in text)
                if matches > 0:
                    article["relevance_score"] = matches
                    filtered.append(article)

            # Sort by relevance
            filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            return filtered[:limit]

        except requests.RequestException as e:
            print(f"Error fetching news: {e}")
            return self._get_mock_news(query, limit)

    def _get_mock_news(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Return mock news data for testing without API key."""
        return [
            {
                "headline": f"Mock news about {query}",
                "summary": f"This is a mock article discussing {query} for testing.",
                "source": "MockNews",
                "datetime": int(datetime.now().timestamp()),
                "url": "https://example.com/mock",
                "relevance_score": 1
            }
        ][:limit]

    def get_news_for_market(
        self,
        market_question: str,
        market_description: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get relevant news for a specific market.

        Args:
            market_question: The market's question
            market_description: Additional market context

        Returns:
            List of relevant news articles
        """
        # Extract keywords
        keywords = self.extract_keywords(f"{market_question} {market_description}")

        # Search with combined keywords
        query = " ".join(keywords[:3])
        articles = self.search_news(query, limit=10)

        return articles

    def calculate_sentiment_score(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate simple sentiment score from articles.

        Returns:
            Score from -1 (negative) to 1 (positive)
        """
        if not articles:
            return 0.0

        positive_words = {
            "win", "success", "approve", "pass", "gain", "rise",
            "increase", "positive", "strong", "lead", "ahead",
            "likely", "expected", "confirm", "support"
        }

        negative_words = {
            "lose", "fail", "reject", "block", "drop", "fall",
            "decrease", "negative", "weak", "behind", "unlikely",
            "doubt", "oppose", "crisis", "concern"
        }

        total_score = 0
        for article in articles:
            text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()

            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)

            if pos + neg > 0:
                total_score += (pos - neg) / (pos + neg)

        return total_score / len(articles) if articles else 0.0


if __name__ == "__main__":
    client = NewsClient()
    keywords = client.extract_keywords("Will Trump win the 2024 election?")
    print(f"Keywords: {keywords}")

    articles = client.search_news("Trump election", limit=5)
    print(f"Found {len(articles)} articles")
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_news_client.py -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/news_client.py scripts/polymarket_swarm/test_news_client.py
git commit -m "feat: add news research client with keyword extraction"
```

---

## Task 4: Swarm Agent Definitions

**Files:**
- Create: `scripts/polymarket_swarm/agents.py`
- Test: `scripts/polymarket_swarm/test_agents.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_agents.py`:
```python
"""Tests for swarm agent definitions."""
import pytest
from agents import (
    AGENT_DEFINITIONS,
    MarketScannerAgent,
    NewsResearcherAgent,
    ProbabilityAnalystAgent,
    RiskValidatorAgent,
    CoordinatorAgent
)

def test_all_agents_defined():
    """Test all 5 agents are defined."""
    assert len(AGENT_DEFINITIONS) == 5

def test_market_scanner_has_convergent_pattern():
    """Test MarketScanner uses convergent cognitive pattern."""
    agent = MarketScannerAgent()
    assert agent.cognitive_pattern == "convergent"

def test_news_researcher_has_divergent_pattern():
    """Test NewsResearcher uses divergent cognitive pattern."""
    agent = NewsResearcherAgent()
    assert agent.cognitive_pattern == "divergent"

def test_agents_have_analyze_method():
    """Test all agents have analyze method."""
    for agent_class in [
        MarketScannerAgent,
        NewsResearcherAgent,
        ProbabilityAnalystAgent,
        RiskValidatorAgent,
        CoordinatorAgent
    ]:
        agent = agent_class()
        assert hasattr(agent, "analyze")
        assert callable(agent.analyze)
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_agents.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/agents.py`:
```python
"""Swarm agent definitions for Polymarket analysis."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class AgentResult:
    """Result from an agent's analysis."""
    agent_name: str
    confidence: float  # 0.0 to 1.0
    recommendation: str  # "YES", "NO", or "SKIP"
    edge_estimate: float  # Estimated edge percentage
    reasoning: str
    data_points: List[Dict[str, Any]] = field(default_factory=list)


class BaseAgent(ABC):
    """Base class for all swarm agents."""

    def __init__(self):
        self.name: str = "BaseAgent"
        self.cognitive_pattern: str = "convergent"
        self.weight: float = 1.0  # Weight in consensus calculation

    @abstractmethod
    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Analyze a market and return recommendation.

        Args:
            market: Market data from Polymarket API
            context: Additional context (news, prices, etc.)

        Returns:
            AgentResult with recommendation
        """
        pass


class MarketScannerAgent(BaseAgent):
    """
    Scans markets for opportunities using convergent thinking.
    Focuses on: volume, liquidity, price movements, market structure.
    """

    def __init__(self):
        super().__init__()
        self.name = "MarketScanner"
        self.cognitive_pattern = "convergent"
        self.weight = 1.0

    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze market structure and trading metrics."""

        # Extract market data
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)

        # Get prices
        prices = context.get("prices", {"yes_price": 0.5, "no_price": 0.5})
        yes_price = prices.get("yes_price", 0.5)
        no_price = prices.get("no_price", 0.5)

        # Calculate market quality score
        volume_score = min(volume / 100000, 1.0)  # Normalize to 100k
        liquidity_score = min(liquidity / 50000, 1.0)

        # Check for extreme prices (potential opportunities)
        price_extremity = abs(yes_price - 0.5) * 2  # 0 at 50%, 1 at extremes

        # Determine if market is tradeable
        market_quality = (volume_score + liquidity_score) / 2

        if market_quality < 0.1:
            return AgentResult(
                agent_name=self.name,
                confidence=0.3,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning="Insufficient volume/liquidity",
                data_points=[{"volume": volume, "liquidity": liquidity}]
            )

        # If price is extreme, there might be opportunity
        if price_extremity > 0.6:  # Price < 0.2 or > 0.8
            return AgentResult(
                agent_name=self.name,
                confidence=market_quality,
                recommendation="YES" if yes_price < 0.5 else "NO",
                edge_estimate=price_extremity * 20,  # Up to 20%
                reasoning=f"Extreme price ({yes_price:.2f}) with good liquidity",
                data_points=[{
                    "volume": volume,
                    "yes_price": yes_price,
                    "market_quality": market_quality
                }]
            )

        return AgentResult(
            agent_name=self.name,
            confidence=market_quality,
            recommendation="SKIP",
            edge_estimate=0,
            reasoning="No clear market structure opportunity",
            data_points=[{"volume": volume, "yes_price": yes_price}]
        )


class NewsResearcherAgent(BaseAgent):
    """
    Researches news and events using divergent thinking.
    Explores multiple sources and perspectives.
    """

    def __init__(self):
        super().__init__()
        self.name = "NewsResearcher"
        self.cognitive_pattern = "divergent"
        self.weight = 1.2  # Slightly higher weight for real-world data

    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """Analyze news and real-world events."""

        articles = context.get("news_articles", [])
        sentiment = context.get("sentiment_score", 0)

        if not articles:
            return AgentResult(
                agent_name=self.name,
                confidence=0.2,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning="No relevant news found",
                data_points=[]
            )

        # Analyze news volume and recency
        recent_articles = len([a for a in articles if a.get("relevance_score", 0) > 0])

        # Strong positive sentiment -> YES is more likely
        # Strong negative sentiment -> NO is more likely
        if abs(sentiment) > 0.3:
            recommendation = "YES" if sentiment > 0 else "NO"
            confidence = min(0.5 + abs(sentiment), 0.9)
            edge = abs(sentiment) * 15  # Up to 15% edge

            return AgentResult(
                agent_name=self.name,
                confidence=confidence,
                recommendation=recommendation,
                edge_estimate=edge,
                reasoning=f"News sentiment: {'positive' if sentiment > 0 else 'negative'} ({sentiment:.2f})",
                data_points=[{
                    "article_count": len(articles),
                    "sentiment": sentiment,
                    "recent_relevant": recent_articles
                }]
            )

        return AgentResult(
            agent_name=self.name,
            confidence=0.4,
            recommendation="SKIP",
            edge_estimate=0,
            reasoning="Neutral news sentiment",
            data_points=[{"article_count": len(articles), "sentiment": sentiment}]
        )


class ProbabilityAnalystAgent(BaseAgent):
    """
    Analyzes true probability using critical thinking.
    Compares market price to estimated fair value.
    """

    def __init__(self):
        super().__init__()
        self.name = "ProbabilityAnalyst"
        self.cognitive_pattern = "critical"
        self.weight = 1.5  # Highest weight for probability analysis

    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """Calculate fair probability and identify mispricing."""

        prices = context.get("prices", {"yes_price": 0.5})
        market_prob = prices.get("yes_price", 0.5)

        # Get inputs from other agents
        news_sentiment = context.get("sentiment_score", 0)
        volume_signal = context.get("volume_signal", 0)

        # Base probability estimation (simplified model)
        # In production, this would use ML models, historical data, etc.

        # Start with market price as base
        estimated_prob = market_prob

        # Adjust based on news sentiment
        sentiment_adjustment = news_sentiment * 0.15  # Up to 15% adjustment
        estimated_prob += sentiment_adjustment

        # Clamp to valid range
        estimated_prob = max(0.05, min(0.95, estimated_prob))

        # Calculate edge
        edge = (estimated_prob - market_prob) * 100  # As percentage

        if abs(edge) >= 10:  # Minimum 10% edge
            recommendation = "YES" if edge > 0 else "NO"
            confidence = min(0.5 + abs(edge) / 50, 0.95)

            return AgentResult(
                agent_name=self.name,
                confidence=confidence,
                recommendation=recommendation,
                edge_estimate=abs(edge),
                reasoning=f"Market: {market_prob:.1%}, Est: {estimated_prob:.1%}, Edge: {edge:+.1f}%",
                data_points=[{
                    "market_probability": market_prob,
                    "estimated_probability": estimated_prob,
                    "edge_percent": edge
                }]
            )

        return AgentResult(
            agent_name=self.name,
            confidence=0.5,
            recommendation="SKIP",
            edge_estimate=abs(edge),
            reasoning=f"Insufficient edge ({edge:+.1f}%)",
            data_points=[{
                "market_probability": market_prob,
                "estimated_probability": estimated_prob,
                "edge_percent": edge
            }]
        )


class RiskValidatorAgent(BaseAgent):
    """
    Validates trades using systems thinking.
    Considers market risks, liquidity, timing.
    """

    def __init__(self):
        super().__init__()
        self.name = "RiskValidator"
        self.cognitive_pattern = "systems"
        self.weight = 1.3

    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """Validate trade from risk perspective."""

        # Check market end date
        end_date = market.get("endDate") or market.get("end_date_iso")

        # Check liquidity
        volume = float(market.get("volume", 0) or 0)
        liquidity = float(market.get("liquidity", 0) or 0)

        # Risk factors
        risks = []
        risk_score = 0

        # Low liquidity risk
        if liquidity < 10000:
            risks.append("Low liquidity - may have slippage")
            risk_score += 0.3

        # Low volume risk
        if volume < 20000:
            risks.append("Low volume - limited price discovery")
            risk_score += 0.2

        # Check for neg_risk markets (complex resolution)
        if market.get("neg_risk") or market.get("negRisk"):
            risks.append("Negative risk market - complex resolution")
            risk_score += 0.1

        # If other agents recommend trade
        other_recommendations = context.get("other_recommendations", [])
        positive_votes = sum(1 for r in other_recommendations if r in ["YES", "NO"])

        if positive_votes == 0:
            return AgentResult(
                agent_name=self.name,
                confidence=0.3,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning="No positive signals from other agents",
                data_points=[{"risks": risks, "risk_score": risk_score}]
            )

        # Validate or reject
        if risk_score > 0.5:
            return AgentResult(
                agent_name=self.name,
                confidence=1 - risk_score,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning=f"High risk: {', '.join(risks)}",
                data_points=[{"risks": risks, "risk_score": risk_score}]
            )

        # Pass through the recommendation from probability agent
        prob_recommendation = context.get("probability_recommendation", "SKIP")
        edge = context.get("probability_edge", 0)

        return AgentResult(
            agent_name=self.name,
            confidence=0.7 - risk_score,
            recommendation=prob_recommendation,
            edge_estimate=edge * (1 - risk_score),  # Discount edge by risk
            reasoning=f"Risk validated (score: {risk_score:.2f})",
            data_points=[{"risks": risks, "risk_score": risk_score}]
        )


class CoordinatorAgent(BaseAgent):
    """
    Coordinates all agents using adaptive thinking.
    Synthesizes results and makes final recommendation.
    """

    def __init__(self):
        super().__init__()
        self.name = "Coordinator"
        self.cognitive_pattern = "adaptive"
        self.weight = 1.0

    def analyze(
        self,
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentResult:
        """Synthesize all agent results into final recommendation."""

        agent_results: List[AgentResult] = context.get("agent_results", [])

        if not agent_results:
            return AgentResult(
                agent_name=self.name,
                confidence=0,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning="No agent results to coordinate",
                data_points=[]
            )

        # Weighted voting
        yes_score = 0
        no_score = 0
        skip_score = 0
        total_weight = 0

        edges = []

        for result in agent_results:
            # Get agent weight (from agent definition or default)
            weight = result.data_points[0].get("agent_weight", 1.0) if result.data_points else 1.0

            if result.recommendation == "YES":
                yes_score += result.confidence * weight
                edges.append(result.edge_estimate)
            elif result.recommendation == "NO":
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

        # Determine final recommendation
        max_score = max(yes_score, no_score, skip_score)

        if max_score == skip_score or max_score < 0.4:
            return AgentResult(
                agent_name=self.name,
                confidence=max_score,
                recommendation="SKIP",
                edge_estimate=0,
                reasoning="Insufficient consensus for trade",
                data_points=[{
                    "yes_score": yes_score,
                    "no_score": no_score,
                    "skip_score": skip_score
                }]
            )

        recommendation = "YES" if yes_score > no_score else "NO"
        avg_edge = sum(edges) / len(edges) if edges else 0

        return AgentResult(
            agent_name=self.name,
            confidence=max_score,
            recommendation=recommendation,
            edge_estimate=avg_edge,
            reasoning=f"Swarm consensus: {recommendation} (score: {max_score:.2f})",
            data_points=[{
                "yes_score": yes_score,
                "no_score": no_score,
                "avg_edge": avg_edge,
                "agent_count": len(agent_results)
            }]
        )


# Agent registry
AGENT_DEFINITIONS = {
    "market_scanner": MarketScannerAgent,
    "news_researcher": NewsResearcherAgent,
    "probability_analyst": ProbabilityAnalystAgent,
    "risk_validator": RiskValidatorAgent,
    "coordinator": CoordinatorAgent
}
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_agents.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/agents.py scripts/polymarket_swarm/test_agents.py
git commit -m "feat: add 5 swarm agents with cognitive patterns"
```

---

## Task 5: Swarm Orchestrator

**Files:**
- Create: `scripts/polymarket_swarm/orchestrator.py`
- Test: `scripts/polymarket_swarm/test_orchestrator.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_orchestrator.py`:
```python
"""Tests for swarm orchestrator."""
import pytest
from orchestrator import SwarmOrchestrator

def test_orchestrator_initialization():
    """Test orchestrator initializes with all agents."""
    orchestrator = SwarmOrchestrator()
    assert len(orchestrator.agents) == 5

def test_orchestrator_has_analyze_market_method():
    """Test orchestrator has analyze_market method."""
    orchestrator = SwarmOrchestrator()
    assert hasattr(orchestrator, "analyze_market")
    assert callable(orchestrator.analyze_market)

def test_orchestrator_has_find_opportunities_method():
    """Test orchestrator has find_opportunities method."""
    orchestrator = SwarmOrchestrator()
    assert hasattr(orchestrator, "find_opportunities")
    assert callable(orchestrator.find_opportunities)
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_orchestrator.py -v
```
Expected: FAIL

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/orchestrator.py`:
```python
"""Swarm orchestrator for coordinating market analysis."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from agents import (
    AGENT_DEFINITIONS,
    AgentResult,
    MarketScannerAgent,
    NewsResearcherAgent,
    ProbabilityAnalystAgent,
    RiskValidatorAgent,
    CoordinatorAgent
)
from polymarket_client import PolymarketClient
from news_client import NewsClient
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


@dataclass
class TradeOpportunity:
    """A trading opportunity identified by the swarm."""
    market_id: str
    market_question: str
    recommendation: str  # "YES" or "NO"
    confidence: float
    edge_percent: float
    current_price: float
    estimated_fair_price: float
    reasoning: str
    agent_votes: Dict[str, str]
    risk_factors: List[str]


class SwarmOrchestrator:
    """
    Orchestrates the swarm of agents to analyze Polymarket opportunities.
    """

    def __init__(self):
        # Initialize all agents
        self.agents = {
            name: agent_class()
            for name, agent_class in AGENT_DEFINITIONS.items()
        }

        # Initialize clients
        self.polymarket = PolymarketClient()
        self.news = NewsClient()

    def analyze_market(self, market: Dict[str, Any]) -> Optional[TradeOpportunity]:
        """
        Run full swarm analysis on a single market.

        Args:
            market: Market data from Polymarket API

        Returns:
            TradeOpportunity if opportunity found, None otherwise
        """
        market_id = market.get("id") or market.get("condition_id", "unknown")
        question = market.get("question", "Unknown market")

        # Get current prices
        prices = self.polymarket.get_market_prices(market)

        # Get relevant news
        articles = self.news.get_news_for_market(
            question,
            market.get("description", "")
        )
        sentiment = self.news.calculate_sentiment_score(articles)

        # Build context for agents
        context = {
            "prices": prices,
            "news_articles": articles,
            "sentiment_score": sentiment,
            "volume_signal": 0,  # Could be enhanced
        }

        # Run agents in sequence (could be parallelized)
        results: List[AgentResult] = []

        # 1. Market Scanner
        scanner_result = self.agents["market_scanner"].analyze(market, context)
        results.append(scanner_result)

        # 2. News Researcher
        news_result = self.agents["news_researcher"].analyze(market, context)
        results.append(news_result)

        # 3. Probability Analyst
        prob_result = self.agents["probability_analyst"].analyze(market, context)
        results.append(prob_result)

        # Update context for risk validator
        context["other_recommendations"] = [r.recommendation for r in results]
        context["probability_recommendation"] = prob_result.recommendation
        context["probability_edge"] = prob_result.edge_estimate

        # 4. Risk Validator
        risk_result = self.agents["risk_validator"].analyze(market, context)
        results.append(risk_result)

        # 5. Coordinator - final synthesis
        context["agent_results"] = results
        final_result = self.agents["coordinator"].analyze(market, context)

        # Check if we have an opportunity
        if final_result.recommendation == "SKIP":
            return None

        if final_result.confidence < MIN_CONFIDENCE:
            return None

        if final_result.edge_estimate < MIN_EDGE_PERCENT:
            return None

        # Build opportunity
        agent_votes = {r.agent_name: r.recommendation for r in results}

        risk_factors = []
        if risk_result.data_points:
            risk_factors = risk_result.data_points[0].get("risks", [])

        # Calculate estimated fair price
        yes_price = prices.get("yes_price", 0.5)
        if final_result.recommendation == "YES":
            fair_price = yes_price + (final_result.edge_estimate / 100)
        else:
            fair_price = yes_price - (final_result.edge_estimate / 100)

        return TradeOpportunity(
            market_id=market_id,
            market_question=question,
            recommendation=final_result.recommendation,
            confidence=final_result.confidence,
            edge_percent=final_result.edge_estimate,
            current_price=yes_price,
            estimated_fair_price=fair_price,
            reasoning=final_result.reasoning,
            agent_votes=agent_votes,
            risk_factors=risk_factors
        )

    def find_opportunities(
        self,
        max_markets: int = 50,
        min_edge: float = MIN_EDGE_PERCENT,
        min_confidence: float = MIN_CONFIDENCE
    ) -> List[TradeOpportunity]:
        """
        Scan markets and find trading opportunities.

        Args:
            max_markets: Maximum markets to analyze
            min_edge: Minimum edge percentage to include
            min_confidence: Minimum confidence score

        Returns:
            List of trading opportunities, sorted by edge
        """
        # Fetch filtered markets
        markets = self.polymarket.fetch_filtered_markets(limit=max_markets)

        opportunities = []

        for i, market in enumerate(markets):
            print(f"Analyzing market {i+1}/{len(markets)}: {market.get('question', '')[:50]}...")

            try:
                opportunity = self.analyze_market(market)
                if opportunity:
                    if opportunity.edge_percent >= min_edge and opportunity.confidence >= min_confidence:
                        opportunities.append(opportunity)
                        print(f"  -> Found opportunity: {opportunity.recommendation} ({opportunity.edge_percent:.1f}% edge)")
            except Exception as e:
                print(f"  -> Error: {e}")
                continue

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        return opportunities

    def generate_report(self, opportunities: List[TradeOpportunity]) -> str:
        """Generate a text report of opportunities."""
        if not opportunities:
            return "No trading opportunities found."

        lines = [
            "=" * 60,
            "POLYMARKET SWARM TRADER - OPPORTUNITY REPORT",
            "=" * 60,
            "",
            f"Found {len(opportunities)} opportunities:",
            ""
        ]

        for i, opp in enumerate(opportunities, 1):
            lines.extend([
                f"--- Opportunity #{i} ---",
                f"Market: {opp.market_question}",
                f"Recommendation: {opp.recommendation}",
                f"Current Price: {opp.current_price:.1%}",
                f"Estimated Fair Price: {opp.estimated_fair_price:.1%}",
                f"Edge: {opp.edge_percent:.1f}%",
                f"Confidence: {opp.confidence:.1%}",
                f"Reasoning: {opp.reasoning}",
                f"Agent Votes: {opp.agent_votes}",
                f"Risk Factors: {', '.join(opp.risk_factors) if opp.risk_factors else 'None'}",
                ""
            ])

        return "\n".join(lines)

    def to_json(self, opportunities: List[TradeOpportunity]) -> str:
        """Convert opportunities to JSON."""
        data = [
            {
                "market_id": opp.market_id,
                "question": opp.market_question,
                "recommendation": opp.recommendation,
                "confidence": opp.confidence,
                "edge_percent": opp.edge_percent,
                "current_price": opp.current_price,
                "fair_price": opp.estimated_fair_price,
                "reasoning": opp.reasoning,
                "agent_votes": opp.agent_votes,
                "risk_factors": opp.risk_factors
            }
            for opp in opportunities
        ]
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    orchestrator = SwarmOrchestrator()
    print("Searching for opportunities...")
    opportunities = orchestrator.find_opportunities(max_markets=20)
    print(orchestrator.generate_report(opportunities))
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_orchestrator.py -v
```
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/orchestrator.py scripts/polymarket_swarm/test_orchestrator.py
git commit -m "feat: add swarm orchestrator with market analysis"
```

---

## Task 6: MCP Swarm Integration

**Files:**
- Create: `scripts/polymarket_swarm/mcp_swarm.py`
- Test: `scripts/polymarket_swarm/test_mcp_swarm.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_mcp_swarm.py`:
```python
"""Tests for MCP swarm integration."""
import pytest
from mcp_swarm import MCPSwarmAnalyzer

def test_mcp_analyzer_initialization():
    """Test MCP analyzer initializes."""
    analyzer = MCPSwarmAnalyzer()
    assert analyzer is not None

def test_has_run_analysis_method():
    """Test analyzer has run_analysis method."""
    analyzer = MCPSwarmAnalyzer()
    assert hasattr(analyzer, "run_analysis")
    assert callable(analyzer.run_analysis)
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_mcp_swarm.py -v
```
Expected: FAIL

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/mcp_swarm.py`:
```python
"""
MCP Swarm Integration for Polymarket Analysis.

This module provides integration with the ruv-swarm MCP server
for distributed agent processing.
"""
from typing import List, Dict, Any, Optional
import json
from dataclasses import asdict

from orchestrator import SwarmOrchestrator, TradeOpportunity


class MCPSwarmAnalyzer:
    """
    Integrates with MCP swarm tools for distributed analysis.

    Uses ruv-swarm MCP server capabilities:
    - swarm_init: Initialize swarm topology
    - agent_spawn: Create specialized agents
    - task_orchestrate: Distribute analysis tasks
    - daa_workflow_execute: Run autonomous workflows
    """

    def __init__(self):
        self.orchestrator = SwarmOrchestrator()
        self.swarm_initialized = False
        self.daa_agents = []

    async def initialize_swarm(self):
        """
        Initialize the MCP swarm with optimal topology.

        Uses hierarchical topology for:
        - Coordinator at top
        - Specialist agents below

        Call mcp__ruv-swarm__swarm_init with:
        - topology: "hierarchical"
        - maxAgents: 5
        - strategy: "specialized"
        """
        # This would be called via MCP tools
        self.swarm_initialized = True
        return {
            "topology": "hierarchical",
            "agents": 5,
            "status": "initialized"
        }

    async def spawn_daa_agents(self):
        """
        Spawn DAA (Decentralized Autonomous Agents) for each role.

        Call mcp__ruv-swarm__daa_agent_create for each:

        1. market_scanner:
           - cognitivePattern: "convergent"
           - capabilities: ["market_analysis", "volume_tracking"]

        2. news_researcher:
           - cognitivePattern: "divergent"
           - capabilities: ["news_search", "sentiment_analysis"]

        3. probability_analyst:
           - cognitivePattern: "critical"
           - capabilities: ["probability_calculation", "edge_detection"]

        4. risk_validator:
           - cognitivePattern: "systems"
           - capabilities: ["risk_assessment", "validation"]

        5. coordinator:
           - cognitivePattern: "adaptive"
           - capabilities: ["synthesis", "decision_making"]
        """
        agent_configs = [
            {
                "id": "market_scanner",
                "cognitivePattern": "convergent",
                "capabilities": ["market_analysis", "volume_tracking"],
                "enableMemory": True,
                "learningRate": 0.1
            },
            {
                "id": "news_researcher",
                "cognitivePattern": "divergent",
                "capabilities": ["news_search", "sentiment_analysis"],
                "enableMemory": True,
                "learningRate": 0.15
            },
            {
                "id": "probability_analyst",
                "cognitivePattern": "critical",
                "capabilities": ["probability_calculation", "edge_detection"],
                "enableMemory": True,
                "learningRate": 0.1
            },
            {
                "id": "risk_validator",
                "cognitivePattern": "systems",
                "capabilities": ["risk_assessment", "validation"],
                "enableMemory": True,
                "learningRate": 0.05
            },
            {
                "id": "coordinator",
                "cognitivePattern": "adaptive",
                "capabilities": ["synthesis", "decision_making"],
                "enableMemory": True,
                "learningRate": 0.2
            }
        ]

        self.daa_agents = agent_configs
        return agent_configs

    async def create_analysis_workflow(self, market_id: str):
        """
        Create DAA workflow for analyzing a single market.

        Call mcp__ruv-swarm__daa_workflow_create with:
        - id: f"analyze_{market_id}"
        - name: "Market Analysis Pipeline"
        - strategy: "adaptive"
        - steps: [fetch, scan, research, analyze, validate, synthesize]
        """
        workflow = {
            "id": f"analyze_{market_id}",
            "name": "Market Analysis Pipeline",
            "strategy": "adaptive",
            "steps": [
                {"id": "fetch", "action": "fetch_market_data", "agent": "market_scanner"},
                {"id": "scan", "action": "scan_market_structure", "agent": "market_scanner"},
                {"id": "research", "action": "gather_news", "agent": "news_researcher"},
                {"id": "analyze", "action": "calculate_probability", "agent": "probability_analyst"},
                {"id": "validate", "action": "assess_risk", "agent": "risk_validator"},
                {"id": "synthesize", "action": "final_decision", "agent": "coordinator"}
            ],
            "dependencies": {
                "scan": ["fetch"],
                "research": ["fetch"],
                "analyze": ["scan", "research"],
                "validate": ["analyze"],
                "synthesize": ["validate"]
            }
        }
        return workflow

    def run_analysis(
        self,
        max_markets: int = 50,
        use_mcp: bool = False
    ) -> List[TradeOpportunity]:
        """
        Run swarm analysis on Polymarket.

        Args:
            max_markets: Maximum markets to analyze
            use_mcp: Whether to use MCP swarm (requires async)

        Returns:
            List of trading opportunities
        """
        # For now, use the local orchestrator
        # MCP integration would be async and use the workflow above
        return self.orchestrator.find_opportunities(max_markets=max_markets)

    def get_mcp_commands(self) -> Dict[str, Any]:
        """
        Get the MCP commands to run for full swarm analysis.

        Returns dict of commands to execute via Claude Code.
        """
        return {
            "1_init_swarm": {
                "tool": "mcp__ruv-swarm__swarm_init",
                "params": {
                    "topology": "hierarchical",
                    "maxAgents": 5,
                    "strategy": "specialized"
                }
            },
            "2_init_daa": {
                "tool": "mcp__ruv-swarm__daa_init",
                "params": {
                    "enableCoordination": True,
                    "enableLearning": True,
                    "persistenceMode": "memory"
                }
            },
            "3_spawn_agents": [
                {
                    "tool": "mcp__ruv-swarm__daa_agent_create",
                    "params": config
                }
                for config in self.daa_agents or []
            ],
            "4_create_workflow": {
                "tool": "mcp__ruv-swarm__daa_workflow_create",
                "params": {
                    "id": "polymarket_analysis",
                    "name": "Polymarket Opportunity Scanner",
                    "strategy": "adaptive"
                }
            },
            "5_execute": {
                "tool": "mcp__ruv-swarm__daa_workflow_execute",
                "params": {
                    "workflowId": "polymarket_analysis",
                    "parallelExecution": True
                }
            }
        }


if __name__ == "__main__":
    analyzer = MCPSwarmAnalyzer()

    # Print MCP command sequence
    print("MCP Commands for Swarm Analysis:")
    print(json.dumps(analyzer.get_mcp_commands(), indent=2))

    # Run local analysis
    print("\nRunning local analysis...")
    opportunities = analyzer.run_analysis(max_markets=10)

    for opp in opportunities:
        print(f"\n{opp.recommendation}: {opp.market_question[:50]}...")
        print(f"  Edge: {opp.edge_percent:.1f}%, Confidence: {opp.confidence:.1%}")
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_mcp_swarm.py -v
```
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/mcp_swarm.py scripts/polymarket_swarm/test_mcp_swarm.py
git commit -m "feat: add MCP swarm integration for distributed analysis"
```

---

## Task 7: Main CLI and Runner

**Files:**
- Create: `scripts/polymarket_swarm/main.py`
- Create: `scripts/polymarket_swarm/run_swarm.py`

**Step 1: Create the main CLI module**

Create `scripts/polymarket_swarm/main.py`:
```python
"""
Main CLI for Polymarket Swarm Trader.

Usage:
    python main.py scan [--max-markets N] [--min-edge N]
    python main.py analyze <market_id>
    python main.py report [--output FILE]
"""
import argparse
import json
import sys
from datetime import datetime

from orchestrator import SwarmOrchestrator
from mcp_swarm import MCPSwarmAnalyzer
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


def cmd_scan(args):
    """Scan markets for opportunities."""
    print(f"Scanning up to {args.max_markets} markets...")
    print(f"Minimum edge: {args.min_edge}%")
    print(f"Minimum confidence: {args.min_confidence}")
    print()

    orchestrator = SwarmOrchestrator()
    opportunities = orchestrator.find_opportunities(
        max_markets=args.max_markets,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence
    )

    # Print report
    print(orchestrator.generate_report(opportunities))

    # Save JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(orchestrator.to_json(opportunities))
        print(f"\nSaved to {args.output}")

    return opportunities


def cmd_analyze(args):
    """Analyze a specific market."""
    orchestrator = SwarmOrchestrator()

    # Fetch the specific market
    market = orchestrator.polymarket.get_market_by_id(args.market_id)

    if not market:
        print(f"Market not found: {args.market_id}")
        return None

    print(f"Analyzing: {market.get('question', 'Unknown')}")
    print()

    opportunity = orchestrator.analyze_market(market)

    if opportunity:
        print(f"Recommendation: {opportunity.recommendation}")
        print(f"Edge: {opportunity.edge_percent:.1f}%")
        print(f"Confidence: {opportunity.confidence:.1%}")
        print(f"Current Price: {opportunity.current_price:.1%}")
        print(f"Fair Price: {opportunity.estimated_fair_price:.1%}")
        print(f"Reasoning: {opportunity.reasoning}")
        print(f"Agent Votes: {opportunity.agent_votes}")
    else:
        print("No trading opportunity identified.")

    return opportunity


def cmd_mcp_setup(args):
    """Print MCP commands for swarm setup."""
    analyzer = MCPSwarmAnalyzer()

    # Initialize agent configs
    import asyncio
    asyncio.run(analyzer.spawn_daa_agents())

    commands = analyzer.get_mcp_commands()
    print("MCP Commands for Swarm Setup:")
    print("=" * 40)
    print(json.dumps(commands, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Swarm Trader - Find high-probability trades"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan markets for opportunities")
    scan_parser.add_argument(
        "--max-markets", "-m",
        type=int,
        default=50,
        help="Maximum markets to analyze"
    )
    scan_parser.add_argument(
        "--min-edge", "-e",
        type=float,
        default=MIN_EDGE_PERCENT,
        help="Minimum edge percentage"
    )
    scan_parser.add_argument(
        "--min-confidence", "-c",
        type=float,
        default=MIN_CONFIDENCE,
        help="Minimum confidence score"
    )
    scan_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific market")
    analyze_parser.add_argument("market_id", help="Market ID to analyze")

    # MCP setup command
    mcp_parser = subparsers.add_parser("mcp-setup", help="Print MCP swarm setup commands")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "mcp-setup":
        cmd_mcp_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Create convenience runner script**

Create `scripts/polymarket_swarm/run_swarm.py`:
```python
#!/usr/bin/env python3
"""
Quick runner for Polymarket Swarm analysis.

This script runs a full scan with default settings.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import SwarmOrchestrator
from datetime import datetime


def main():
    print("=" * 60)
    print("POLYMARKET SWARM TRADER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("Categories: Politics, Economy, Technology, Entertainment")
    print("Excluded: Crypto, Sports")
    print()

    orchestrator = SwarmOrchestrator()

    # Run analysis
    opportunities = orchestrator.find_opportunities(max_markets=30)

    # Generate and print report
    report = orchestrator.generate_report(opportunities)
    print(report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"polymarket_opportunities_{timestamp}.json"

    with open(output_file, 'w') as f:
        f.write(orchestrator.to_json(opportunities))

    print(f"\nResults saved to: {output_file}")

    return opportunities


if __name__ == "__main__":
    main()
```

**Step 3: Make runner executable and test**

```bash
cd scripts/polymarket_swarm && python run_swarm.py
```

**Step 4: Commit**

```bash
git add scripts/polymarket_swarm/main.py scripts/polymarket_swarm/run_swarm.py
git commit -m "feat: add CLI and runner script"
```

---

## Task 8: Live MCP Swarm Execution

**Purpose:** Run the full swarm using MCP tools for distributed cognitive analysis.

**Step 1: Initialize the MCP swarm**

Execute via Claude Code:
```
mcp__ruv-swarm__swarm_init(topology="hierarchical", maxAgents=5, strategy="specialized")
```

**Step 2: Initialize DAA service**

```
mcp__ruv-swarm__daa_init(enableCoordination=True, enableLearning=True, persistenceMode="memory")
```

**Step 3: Spawn specialized agents**

Execute 5 agent creates in parallel:
```
mcp__ruv-swarm__daa_agent_create(id="market_scanner", cognitivePattern="convergent", capabilities=["market_analysis"], enableMemory=True)
mcp__ruv-swarm__daa_agent_create(id="news_researcher", cognitivePattern="divergent", capabilities=["news_search"], enableMemory=True)
mcp__ruv-swarm__daa_agent_create(id="probability_analyst", cognitivePattern="critical", capabilities=["edge_detection"], enableMemory=True)
mcp__ruv-swarm__daa_agent_create(id="risk_validator", cognitivePattern="systems", capabilities=["risk_assessment"], enableMemory=True)
mcp__ruv-swarm__daa_agent_create(id="coordinator", cognitivePattern="adaptive", capabilities=["synthesis"], enableMemory=True)
```

**Step 4: Create analysis workflow**

```
mcp__ruv-swarm__daa_workflow_create(
    id="polymarket_scan",
    name="Polymarket Opportunity Scanner",
    strategy="adaptive",
    steps=[
        {"id": "fetch", "agent": "market_scanner"},
        {"id": "research", "agent": "news_researcher"},
        {"id": "analyze", "agent": "probability_analyst"},
        {"id": "validate", "agent": "risk_validator"},
        {"id": "decide", "agent": "coordinator"}
    ]
)
```

**Step 5: Execute the workflow**

```
mcp__ruv-swarm__daa_workflow_execute(workflowId="polymarket_scan", parallelExecution=True)
```

**Step 6: Check results**

```
mcp__ruv-swarm__daa_performance_metrics(category="all")
mcp__ruv-swarm__daa_learning_status(detailed=True)
```

---

## Task 9: Integration Testing

**Files:**
- Create: `scripts/polymarket_swarm/test_integration.py`

**Step 1: Write integration tests**

Create `scripts/polymarket_swarm/test_integration.py`:
```python
"""Integration tests for Polymarket Swarm Trader."""
import pytest
from orchestrator import SwarmOrchestrator
from polymarket_client import PolymarketClient
from news_client import NewsClient


class TestIntegration:
    """Integration tests that hit real APIs."""

    def test_full_pipeline_single_market(self):
        """Test full analysis pipeline on a real market."""
        client = PolymarketClient()
        markets = client.fetch_filtered_markets(limit=1)

        assert len(markets) > 0, "Should fetch at least one market"

        orchestrator = SwarmOrchestrator()
        # Just test that analysis runs without error
        result = orchestrator.analyze_market(markets[0])
        # Result can be None (no opportunity) or TradeOpportunity
        assert result is None or hasattr(result, 'recommendation')

    def test_news_api_connection(self):
        """Test that news API responds."""
        client = NewsClient()
        articles = client.search_news("politics", limit=3)
        # Should return list (possibly empty if no API key)
        assert isinstance(articles, list)

    def test_polymarket_api_connection(self):
        """Test that Polymarket API responds."""
        client = PolymarketClient()
        markets = client.fetch_markets(limit=5)

        assert len(markets) > 0, "Polymarket API should return markets"
        assert "question" in markets[0] or "condition_id" in markets[0]

    def test_filter_excludes_crypto_sports(self):
        """Test that filtering properly excludes crypto and sports."""
        client = PolymarketClient()
        markets = client.fetch_filtered_markets(limit=20)

        crypto_terms = ["bitcoin", "ethereum", "crypto", "defi"]
        sports_terms = ["nfl", "nba", "mlb", "soccer", "football"]

        for market in markets:
            question = market.get("question", "").lower()
            for term in crypto_terms + sports_terms:
                assert term not in question, f"Found excluded term '{term}' in: {question}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Run integration tests**

```bash
cd scripts/polymarket_swarm && python -m pytest test_integration.py -v
```

**Step 3: Commit**

```bash
git add scripts/polymarket_swarm/test_integration.py
git commit -m "test: add integration tests for full pipeline"
```

---

## Task 10: Documentation and Final Commit

**Files:**
- Create: `scripts/polymarket_swarm/README.md`

**Step 1: Create README**

Create `scripts/polymarket_swarm/README.md`:
```markdown
# Polymarket Swarm Trader

AI-powered trading intelligence for Polymarket prediction markets using swarm agents with diverse cognitive patterns.

## Features

- **5 Specialized Agents**: Market Scanner, News Researcher, Probability Analyst, Risk Validator, Coordinator
- **Cognitive Diversity**: Each agent uses different thinking patterns (convergent, divergent, critical, systems, adaptive)
- **Real-time Data**: Integrates Polymarket API and news sources
- **Smart Filtering**: Focuses on politics, economy, tech - excludes crypto/sports
- **MCP Integration**: Can run as distributed swarm via ruv-swarm

## Quick Start

```bash
# Install dependencies
pip install -r requirements-polymarket.txt

# Run a scan
cd scripts/polymarket_swarm
python run_swarm.py

# Or use CLI
python main.py scan --max-markets 50 --min-edge 10
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Coordinator Agent             │
│        (Adaptive Thinking)              │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴───┐   ┌─────┴─────┐   ┌───┴───┐
│Market │   │Probability│   │ Risk  │
│Scanner│   │ Analyst   │   │Validat│
└───┬───┘   └─────┬─────┘   └───┬───┘
    │             │             │
    │       ┌─────┴─────┐       │
    │       │   News    │       │
    │       │Researcher │       │
    │       └───────────┘       │
    └─────────────┴─────────────┘
```

## Data Sources

| Source | Purpose |
|--------|---------|
| Polymarket Gamma API | Market data, prices |
| Finnhub News | Real-time news |
| WebSearch | Breaking events |

## Output

Opportunities are reported with:
- **Recommendation**: YES or NO
- **Edge**: Estimated % advantage
- **Confidence**: Swarm consensus score
- **Reasoning**: Why this trade

## License

MIT
```

**Step 2: Final commit**

```bash
git add scripts/polymarket_swarm/README.md
git add -A scripts/polymarket_swarm/
git commit -m "docs: add README and complete polymarket swarm trader"
```

---

## Summary

This plan creates a complete Polymarket swarm trading system with:

1. **Polymarket Client** - Fetches and filters markets (excludes crypto/sports)
2. **News Client** - Gathers real-world data for validation
3. **5 Swarm Agents** - Each with specialized cognitive patterns
4. **Orchestrator** - Coordinates agents and synthesizes recommendations
5. **MCP Integration** - Distributed execution via ruv-swarm
6. **CLI** - Easy command-line interface

**Total Tasks**: 10
**Estimated Commits**: 10

**Key Sources**:
- [Polymarket Documentation](https://docs.polymarket.com/)
- [Polymarket Gamma API](https://docs.polymarket.com/developers/gamma-markets-api/overview)
- [Polymarket Agents GitHub](https://github.com/Polymarket/agents)
- [Finnhub API](https://finnhub.io/docs/api/market-news)
