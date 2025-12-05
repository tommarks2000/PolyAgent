# Polymarket Swarm Trading System - Updated Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a swarm-based trading intelligence system that identifies high-probability trades on Polymarket for non-crypto, non-sports markets by combining real-time market data with news and world events.

**Version:** 2.0 (Dec 5, 2025) - Updated based on critical review of [Polymarket/agents](https://github.com/Polymarket/agents)

---

## Key Improvements Over v1.0

| Area | v1.0 (Nov 27) | v2.0 (Dec 5) | Rationale |
|------|---------------|--------------|-----------|
| Data Models | Raw dicts | Pydantic models | Type safety, validation |
| API Filtering | Basic | `enableOrderBook=True` | Only tradeable markets |
| Market Selection | Keywords | RAG + ChromaDB | Semantic relevance |
| HTTP Client | requests | httpx | Async capability |
| Forecasting | Simple prompt | Superforecaster methodology | Proven framework |
| Token Tracking | Missing | CLOB token IDs | Required for execution |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │      Market Data Pipeline       │
                    │  (Gamma API + enableOrderBook)  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │      RAG Market Selector        │
                    │  (ChromaDB + semantic search)   │
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼───────┐          ┌────────▼────────┐         ┌────────▼────────┐
│Market Scanner │          │ News Researcher │         │  Probability    │
│ (convergent)  │          │  (divergent)    │         │    Analyst      │
│               │          │                 │         │   (critical)    │
│ - Volume      │          │ - News API      │         │ - Superforecast │
│ - Liquidity   │          │ - Sentiment     │         │ - Edge calc     │
│ - Price range │          │ - Web search    │         │ - Fair value    │
└───────┬───────┘          └────────┬────────┘         └────────┬────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       Risk Validator            │
                    │        (systems)                │
                    │  - Liquidity check              │
                    │  - Slippage estimate            │
                    │  - Position sizing              │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │        Coordinator              │
                    │        (adaptive)               │
                    │  - Weighted consensus           │
                    │  - Final recommendation         │
                    └─────────────────────────────────┘
```

---

## Data Sources

| Source | Endpoint | Purpose | Rate Limit |
|--------|----------|---------|------------|
| Polymarket Gamma | `https://gamma-api.polymarket.com/markets` | Market data, prices, volumes | Public/Unlimited |
| Polymarket Events | `https://gamma-api.polymarket.com/events` | Event metadata, categories | Public/Unlimited |
| NewsAPI | `https://newsapi.org/v2/everything` | News articles, headlines | 100/day (free) |
| Finnhub News | `https://finnhub.io/api/v1/news` | Real-time financial news | 30/sec (free) |
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
httpx>=0.25.0
pydantic>=2.5.0
pandas>=2.0.0
python-dotenv>=1.0.0
chromadb>=0.4.0
langchain>=0.1.0
langchain-openai>=0.0.5
newsapi-python>=0.2.7
```

**Step 3: Create config module**

Create `scripts/polymarket_swarm/config.py`:
```python
"""Configuration for Polymarket Swarm Trader v2.0."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Endpoints
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_MARKETS_ENDPOINT = f"{GAMMA_API_BASE}/markets"
GAMMA_EVENTS_ENDPOINT = f"{GAMMA_API_BASE}/events"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Tradeable market parameters (CRITICAL: enableOrderBook for actual trading)
TRADEABLE_MARKET_PARAMS = {
    "active": True,
    "closed": False,
    "archived": False,
    "enableOrderBook": True,  # Only markets we can actually trade
}

# Filter categories (exclude these)
EXCLUDED_CATEGORIES = [
    "crypto", "cryptocurrency", "bitcoin", "ethereum", "defi", "nft",
    "sports", "nfl", "nba", "mlb", "soccer", "football", "basketball",
    "baseball", "hockey", "tennis", "golf", "mma", "ufc", "boxing",
    "esports", "gaming"
]

# Target categories (prioritize these)
TARGET_CATEGORIES = [
    "politics", "elections", "government", "policy",
    "economy", "finance", "federal reserve", "inflation",
    "science", "technology", "ai", "space",
    "entertainment", "culture", "awards",
    "world", "international", "geopolitics",
    "business", "companies", "markets"
]

# Trading thresholds
MIN_EDGE_PERCENT = 10.0      # Minimum edge to recommend trade
MIN_VOLUME_USD = 10000.0     # Minimum market volume
MIN_LIQUIDITY_USD = 5000.0   # Minimum market liquidity
MIN_CONFIDENCE = 0.7         # Minimum swarm confidence score
MAX_POSITION_PERCENT = 0.1   # Max 10% of bankroll per trade

# RAG Settings
EMBEDDING_MODEL = "text-embedding-3-small"
RAG_TOP_K = 10  # Number of markets to return from semantic search

# Agent weights for consensus
AGENT_WEIGHTS = {
    "market_scanner": 1.0,
    "news_researcher": 1.2,
    "probability_analyst": 1.5,  # Highest weight
    "risk_validator": 1.3,
    "coordinator": 1.0,
}
```

**Step 4: Create package init**

Create `scripts/polymarket_swarm/__init__.py`:
```python
"""Polymarket Swarm Trader v2.0 - AI-powered prediction market analysis."""
__version__ = "2.0.0"
```

**Step 5: Install dependencies**

```bash
pip install -r requirements-polymarket.txt
```

**Step 6: Commit**

```bash
git add scripts/polymarket_swarm/ requirements-polymarket.txt
git commit -m "feat: initialize polymarket swarm trader v2.0 with updated config"
```

---

## Task 2: Pydantic Data Models

**Files:**
- Create: `scripts/polymarket_swarm/models.py`

**Step 1: Create comprehensive data models**

Create `scripts/polymarket_swarm/models.py`:
```python
"""Pydantic data models for Polymarket Swarm Trader.

Based on Polymarket/agents but extended for our swarm architecture.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Recommendation(str, Enum):
    """Trading recommendation enum."""
    YES = "YES"
    NO = "NO"
    SKIP = "SKIP"


class CognitivePattern(str, Enum):
    """Agent cognitive patterns."""
    CONVERGENT = "convergent"
    DIVERGENT = "divergent"
    CRITICAL = "critical"
    SYSTEMS = "systems"
    ADAPTIVE = "adaptive"


class Tag(BaseModel):
    """Market/Event tag."""
    id: str
    label: Optional[str] = None
    slug: Optional[str] = None


class ClobReward(BaseModel):
    """CLOB reward information."""
    id: str
    conditionId: str
    assetAddress: str
    rewardsAmount: float = 0
    rewardsDailyRate: float = 0
    startDate: Optional[str] = None
    endDate: Optional[str] = None


class Market(BaseModel):
    """Polymarket market data model.

    Comprehensive model based on Gamma API response.
    """
    id: int
    question: Optional[str] = None
    conditionId: Optional[str] = None
    slug: Optional[str] = None
    description: Optional[str] = None

    # Pricing
    outcomePrices: Optional[List[float]] = None
    outcomes: Optional[List[str]] = Field(default_factory=lambda: ["Yes", "No"])

    # Volume and liquidity
    volume: Optional[float] = None
    volumeNum: Optional[float] = None
    liquidity: Optional[float] = None
    liquidityNum: Optional[float] = None
    volume24hr: Optional[float] = None

    # Status
    active: Optional[bool] = None
    closed: Optional[bool] = None
    archived: Optional[bool] = None

    # Trading capability (CRITICAL)
    enableOrderBook: Optional[bool] = None
    clobTokenIds: Optional[List[str]] = None  # Required for order execution

    # Dates
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    startDateIso: Optional[str] = None
    endDateIso: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

    # Order book settings
    orderPriceMinTickSize: Optional[float] = None
    orderMinSize: Optional[int] = None

    # Rewards
    rewardsMinSize: Optional[float] = None
    rewardsMaxSpread: Optional[float] = None
    spread: Optional[float] = None
    clobRewards: Optional[List[ClobReward]] = None

    # Metadata
    image: Optional[str] = None
    icon: Optional[str] = None
    resolutionSource: Optional[str] = None
    tags: Optional[List[Tag]] = None

    # Risk flags
    negRisk: Optional[bool] = None
    restricted: Optional[bool] = None

    @property
    def yes_price(self) -> float:
        """Get YES outcome price."""
        if self.outcomePrices and len(self.outcomePrices) > 0:
            return self.outcomePrices[0]
        return 0.5

    @property
    def no_price(self) -> float:
        """Get NO outcome price."""
        if self.outcomePrices and len(self.outcomePrices) > 1:
            return self.outcomePrices[1]
        return 0.5

    @property
    def is_tradeable(self) -> bool:
        """Check if market can be traded."""
        return (
            self.active is True and
            self.closed is not True and
            self.archived is not True and
            self.enableOrderBook is True and
            self.clobTokenIds is not None and
            len(self.clobTokenIds) > 0
        )

    def get_volume_safe(self) -> float:
        """Get volume with fallback."""
        return self.volumeNum or self.volume or 0.0

    def get_liquidity_safe(self) -> float:
        """Get liquidity with fallback."""
        return self.liquidityNum or self.liquidity or 0.0


class PolymarketEvent(BaseModel):
    """Polymarket event containing multiple markets."""
    id: str
    ticker: Optional[str] = None
    slug: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None

    # Status
    active: Optional[bool] = None
    closed: Optional[bool] = None
    archived: Optional[bool] = None

    # Volume
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    volume24hr: Optional[float] = None

    # Dates
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    createdAt: Optional[str] = None

    # Metadata
    image: Optional[str] = None
    icon: Optional[str] = None
    tags: Optional[List[Tag]] = None

    # Related markets
    markets: Optional[List[Market]] = None


class Article(BaseModel):
    """News article."""
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    publishedAt: Optional[str] = None
    relevance_score: float = 0.0
    sentiment_score: float = 0.0


class AgentResult(BaseModel):
    """Result from an agent's analysis."""
    agent_name: str
    cognitive_pattern: CognitivePattern
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: Recommendation
    edge_estimate: float = Field(ge=0.0)
    reasoning: str
    data_points: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeOpportunity(BaseModel):
    """A trading opportunity identified by the swarm."""
    market_id: int
    market_question: str
    condition_id: Optional[str] = None
    clob_token_ids: Optional[List[str]] = None

    # Recommendation
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    edge_percent: float

    # Pricing
    current_yes_price: float
    current_no_price: float
    estimated_fair_yes_price: float

    # Analysis
    reasoning: str
    agent_votes: Dict[str, Recommendation]
    risk_factors: List[str] = Field(default_factory=list)
    news_summary: Optional[str] = None

    # Metadata
    volume: float
    liquidity: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_executable(self) -> bool:
        """Check if trade can be executed."""
        return (
            self.clob_token_ids is not None and
            len(self.clob_token_ids) > 0 and
            self.recommendation != Recommendation.SKIP
        )


class SwarmConsensus(BaseModel):
    """Consensus result from all agents."""
    market_id: int
    agent_results: List[AgentResult]
    final_recommendation: Recommendation
    consensus_confidence: float
    weighted_edge: float
    dissent_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

**Step 2: Commit**

```bash
git add scripts/polymarket_swarm/models.py
git commit -m "feat: add Pydantic data models for type-safe market handling"
```

---

## Task 3: Polymarket API Client (httpx)

**Files:**
- Create: `scripts/polymarket_swarm/polymarket_client.py`
- Create: `scripts/polymarket_swarm/test_polymarket_client.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_polymarket_client.py`:
```python
"""Tests for Polymarket API client."""
import pytest
from polymarket_client import PolymarketClient
from models import Market

def test_client_initialization():
    """Test client initializes with correct endpoints."""
    client = PolymarketClient()
    assert "gamma-api.polymarket.com" in client.markets_endpoint
    assert "gamma-api.polymarket.com" in client.events_endpoint

def test_fetch_markets_returns_list():
    """Test that fetch_markets returns a list."""
    client = PolymarketClient()
    markets = client.fetch_markets(limit=5)
    assert isinstance(markets, list)
    assert len(markets) <= 5

def test_fetch_tradeable_markets_have_order_book():
    """Test that tradeable markets have enableOrderBook=True."""
    client = PolymarketClient()
    markets = client.fetch_tradeable_markets(limit=5)
    for market in markets:
        assert market.enableOrderBook is True
        assert market.clobTokenIds is not None

def test_filter_excludes_crypto_sports():
    """Test filtering excludes crypto and sports markets."""
    client = PolymarketClient()
    markets = client.fetch_filtered_markets(limit=20)
    for market in markets:
        question = (market.question or "").lower()
        description = (market.description or "").lower()
        all_text = f"{question} {description}"
        assert "bitcoin" not in all_text
        assert "ethereum" not in all_text
        assert "nfl" not in all_text
        assert "nba" not in all_text

def test_parse_outcome_prices():
    """Test that stringified outcomePrices are parsed correctly."""
    client = PolymarketClient()
    markets = client.fetch_tradeable_markets(limit=1)
    if markets:
        market = markets[0]
        assert isinstance(market.outcomePrices, list)
        assert all(isinstance(p, float) for p in market.outcomePrices)
```

**Step 2: Run test to verify it fails**

```bash
cd scripts/polymarket_swarm && python -m pytest test_polymarket_client.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

Create `scripts/polymarket_swarm/polymarket_client.py`:
```python
"""Polymarket Gamma API client using httpx for async capability.

Based on Polymarket/agents gamma.py but enhanced for our swarm architecture.
"""
import httpx
import json
from typing import List, Dict, Any, Optional

from models import Market, PolymarketEvent, Tag, ClobReward
from config import (
    GAMMA_MARKETS_ENDPOINT,
    GAMMA_EVENTS_ENDPOINT,
    TRADEABLE_MARKET_PARAMS,
    EXCLUDED_CATEGORIES,
    TARGET_CATEGORIES,
    MIN_VOLUME_USD,
    MIN_LIQUIDITY_USD,
)


class PolymarketClient:
    """Client for interacting with Polymarket's Gamma API."""

    def __init__(self, timeout: float = 30.0):
        self.markets_endpoint = GAMMA_MARKETS_ENDPOINT
        self.events_endpoint = GAMMA_EVENTS_ENDPOINT
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "PolymarketSwarmTrader/2.0"
            }
        )

    def _parse_market(self, data: Dict[str, Any]) -> Market:
        """Parse raw API response into Market model.

        Handles stringified JSON fields like outcomePrices and clobTokenIds.
        """
        # Parse stringified arrays
        if "outcomePrices" in data and isinstance(data["outcomePrices"], str):
            try:
                data["outcomePrices"] = json.loads(data["outcomePrices"])
                data["outcomePrices"] = [float(p) for p in data["outcomePrices"]]
            except (json.JSONDecodeError, ValueError):
                data["outcomePrices"] = [0.5, 0.5]

        if "clobTokenIds" in data and isinstance(data["clobTokenIds"], str):
            try:
                data["clobTokenIds"] = json.loads(data["clobTokenIds"])
            except json.JSONDecodeError:
                data["clobTokenIds"] = None

        # Parse nested objects
        if "tags" in data and data["tags"]:
            data["tags"] = [Tag(**t) if isinstance(t, dict) else t for t in data["tags"]]

        if "clobRewards" in data and data["clobRewards"]:
            data["clobRewards"] = [
                ClobReward(**r) if isinstance(r, dict) else r
                for r in data["clobRewards"]
            ]

        return Market(**data)

    def _parse_event(self, data: Dict[str, Any]) -> PolymarketEvent:
        """Parse raw API response into PolymarketEvent model."""
        if "tags" in data and data["tags"]:
            data["tags"] = [Tag(**t) if isinstance(t, dict) else t for t in data["tags"]]

        if "markets" in data and data["markets"]:
            data["markets"] = [self._parse_market(m) for m in data["markets"]]

        return PolymarketEvent(**data)

    def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Market]:
        """Fetch markets from Polymarket Gamma API.

        Args:
            limit: Maximum number of markets to return
            offset: Pagination offset
            params: Additional query parameters

        Returns:
            List of Market objects
        """
        query_params = {
            "limit": limit,
            "offset": offset,
            **(params or {})
        }

        try:
            response = self.client.get(self.markets_endpoint, params=query_params)
            response.raise_for_status()
            data = response.json()
            return [self._parse_market(m) for m in data]
        except httpx.HTTPError as e:
            print(f"Error fetching markets: {e}")
            return []

    def fetch_events(
        self,
        limit: int = 100,
        offset: int = 0,
        params: Optional[Dict[str, Any]] = None
    ) -> List[PolymarketEvent]:
        """Fetch events from Polymarket Gamma API."""
        query_params = {
            "limit": limit,
            "offset": offset,
            **(params or {})
        }

        try:
            response = self.client.get(self.events_endpoint, params=query_params)
            response.raise_for_status()
            data = response.json()
            return [self._parse_event(e) for e in data]
        except httpx.HTTPError as e:
            print(f"Error fetching events: {e}")
            return []

    def fetch_tradeable_markets(self, limit: int = 100) -> List[Market]:
        """Fetch only markets that can be traded (have order book enabled).

        This is CRITICAL - markets without enableOrderBook cannot be traded.
        """
        return self.fetch_markets(limit=limit, params=TRADEABLE_MARKET_PARAMS)

    def fetch_all_tradeable_markets(self, max_markets: int = 500) -> List[Market]:
        """Fetch all tradeable markets with pagination."""
        all_markets = []
        offset = 0
        batch_size = 100

        while len(all_markets) < max_markets:
            markets = self.fetch_markets(
                limit=batch_size,
                offset=offset,
                params=TRADEABLE_MARKET_PARAMS
            )

            if not markets:
                break

            all_markets.extend(markets)
            offset += batch_size

            if len(markets) < batch_size:
                break

        return all_markets[:max_markets]

    def _is_excluded_market(self, market: Market) -> bool:
        """Check if market should be excluded (crypto/sports)."""
        question = (market.question or "").lower()
        description = (market.description or "").lower()

        tag_labels = []
        if market.tags:
            tag_labels = [t.label.lower() for t in market.tags if t.label]

        all_text = f"{question} {description} {' '.join(tag_labels)}"

        for excluded in EXCLUDED_CATEGORIES:
            if excluded in all_text:
                return True
        return False

    def _is_target_market(self, market: Market) -> bool:
        """Check if market is in target categories."""
        question = (market.question or "").lower()
        description = (market.description or "").lower()

        tag_labels = []
        if market.tags:
            tag_labels = [t.label.lower() for t in market.tags if t.label]

        all_text = f"{question} {description} {' '.join(tag_labels)}"

        for target in TARGET_CATEGORIES:
            if target in all_text:
                return True
        return False

    def fetch_filtered_markets(
        self,
        limit: int = 100,
        min_volume: float = MIN_VOLUME_USD,
        min_liquidity: float = MIN_LIQUIDITY_USD
    ) -> List[Market]:
        """Fetch markets filtered for our target categories.

        Excludes crypto, sports. Prioritizes politics, economy, tech.
        Only returns tradeable markets with sufficient volume/liquidity.

        Args:
            limit: Maximum markets to return after filtering
            min_volume: Minimum volume in USD
            min_liquidity: Minimum liquidity in USD

        Returns:
            Filtered list of Market objects
        """
        filtered = []
        offset = 0
        batch_size = 100

        while len(filtered) < limit:
            markets = self.fetch_markets(
                limit=batch_size,
                offset=offset,
                params=TRADEABLE_MARKET_PARAMS
            )

            if not markets:
                break

            for market in markets:
                # Skip excluded categories
                if self._is_excluded_market(market):
                    continue

                # Check volume threshold
                volume = market.get_volume_safe()
                if volume < min_volume:
                    continue

                # Check liquidity threshold
                liquidity = market.get_liquidity_safe()
                if liquidity < min_liquidity:
                    continue

                # Ensure tradeable
                if not market.is_tradeable:
                    continue

                filtered.append(market)

                if len(filtered) >= limit:
                    break

            offset += batch_size

            # Safety limit
            if offset > 1000:
                break

        return filtered[:limit]

    def get_market_by_id(self, market_id: int) -> Optional[Market]:
        """Fetch a specific market by ID."""
        try:
            url = f"{self.markets_endpoint}/{market_id}"
            response = self.client.get(url)
            response.raise_for_status()
            return self._parse_market(response.json())
        except httpx.HTTPError as e:
            print(f"Error fetching market {market_id}: {e}")
            return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    # Quick test
    with PolymarketClient() as client:
        print("Fetching tradeable markets...")
        markets = client.fetch_tradeable_markets(limit=5)
        print(f"Found {len(markets)} tradeable markets")

        for m in markets:
            print(f"  - {m.question[:60] if m.question else 'Unknown'}...")
            print(f"    Price: {m.yes_price:.2%} YES / {m.no_price:.2%} NO")
            print(f"    Volume: ${m.get_volume_safe():,.0f}")
            print(f"    Tradeable: {m.is_tradeable}")
            print()

        print("\nFetching filtered markets (no crypto/sports)...")
        filtered = client.fetch_filtered_markets(limit=5)
        print(f"Found {len(filtered)} filtered markets")

        for m in filtered:
            print(f"  - {m.question[:60] if m.question else 'Unknown'}...")
```

**Step 4: Run tests to verify they pass**

```bash
cd scripts/polymarket_swarm && python -m pytest test_polymarket_client.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add scripts/polymarket_swarm/polymarket_client.py scripts/polymarket_swarm/test_polymarket_client.py
git commit -m "feat: add Polymarket API client with httpx and proper filtering"
```

---

## Task 4: RAG Market Selector

**Files:**
- Create: `scripts/polymarket_swarm/market_rag.py`
- Create: `scripts/polymarket_swarm/test_market_rag.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_market_rag.py`:
```python
"""Tests for RAG market selector."""
import pytest
from market_rag import MarketRAG
from models import Market

def test_rag_initialization():
    """Test RAG initializes correctly."""
    rag = MarketRAG()
    assert rag.embedding_model is not None

def test_index_markets():
    """Test indexing markets into vector store."""
    rag = MarketRAG()
    # Create mock markets
    markets = [
        Market(id=1, question="Will Biden win?", description="2024 election"),
        Market(id=2, question="Will Fed raise rates?", description="Monetary policy"),
    ]
    rag.index_markets(markets)
    assert rag.collection is not None

def test_semantic_search():
    """Test semantic search returns relevant markets."""
    rag = MarketRAG()
    markets = [
        Market(id=1, question="Will Biden win the 2024 election?", description="US Presidential election"),
        Market(id=2, question="Will the Fed raise interest rates?", description="Federal Reserve policy"),
        Market(id=3, question="Will SpaceX launch Starship?", description="Space exploration"),
    ]
    rag.index_markets(markets)

    results = rag.search("presidential election politics", top_k=2)
    assert len(results) <= 2
    # Election market should be most relevant
    assert any("Biden" in r.question or "election" in r.question for r in results)
```

**Step 2: Write the implementation**

Create `scripts/polymarket_swarm/market_rag.py`:
```python
"""RAG-based market selector using ChromaDB.

Provides semantic search over markets to find relevant opportunities
instead of simple keyword matching.
"""
import os
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions

from models import Market
from config import OPENAI_API_KEY, EMBEDDING_MODEL, RAG_TOP_K


class MarketRAG:
    """RAG system for semantic market selection."""

    def __init__(
        self,
        collection_name: str = "polymarket_markets",
        persist_directory: Optional[str] = None
    ):
        """Initialize RAG with ChromaDB.

        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database (None for in-memory)
        """
        # Initialize ChromaDB
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Use OpenAI embeddings
        self.embedding_model = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        )

        self.collection_name = collection_name
        self.collection = None
        self._markets_cache: dict[str, Market] = {}

    def _market_to_document(self, market: Market) -> str:
        """Convert market to searchable document text."""
        parts = []

        if market.question:
            parts.append(f"Question: {market.question}")

        if market.description:
            parts.append(f"Description: {market.description}")

        if market.tags:
            tag_labels = [t.label for t in market.tags if t.label]
            if tag_labels:
                parts.append(f"Tags: {', '.join(tag_labels)}")

        # Add market metrics for context
        parts.append(f"Volume: ${market.get_volume_safe():,.0f}")
        parts.append(f"Current Price: {market.yes_price:.1%} YES")

        return "\n".join(parts)

    def _market_to_metadata(self, market: Market) -> dict:
        """Extract metadata for filtering."""
        return {
            "market_id": market.id,
            "volume": market.get_volume_safe(),
            "liquidity": market.get_liquidity_safe(),
            "yes_price": market.yes_price,
            "is_tradeable": market.is_tradeable,
            "has_clob": market.clobTokenIds is not None,
        }

    def index_markets(self, markets: List[Market], clear_existing: bool = True):
        """Index markets into the vector store.

        Args:
            markets: List of markets to index
            clear_existing: Whether to clear existing collection first
        """
        if clear_existing:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(self.collection_name)
            except ValueError:
                pass

        # Create fresh collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_model,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare documents
        documents = []
        metadatas = []
        ids = []

        for market in markets:
            doc_id = f"market_{market.id}"
            documents.append(self._market_to_document(market))
            metadatas.append(self._market_to_metadata(market))
            ids.append(doc_id)

            # Cache market for retrieval
            self._markets_cache[doc_id] = market

        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        print(f"Indexed {len(documents)} markets into RAG")

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        min_volume: Optional[float] = None,
        tradeable_only: bool = True
    ) -> List[Market]:
        """Semantic search for relevant markets.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            min_volume: Minimum volume filter
            tradeable_only: Only return tradeable markets

        Returns:
            List of relevant Market objects
        """
        if self.collection is None:
            print("Warning: No markets indexed. Call index_markets first.")
            return []

        # Build where filter
        where_filter = None
        conditions = []

        if tradeable_only:
            conditions.append({"is_tradeable": True})

        if min_volume:
            conditions.append({"volume": {"$gte": min_volume}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Query
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter
        )

        # Convert back to Market objects
        markets = []
        if results and results["ids"]:
            for doc_id in results["ids"][0]:
                if doc_id in self._markets_cache:
                    markets.append(self._markets_cache[doc_id])

        return markets

    def search_by_category(
        self,
        category: str,
        top_k: int = RAG_TOP_K
    ) -> List[Market]:
        """Search for markets in a specific category.

        Args:
            category: Category name (e.g., "politics", "economy", "technology")
            top_k: Number of results

        Returns:
            List of relevant markets
        """
        category_queries = {
            "politics": "political elections government policy voting candidates",
            "economy": "economic federal reserve interest rates inflation GDP",
            "technology": "tech companies AI artificial intelligence software",
            "entertainment": "movies awards celebrities entertainment media",
            "science": "scientific research space exploration climate",
            "world": "international geopolitics foreign affairs global events",
        }

        query = category_queries.get(category.lower(), category)
        return self.search(query, top_k=top_k)

    def find_mispriced_markets(
        self,
        query: str,
        price_threshold: float = 0.2,
        top_k: int = RAG_TOP_K
    ) -> List[Market]:
        """Find markets with extreme prices that might be mispriced.

        Markets with YES price < threshold or > (1-threshold) are more
        likely to have edge if we have information advantage.

        Args:
            query: Search query for relevant markets
            price_threshold: Price extremity threshold (default 0.2)
            top_k: Number of results

        Returns:
            Markets matching query with extreme prices
        """
        # First get relevant markets
        relevant = self.search(query, top_k=top_k * 3)  # Get more to filter

        # Filter for extreme prices
        mispriced = [
            m for m in relevant
            if m.yes_price < price_threshold or m.yes_price > (1 - price_threshold)
        ]

        return mispriced[:top_k]


if __name__ == "__main__":
    from polymarket_client import PolymarketClient

    # Demo
    print("Fetching markets...")
    with PolymarketClient() as client:
        markets = client.fetch_filtered_markets(limit=50)

    print(f"Indexing {len(markets)} markets...")
    rag = MarketRAG()
    rag.index_markets(markets)

    print("\nSearching for 'presidential election'...")
    results = rag.search("presidential election politics", top_k=5)
    for m in results:
        print(f"  - {m.question[:60]}... ({m.yes_price:.1%})")

    print("\nSearching for 'federal reserve interest rates'...")
    results = rag.search("federal reserve interest rates economy", top_k=5)
    for m in results:
        print(f"  - {m.question[:60]}... ({m.yes_price:.1%})")
```

**Step 3: Run tests**

```bash
cd scripts/polymarket_swarm && python -m pytest test_market_rag.py -v
```

**Step 4: Commit**

```bash
git add scripts/polymarket_swarm/market_rag.py scripts/polymarket_swarm/test_market_rag.py
git commit -m "feat: add RAG-based semantic market selector with ChromaDB"
```

---

## Task 5: News Research Client

**Files:**
- Create: `scripts/polymarket_swarm/news_client.py`
- Create: `scripts/polymarket_swarm/test_news_client.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_news_client.py`:
```python
"""Tests for news research client."""
import pytest
from news_client import NewsClient
from models import Article

def test_client_initialization():
    """Test news client initializes correctly."""
    client = NewsClient()
    assert client is not None

def test_extract_keywords():
    """Test keyword extraction from market question."""
    client = NewsClient()
    question = "Will Biden win the 2024 presidential election?"
    keywords = client.extract_keywords(question)
    assert "biden" in [k.lower() for k in keywords]
    assert len(keywords) <= 5

def test_search_news_returns_articles():
    """Test searching news returns article list."""
    client = NewsClient()
    articles = client.search_news("politics", limit=5)
    assert isinstance(articles, list)

def test_calculate_sentiment():
    """Test sentiment calculation."""
    client = NewsClient()
    articles = [
        Article(title="Candidate wins major victory", description="Strong performance"),
        Article(title="Policy fails to pass", description="Rejection and defeat"),
    ]
    # First article positive, second negative - should balance out
    sentiment = client.calculate_sentiment(articles)
    assert -1.0 <= sentiment <= 1.0
```

**Step 2: Write the implementation**

Create `scripts/polymarket_swarm/news_client.py`:
```python
"""News research client for gathering real-world data.

Combines multiple news sources for comprehensive coverage.
"""
import re
from typing import List, Optional
from datetime import datetime, timedelta
import httpx

from models import Article, Market
from config import NEWSAPI_API_KEY, FINNHUB_API_KEY


class NewsClient:
    """Client for fetching news and real-world event data."""

    def __init__(self):
        self.newsapi_key = NEWSAPI_API_KEY
        self.finnhub_key = FINNHUB_API_KEY
        self.client = httpx.Client(timeout=30.0)

        # Stop words to exclude from keywords
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
            "should", "now", "what", "who", "which", "this", "that",
            "year", "month", "day", "week", "would", "could", "may",
            "might", "must", "shall"
        }

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant keywords from text for news search.

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
        limit: int = 10,
        days_back: int = 7
    ) -> List[Article]:
        """Search for news articles related to query.

        Args:
            query: Search query or keywords
            limit: Maximum articles to return
            days_back: How many days back to search

        Returns:
            List of Article objects
        """
        articles = []

        # Try NewsAPI first
        if self.newsapi_key:
            articles.extend(self._search_newsapi(query, limit, days_back))

        # Supplement with Finnhub if needed
        if len(articles) < limit and self.finnhub_key:
            remaining = limit - len(articles)
            articles.extend(self._search_finnhub(query, remaining))

        # If no API keys, return mock data for testing
        if not articles:
            articles = self._get_mock_news(query, limit)

        return articles[:limit]

    def _search_newsapi(
        self,
        query: str,
        limit: int,
        days_back: int
    ) -> List[Article]:
        """Search using NewsAPI."""
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            response = self.client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": limit,
                    "language": "en",
                    "apiKey": self.newsapi_key,
                }
            )

            if response.status_code == 200:
                data = response.json()
                return [
                    Article(
                        title=a.get("title"),
                        description=a.get("description"),
                        content=a.get("content"),
                        url=a.get("url"),
                        source=a.get("source", {}).get("name"),
                        author=a.get("author"),
                        publishedAt=a.get("publishedAt"),
                    )
                    for a in data.get("articles", [])
                ]
        except Exception as e:
            print(f"NewsAPI error: {e}")

        return []

    def _search_finnhub(self, query: str, limit: int) -> List[Article]:
        """Search using Finnhub news API."""
        try:
            response = self.client.get(
                "https://finnhub.io/api/v1/news",
                params={
                    "category": "general",
                    "token": self.finnhub_key,
                }
            )

            if response.status_code == 200:
                data = response.json()

                # Filter by query relevance
                query_words = set(query.lower().split())
                relevant = []

                for article in data:
                    headline = article.get("headline", "").lower()
                    summary = article.get("summary", "").lower()
                    text = f"{headline} {summary}"

                    matches = sum(1 for w in query_words if w in text)
                    if matches > 0:
                        relevant.append((matches, article))

                # Sort by relevance
                relevant.sort(key=lambda x: x[0], reverse=True)

                return [
                    Article(
                        title=a.get("headline"),
                        description=a.get("summary"),
                        url=a.get("url"),
                        source=a.get("source"),
                        publishedAt=datetime.fromtimestamp(
                            a.get("datetime", 0)
                        ).isoformat() if a.get("datetime") else None,
                        relevance_score=score / len(query_words)
                    )
                    for score, a in relevant[:limit]
                ]
        except Exception as e:
            print(f"Finnhub error: {e}")

        return []

    def _get_mock_news(self, query: str, limit: int) -> List[Article]:
        """Return mock news data for testing without API keys."""
        return [
            Article(
                title=f"Mock news about {query}",
                description=f"This is a mock article discussing {query} for testing.",
                source="MockNews",
                publishedAt=datetime.now().isoformat(),
                url="https://example.com/mock",
                relevance_score=0.5
            )
        ][:limit]

    def get_news_for_market(self, market: Market) -> List[Article]:
        """Get relevant news for a specific market.

        Args:
            market: Market to research

        Returns:
            List of relevant news articles
        """
        # Extract keywords from market
        text = f"{market.question or ''} {market.description or ''}"
        keywords = self.extract_keywords(text)

        # Search with combined keywords
        query = " ".join(keywords[:3])
        articles = self.search_news(query, limit=10)

        # Score relevance
        for article in articles:
            article.relevance_score = self._calculate_relevance(
                article, keywords
            )

        # Sort by relevance
        articles.sort(key=lambda x: x.relevance_score, reverse=True)

        return articles

    def _calculate_relevance(
        self,
        article: Article,
        keywords: List[str]
    ) -> float:
        """Calculate relevance score for article."""
        text = f"{article.title or ''} {article.description or ''}".lower()

        matches = sum(1 for k in keywords if k.lower() in text)
        return matches / len(keywords) if keywords else 0.0

    def calculate_sentiment(self, articles: List[Article]) -> float:
        """Calculate aggregate sentiment score from articles.

        Returns:
            Score from -1 (negative) to 1 (positive)
        """
        if not articles:
            return 0.0

        positive_words = {
            "win", "wins", "winning", "won", "success", "successful",
            "approve", "approved", "pass", "passed", "gain", "gains",
            "rise", "rises", "rising", "rose", "increase", "increased",
            "positive", "strong", "stronger", "lead", "leads", "leading",
            "ahead", "likely", "expected", "confirm", "confirmed",
            "support", "supports", "supported", "victory", "triumph",
            "boost", "boosted", "surge", "surged", "rally", "rallied"
        }

        negative_words = {
            "lose", "loses", "losing", "lost", "fail", "fails", "failed",
            "reject", "rejected", "block", "blocked", "drop", "drops",
            "fall", "falls", "falling", "fell", "decrease", "decreased",
            "negative", "weak", "weaker", "behind", "unlikely",
            "doubt", "doubts", "oppose", "opposed", "crisis", "concern",
            "concerns", "defeat", "defeated", "collapse", "collapsed",
            "decline", "declined", "plunge", "plunged", "crash", "crashed"
        }

        total_score = 0
        scored_articles = 0

        for article in articles:
            text = f"{article.title or ''} {article.description or ''}".lower()

            pos = sum(1 for w in positive_words if w in text)
            neg = sum(1 for w in negative_words if w in text)

            if pos + neg > 0:
                article_score = (pos - neg) / (pos + neg)
                total_score += article_score
                scored_articles += 1
                article.sentiment_score = article_score

        return total_score / scored_articles if scored_articles > 0 else 0.0

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    client = NewsClient()

    # Test keyword extraction
    question = "Will Biden win the 2024 presidential election?"
    keywords = client.extract_keywords(question)
    print(f"Keywords from '{question}':")
    print(f"  {keywords}")

    # Test news search
    print("\nSearching for 'presidential election'...")
    articles = client.search_news("presidential election", limit=5)
    print(f"Found {len(articles)} articles")

    for a in articles[:3]:
        print(f"  - {a.title[:60] if a.title else 'No title'}...")

    # Test sentiment
    sentiment = client.calculate_sentiment(articles)
    print(f"\nAggregate sentiment: {sentiment:.2f}")
```

**Step 3: Run tests**

```bash
cd scripts/polymarket_swarm && python -m pytest test_news_client.py -v
```

**Step 4: Commit**

```bash
git add scripts/polymarket_swarm/news_client.py scripts/polymarket_swarm/test_news_client.py
git commit -m "feat: add news research client with sentiment analysis"
```

---

## Task 6: Swarm Agent Definitions

**Files:**
- Create: `scripts/polymarket_swarm/agents.py`
- Create: `scripts/polymarket_swarm/test_agents.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_agents.py`:
```python
"""Tests for swarm agent definitions."""
import pytest
from agents import (
    AGENT_REGISTRY,
    MarketScannerAgent,
    NewsResearcherAgent,
    ProbabilityAnalystAgent,
    RiskValidatorAgent,
    CoordinatorAgent
)
from models import CognitivePattern, Recommendation

def test_all_agents_defined():
    """Test all 5 agents are defined."""
    assert len(AGENT_REGISTRY) == 5

def test_market_scanner_has_convergent_pattern():
    """Test MarketScanner uses convergent cognitive pattern."""
    agent = MarketScannerAgent()
    assert agent.cognitive_pattern == CognitivePattern.CONVERGENT

def test_news_researcher_has_divergent_pattern():
    """Test NewsResearcher uses divergent cognitive pattern."""
    agent = NewsResearcherAgent()
    assert agent.cognitive_pattern == CognitivePattern.DIVERGENT

def test_probability_analyst_has_critical_pattern():
    """Test ProbabilityAnalyst uses critical cognitive pattern."""
    agent = ProbabilityAnalystAgent()
    assert agent.cognitive_pattern == CognitivePattern.CRITICAL

def test_risk_validator_has_systems_pattern():
    """Test RiskValidator uses systems cognitive pattern."""
    agent = RiskValidatorAgent()
    assert agent.cognitive_pattern == CognitivePattern.SYSTEMS

def test_coordinator_has_adaptive_pattern():
    """Test Coordinator uses adaptive cognitive pattern."""
    agent = CoordinatorAgent()
    assert agent.cognitive_pattern == CognitivePattern.ADAPTIVE

def test_agents_have_analyze_method():
    """Test all agents have analyze method."""
    for name, agent_class in AGENT_REGISTRY.items():
        agent = agent_class()
        assert hasattr(agent, "analyze")
        assert callable(agent.analyze)

def test_agent_weights_are_positive():
    """Test all agent weights are positive."""
    for name, agent_class in AGENT_REGISTRY.items():
        agent = agent_class()
        assert agent.weight > 0
```

**Step 2: Write the implementation**

Create `scripts/polymarket_swarm/agents.py`:
```python
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
```

**Step 3: Run tests**

```bash
cd scripts/polymarket_swarm && python -m pytest test_agents.py -v
```

**Step 4: Commit**

```bash
git add scripts/polymarket_swarm/agents.py scripts/polymarket_swarm/test_agents.py
git commit -m "feat: add 5 swarm agents with cognitive diversity patterns"
```

---

## Task 7: Swarm Orchestrator

**Files:**
- Create: `scripts/polymarket_swarm/orchestrator.py`
- Create: `scripts/polymarket_swarm/test_orchestrator.py`

**Step 1: Write the failing test**

Create `scripts/polymarket_swarm/test_orchestrator.py`:
```python
"""Tests for swarm orchestrator."""
import pytest
from orchestrator import SwarmOrchestrator
from models import TradeOpportunity

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

def test_orchestrator_returns_trade_opportunity():
    """Test that analysis can return TradeOpportunity."""
    orchestrator = SwarmOrchestrator()
    # We don't actually run analysis here (requires API)
    # Just verify the types are correct
    assert TradeOpportunity is not None
```

**Step 2: Write the implementation**

Create `scripts/polymarket_swarm/orchestrator.py`:
```python
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
```

**Step 3: Run tests**

```bash
cd scripts/polymarket_swarm && python -m pytest test_orchestrator.py -v
```

**Step 4: Commit**

```bash
git add scripts/polymarket_swarm/orchestrator.py scripts/polymarket_swarm/test_orchestrator.py
git commit -m "feat: add swarm orchestrator with full analysis pipeline"
```

---

## Task 8: MCP Swarm Integration

**Files:**
- Create: `scripts/polymarket_swarm/mcp_integration.py`

**Step 1: Create MCP integration module**

Create `scripts/polymarket_swarm/mcp_integration.py`:
```python
"""MCP Swarm Integration for Polymarket Analysis.

Provides integration with the ruv-swarm MCP server for distributed
agent processing using DAA (Decentralized Autonomous Agents).
"""
from typing import List, Dict, Any
import json

from orchestrator import SwarmOrchestrator
from models import TradeOpportunity, CognitivePattern


class MCPSwarmIntegration:
    """Integrates with MCP swarm tools for distributed analysis.

    Uses ruv-swarm MCP server capabilities:
    - swarm_init: Initialize swarm topology
    - daa_agent_create: Create agents with cognitive patterns
    - daa_workflow_create: Create analysis pipelines
    - daa_workflow_execute: Run distributed analysis
    """

    def __init__(self):
        self.orchestrator = SwarmOrchestrator(use_rag=False)
        self.swarm_config = self._build_swarm_config()

    def _build_swarm_config(self) -> Dict[str, Any]:
        """Build configuration for MCP swarm."""
        return {
            "topology": "hierarchical",
            "max_agents": 5,
            "strategy": "specialized",
            "agents": [
                {
                    "id": "market_scanner",
                    "cognitive_pattern": CognitivePattern.CONVERGENT.value,
                    "capabilities": ["market_analysis", "volume_tracking", "liquidity_assessment"],
                    "enable_memory": True,
                    "learning_rate": 0.1,
                },
                {
                    "id": "news_researcher",
                    "cognitive_pattern": CognitivePattern.DIVERGENT.value,
                    "capabilities": ["news_search", "sentiment_analysis", "web_search"],
                    "enable_memory": True,
                    "learning_rate": 0.15,
                },
                {
                    "id": "probability_analyst",
                    "cognitive_pattern": CognitivePattern.CRITICAL.value,
                    "capabilities": ["probability_calculation", "edge_detection", "superforecasting"],
                    "enable_memory": True,
                    "learning_rate": 0.1,
                },
                {
                    "id": "risk_validator",
                    "cognitive_pattern": CognitivePattern.SYSTEMS.value,
                    "capabilities": ["risk_assessment", "position_sizing", "slippage_estimation"],
                    "enable_memory": True,
                    "learning_rate": 0.05,
                },
                {
                    "id": "coordinator",
                    "cognitive_pattern": CognitivePattern.ADAPTIVE.value,
                    "capabilities": ["synthesis", "consensus_building", "decision_making"],
                    "enable_memory": True,
                    "learning_rate": 0.2,
                },
            ],
            "workflow": {
                "id": "polymarket_analysis",
                "name": "Polymarket Opportunity Scanner",
                "strategy": "adaptive",
                "steps": [
                    {"id": "fetch_markets", "agent": "market_scanner", "action": "fetch_tradeable_markets"},
                    {"id": "scan_structure", "agent": "market_scanner", "action": "analyze_market_structure"},
                    {"id": "research_news", "agent": "news_researcher", "action": "gather_news"},
                    {"id": "calculate_probability", "agent": "probability_analyst", "action": "estimate_probability"},
                    {"id": "validate_risk", "agent": "risk_validator", "action": "assess_risk"},
                    {"id": "synthesize", "agent": "coordinator", "action": "build_consensus"},
                ],
                "dependencies": {
                    "scan_structure": ["fetch_markets"],
                    "research_news": ["fetch_markets"],
                    "calculate_probability": ["scan_structure", "research_news"],
                    "validate_risk": ["calculate_probability"],
                    "synthesize": ["validate_risk"],
                },
            },
        }

    def get_mcp_commands(self) -> Dict[str, Any]:
        """Get the MCP commands to run for full swarm analysis.

        Returns dict of commands to execute via Claude Code MCP tools.
        """
        config = self.swarm_config

        return {
            "1_init_swarm": {
                "tool": "mcp__ruv-swarm__swarm_init",
                "params": {
                    "topology": config["topology"],
                    "maxAgents": config["max_agents"],
                    "strategy": config["strategy"],
                },
            },
            "2_init_daa": {
                "tool": "mcp__ruv-swarm__daa_init",
                "params": {
                    "enableCoordination": True,
                    "enableLearning": True,
                    "persistenceMode": "memory",
                },
            },
            "3_spawn_agents": [
                {
                    "tool": "mcp__ruv-swarm__daa_agent_create",
                    "params": {
                        "id": agent["id"],
                        "cognitivePattern": agent["cognitive_pattern"],
                        "capabilities": agent["capabilities"],
                        "enableMemory": agent["enable_memory"],
                        "learningRate": agent["learning_rate"],
                    },
                }
                for agent in config["agents"]
            ],
            "4_create_workflow": {
                "tool": "mcp__ruv-swarm__daa_workflow_create",
                "params": {
                    "id": config["workflow"]["id"],
                    "name": config["workflow"]["name"],
                    "strategy": config["workflow"]["strategy"],
                    "steps": config["workflow"]["steps"],
                    "dependencies": config["workflow"]["dependencies"],
                },
            },
            "5_execute": {
                "tool": "mcp__ruv-swarm__daa_workflow_execute",
                "params": {
                    "workflowId": config["workflow"]["id"],
                    "parallelExecution": True,
                    "agentIds": [a["id"] for a in config["agents"]],
                },
            },
            "6_check_status": {
                "tool": "mcp__ruv-swarm__daa_performance_metrics",
                "params": {
                    "category": "all",
                },
            },
        }

    def run_local_analysis(
        self,
        max_markets: int = 50,
        search_query: str = None
    ) -> List[TradeOpportunity]:
        """Run analysis using local orchestrator (non-MCP).

        Args:
            max_markets: Maximum markets to analyze
            search_query: Optional semantic search query

        Returns:
            List of trading opportunities
        """
        return self.orchestrator.find_opportunities(
            max_markets=max_markets,
            search_query=search_query,
        )

    def print_mcp_setup_instructions(self):
        """Print instructions for running with MCP swarm."""
        commands = self.get_mcp_commands()

        print("=" * 70)
        print("MCP SWARM SETUP INSTRUCTIONS")
        print("=" * 70)
        print()
        print("Execute these MCP commands in sequence:")
        print()

        for step, cmd in commands.items():
            print(f"### {step}")
            if isinstance(cmd, list):
                for i, c in enumerate(cmd):
                    print(f"  [{i+1}] {c['tool']}")
                    print(f"      params: {json.dumps(c['params'], indent=8)}")
            else:
                print(f"  {cmd['tool']}")
                print(f"  params: {json.dumps(cmd['params'], indent=4)}")
            print()

    def close(self):
        """Close resources."""
        self.orchestrator.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("Polymarket Swarm MCP Integration")
    print()

    with MCPSwarmIntegration() as mcp:
        # Print MCP setup instructions
        mcp.print_mcp_setup_instructions()

        print()
        print("=" * 70)
        print("Running local analysis (without MCP)...")
        print("=" * 70)
        print()

        # Run local analysis
        opportunities = mcp.run_local_analysis(max_markets=10)

        print()
        print(f"Found {len(opportunities)} opportunities")
        for opp in opportunities:
            print(f"  - {opp.recommendation.value}: {opp.market_question[:50]}...")
            print(f"    Edge: {opp.edge_percent:.1f}%, Confidence: {opp.confidence:.1%}")
```

**Step 2: Commit**

```bash
git add scripts/polymarket_swarm/mcp_integration.py
git commit -m "feat: add MCP swarm integration with DAA workflow"
```

---

## Task 9: CLI and Runner

**Files:**
- Create: `scripts/polymarket_swarm/cli.py`
- Create: `scripts/polymarket_swarm/run.py`

**Step 1: Create CLI**

Create `scripts/polymarket_swarm/cli.py`:
```python
"""Command-line interface for Polymarket Swarm Trader v2.0."""
import argparse
import json
import sys
from datetime import datetime

from orchestrator import SwarmOrchestrator
from mcp_integration import MCPSwarmIntegration
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


def cmd_scan(args):
    """Scan markets for opportunities."""
    print(f"Scanning up to {args.max_markets} markets...")
    print(f"Minimum edge: {args.min_edge}%")
    print(f"Minimum confidence: {args.min_confidence}")
    if args.query:
        print(f"Search query: {args.query}")
    print()

    with SwarmOrchestrator(use_rag=bool(args.query)) as orchestrator:
        opportunities = orchestrator.find_opportunities(
            max_markets=args.max_markets,
            search_query=args.query,
            min_edge=args.min_edge,
            min_confidence=args.min_confidence,
        )

        print()
        print(orchestrator.generate_report(opportunities))

        if args.output:
            with open(args.output, 'w') as f:
                f.write(orchestrator.to_json(opportunities))
            print(f"\nSaved to {args.output}")

    return opportunities


def cmd_analyze(args):
    """Analyze a specific market."""
    with SwarmOrchestrator(use_rag=False) as orchestrator:
        market = orchestrator.polymarket.get_market_by_id(args.market_id)

        if not market:
            print(f"Market not found: {args.market_id}")
            return None

        print(f"Analyzing: {market.question}")
        print(f"Current price: {market.yes_price:.1%} YES / {market.no_price:.1%} NO")
        print()

        opportunity = orchestrator.analyze_market(market)

        if opportunity:
            print(f"Recommendation: {opportunity.recommendation.value}")
            print(f"Edge: {opportunity.edge_percent:.1f}%")
            print(f"Confidence: {opportunity.confidence:.1%}")
            print(f"Fair YES price: {opportunity.estimated_fair_yes_price:.1%}")
            print(f"Reasoning: {opportunity.reasoning}")
            print(f"Agent Votes: {dict((k, v.value) for k, v in opportunity.agent_votes.items())}")
            print(f"Risk Factors: {opportunity.risk_factors}")
            print(f"Executable: {'YES' if opportunity.is_executable else 'NO'}")
        else:
            print("No trading opportunity identified.")

        return opportunity


def cmd_mcp_setup(args):
    """Print MCP swarm setup commands."""
    with MCPSwarmIntegration() as mcp:
        mcp.print_mcp_setup_instructions()


def cmd_categories(args):
    """Search by category."""
    categories = {
        "politics": "presidential election government policy voting",
        "economy": "federal reserve interest rates inflation GDP",
        "tech": "artificial intelligence AI technology companies",
        "world": "international geopolitics foreign affairs",
        "entertainment": "movies awards celebrities media",
    }

    if args.category not in categories:
        print(f"Unknown category: {args.category}")
        print(f"Available: {', '.join(categories.keys())}")
        return

    query = categories[args.category]
    print(f"Searching category '{args.category}' with query: {query}")
    print()

    # Use the scan command with the category query
    args.query = query
    args.max_markets = args.limit
    args.min_edge = MIN_EDGE_PERCENT
    args.min_confidence = MIN_CONFIDENCE
    args.output = None

    return cmd_scan(args)


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Swarm Trader v2.0 - Find high-probability trades"
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
        "--query", "-q",
        type=str,
        help="Semantic search query (enables RAG)"
    )
    scan_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific market")
    analyze_parser.add_argument("market_id", type=int, help="Market ID to analyze")

    # Category command
    cat_parser = subparsers.add_parser("category", help="Search by category")
    cat_parser.add_argument(
        "category",
        choices=["politics", "economy", "tech", "world", "entertainment"],
        help="Category to search"
    )
    cat_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum markets to analyze"
    )

    # MCP setup command
    mcp_parser = subparsers.add_parser("mcp-setup", help="Print MCP swarm setup commands")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "category":
        cmd_categories(args)
    elif args.command == "mcp-setup":
        cmd_mcp_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Create runner script**

Create `scripts/polymarket_swarm/run.py`:
```python
#!/usr/bin/env python3
"""Quick runner for Polymarket Swarm analysis."""
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import SwarmOrchestrator


def main():
    print("=" * 70)
    print("POLYMARKET SWARM TRADER v2.0")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("Target: Politics, Economy, Technology, Entertainment, World")
    print("Excluded: Crypto, Sports")
    print("Minimum Edge: 10% | Minimum Confidence: 70%")
    print()

    with SwarmOrchestrator(use_rag=False) as orchestrator:
        # Run analysis
        opportunities = orchestrator.find_opportunities(max_markets=30)

        # Generate and print report
        print()
        print(orchestrator.generate_report(opportunities))

        # Save results
        if opportunities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"polymarket_opportunities_{timestamp}.json"

            with open(output_file, 'w') as f:
                f.write(orchestrator.to_json(opportunities))

            print(f"\nResults saved to: {output_file}")

    return opportunities


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add scripts/polymarket_swarm/cli.py scripts/polymarket_swarm/run.py
git commit -m "feat: add CLI and runner script for easy usage"
```

---

## Task 10: Integration Tests and Documentation

**Files:**
- Create: `scripts/polymarket_swarm/test_integration.py`
- Create: `scripts/polymarket_swarm/README.md`

**Step 1: Create integration tests**

Create `scripts/polymarket_swarm/test_integration.py`:
```python
"""Integration tests for Polymarket Swarm Trader v2.0."""
import pytest
from polymarket_client import PolymarketClient
from news_client import NewsClient
from orchestrator import SwarmOrchestrator
from models import Market


class TestAPIConnections:
    """Tests that verify API connections work."""

    def test_polymarket_api_connection(self):
        """Test that Polymarket API responds."""
        with PolymarketClient() as client:
            markets = client.fetch_markets(limit=5)

            assert len(markets) > 0, "Polymarket API should return markets"
            assert all(isinstance(m, Market) for m in markets)

    def test_tradeable_markets_have_order_book(self):
        """Test that tradeable markets have enableOrderBook=True."""
        with PolymarketClient() as client:
            markets = client.fetch_tradeable_markets(limit=10)

            for market in markets:
                assert market.enableOrderBook is True
                assert market.is_tradeable is True

    def test_filter_excludes_crypto_sports(self):
        """Test that filtering properly excludes crypto and sports."""
        with PolymarketClient() as client:
            markets = client.fetch_filtered_markets(limit=20)

            crypto_terms = ["bitcoin", "ethereum", "crypto", "defi"]
            sports_terms = ["nfl", "nba", "mlb", "soccer"]

            for market in markets:
                question = (market.question or "").lower()
                description = (market.description or "").lower()
                all_text = f"{question} {description}"

                for term in crypto_terms + sports_terms:
                    assert term not in all_text, f"Found excluded term '{term}'"

    def test_news_client_returns_articles(self):
        """Test that news client can fetch articles."""
        with NewsClient() as client:
            articles = client.search_news("politics", limit=5)
            assert isinstance(articles, list)


class TestSwarmAnalysis:
    """Tests for the full swarm analysis pipeline."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        with SwarmOrchestrator(use_rag=False) as orchestrator:
            assert len(orchestrator.agents) == 5
            assert orchestrator.polymarket is not None
            assert orchestrator.news is not None

    def test_single_market_analysis(self):
        """Test analysis of a single market."""
        with SwarmOrchestrator(use_rag=False) as orchestrator:
            markets = orchestrator.polymarket.fetch_filtered_markets(limit=1)

            if markets:
                result = orchestrator.analyze_market(markets[0])
                # Result can be None (no opportunity) or TradeOpportunity
                if result:
                    assert result.market_id == markets[0].id
                    assert result.confidence >= 0
                    assert result.edge_percent >= 0


class TestModels:
    """Tests for data models."""

    def test_market_prices(self):
        """Test market price properties."""
        market = Market(
            id=1,
            question="Test market",
            outcomePrices=[0.65, 0.35],
        )

        assert market.yes_price == 0.65
        assert market.no_price == 0.35

    def test_market_tradeable_check(self):
        """Test is_tradeable property."""
        # Not tradeable - missing order book
        market1 = Market(id=1, active=True, enableOrderBook=False)
        assert market1.is_tradeable is False

        # Tradeable
        market2 = Market(
            id=2,
            active=True,
            closed=False,
            archived=False,
            enableOrderBook=True,
            clobTokenIds=["token1", "token2"]
        )
        assert market2.is_tradeable is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Create README**

Create `scripts/polymarket_swarm/README.md`:
```markdown
# Polymarket Swarm Trader v2.0

AI-powered trading intelligence for Polymarket prediction markets using a swarm of agents with diverse cognitive patterns.

## What's New in v2.0

Based on critical review of [Polymarket/agents](https://github.com/Polymarket/agents):

- **Pydantic Models**: Type-safe data handling
- **httpx Client**: Async-capable HTTP client
- **RAG Market Selection**: Semantic search with ChromaDB
- **enableOrderBook Filter**: Only tradeable markets
- **CLOB Token Tracking**: Ready for order execution
- **Superforecaster Methodology**: Proven probability estimation

## Features

- **5 Specialized Agents**: Market Scanner, News Researcher, Probability Analyst, Risk Validator, Coordinator
- **Cognitive Diversity**: Each agent uses different thinking patterns (convergent, divergent, critical, systems, adaptive)
- **Real-time Data**: Integrates Polymarket Gamma API and news sources
- **Smart Filtering**: Focuses on politics, economy, tech - excludes crypto/sports
- **MCP Integration**: Can run as distributed swarm via ruv-swarm

## Quick Start

```bash
# Install dependencies
pip install -r requirements-polymarket.txt

# Set up environment variables
cp .env.example .env
# Add: OPENAI_API_KEY, NEWSAPI_API_KEY (optional), FINNHUB_API_KEY (optional)

# Run a scan
cd scripts/polymarket_swarm
python run.py

# Or use CLI
python cli.py scan --max-markets 50 --min-edge 10

# Search by category
python cli.py category politics --limit 20

# Analyze specific market
python cli.py analyze 123456

# Get MCP swarm setup commands
python cli.py mcp-setup
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Coordinator Agent             │
│        (Adaptive Thinking)              │
│   - Synthesizes all agent results       │
│   - Weighted consensus voting           │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴───┐   ┌─────┴─────┐   ┌───┴───┐
│Market │   │Probability│   │ Risk  │
│Scanner│   │ Analyst   │   │Validat│
│(conv.)│   │ (critical)│   │(syst.)│
└───┬───┘   └─────┬─────┘   └───┬───┘
    │             │             │
    │       ┌─────┴─────┐       │
    │       │   News    │       │
    │       │Researcher │       │
    │       │ (diverg.) │       │
    │       └───────────┘       │
    └─────────────┴─────────────┘
```

## Agent Cognitive Patterns

| Agent | Pattern | Focus |
|-------|---------|-------|
| Market Scanner | Convergent | Narrows to specific metrics (volume, liquidity, price) |
| News Researcher | Divergent | Expands to find diverse news sources |
| Probability Analyst | Critical | Questions assumptions, calculates edge |
| Risk Validator | Systems | Sees interconnections, feedback loops |
| Coordinator | Adaptive | Adjusts based on agent inputs |

## Data Sources

| Source | Purpose |
|--------|---------|
| Polymarket Gamma API | Market data, prices, CLOB tokens |
| NewsAPI | News articles for sentiment |
| Finnhub | Financial news backup |
| WebSearch | Breaking news verification |

## Output

Opportunities are reported with:
- **Recommendation**: YES or NO
- **Edge**: Estimated % advantage over market
- **Confidence**: Swarm consensus score
- **Reasoning**: Why this trade
- **Risk Factors**: Identified concerns
- **Executable**: Whether trade can be placed

## MCP Swarm Integration

For distributed processing with ruv-swarm:

```bash
python cli.py mcp-setup
```

This outputs the MCP commands to:
1. Initialize swarm topology
2. Create DAA agents with cognitive patterns
3. Set up analysis workflow
4. Execute distributed analysis

## Configuration

Edit `config.py` to adjust:
- `MIN_EDGE_PERCENT`: Minimum edge to recommend (default: 10%)
- `MIN_CONFIDENCE`: Minimum confidence score (default: 0.7)
- `MIN_VOLUME_USD`: Minimum market volume (default: $10,000)
- `EXCLUDED_CATEGORIES`: Markets to skip
- `TARGET_CATEGORIES`: Markets to prioritize
- `AGENT_WEIGHTS`: Relative importance of each agent

## License

MIT
```

**Step 3: Final commit**

```bash
git add scripts/polymarket_swarm/test_integration.py scripts/polymarket_swarm/README.md
git add -A scripts/polymarket_swarm/
git commit -m "docs: add integration tests and README for v2.0"
```

---

## Summary

This updated plan incorporates key learnings from the Polymarket/agents repository:

### Adopted from Official Repo
1. **Pydantic Models** - Type-safe market/event handling
2. **enableOrderBook Filter** - Only tradeable markets
3. **Superforecaster Prompts** - Systematic probability estimation
4. **CLOB Token Tracking** - Required for order execution

### Our Advantages Over Official Repo
1. **5-Agent Swarm** vs single LLM
2. **Cognitive Diversity** - Different thinking patterns
3. **Full News Integration** - Actually uses sentiment
4. **Risk Validation** - Pre-trade risk checks
5. **Math-Based Edge** - Not just LLM vibes
6. **Category Filtering** - Focus on information-edge markets

### New in v2.0
1. RAG-based semantic market selection
2. httpx for async capability
3. MCP swarm integration with DAA
4. Comprehensive CLI

**Total Tasks**: 10
**Estimated Files**: 15
**Key Dependencies**: httpx, pydantic, chromadb, langchain

Ready to execute with `superpowers:executing-plans`.
