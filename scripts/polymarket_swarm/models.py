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
