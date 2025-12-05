"""Configuration for Polymarket Swarm Trader v3.0."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Endpoints
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_MARKETS_ENDPOINT = f"{GAMMA_API_BASE}/markets"
GAMMA_EVENTS_ENDPOINT = f"{GAMMA_API_BASE}/events"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Tradeable market parameters (CRITICAL: enableOrderBook for actual trading)
TRADEABLE_MARKET_PARAMS = {
    "active": True,
    "closed": False,
    "archived": False,
    "enableOrderBook": True,  # Only markets we can actually trade
}

# Filter categories (exclude these)
EXCLUDED_CATEGORIES = [
    # Crypto
    "crypto", "cryptocurrency", "bitcoin", "ethereum", "defi", "nft",
    "solana", "dogecoin", "xrp", "cardano", "token", "coin",
    # Sports - leagues
    "sports", "nfl", "nba", "mlb", "nhl", "mls", "uefa", "fifa",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "world cup", "olympics", "f1", "formula 1",
    # Sports - types
    "soccer", "football", "basketball", "baseball", "hockey", "tennis",
    "golf", "mma", "ufc", "boxing", "cricket", "rugby", "racing",
    # Sports - terms
    "championship", "playoff", "super bowl", "world series", "stanley cup",
    "drivers champion", "win the 202", "win on 202", "beat the", "defeat",
    "end in a draw", " vs. ", " fc ", " sk ", " afc",
    # Gaming
    "esports", "gaming", "league of legends", "dota", "csgo",
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
