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
