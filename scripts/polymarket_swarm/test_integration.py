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
