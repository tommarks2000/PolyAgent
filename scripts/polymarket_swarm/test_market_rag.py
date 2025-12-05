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
    # Create tradeable mock markets (need active, enableOrderBook, clobTokenIds)
    markets = [
        Market(id=1, question="Will Biden win the 2024 election?", description="US Presidential election",
               active=True, closed=False, archived=False, enableOrderBook=True, clobTokenIds=["t1", "t2"]),
        Market(id=2, question="Will the Fed raise interest rates?", description="Federal Reserve policy",
               active=True, closed=False, archived=False, enableOrderBook=True, clobTokenIds=["t3", "t4"]),
        Market(id=3, question="Will SpaceX launch Starship?", description="Space exploration",
               active=True, closed=False, archived=False, enableOrderBook=True, clobTokenIds=["t5", "t6"]),
    ]
    rag.index_markets(markets)

    results = rag.search("presidential election politics", top_k=2)
    assert len(results) <= 2
    # Election market should be most relevant
    assert any("Biden" in r.question or "election" in r.question for r in results)
