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
