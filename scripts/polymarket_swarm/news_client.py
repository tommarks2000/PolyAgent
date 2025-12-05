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
                        relevance_score=score / len(query_words) if query_words else 0
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
