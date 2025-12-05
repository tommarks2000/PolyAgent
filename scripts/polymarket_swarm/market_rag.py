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

        # Use OpenAI embeddings if key available, otherwise default
        if OPENAI_API_KEY:
            self.embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name=EMBEDDING_MODEL
            )
        else:
            # Fall back to default embedding function for testing
            self.embedding_model = embedding_functions.DefaultEmbeddingFunction()

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
            except Exception:
                # Collection may not exist - that's fine
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
