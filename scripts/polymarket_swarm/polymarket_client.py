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

        # Parse stringified outcomes array
        if "outcomes" in data and isinstance(data["outcomes"], str):
            try:
                data["outcomes"] = json.loads(data["outcomes"])
            except json.JSONDecodeError:
                data["outcomes"] = ["Yes", "No"]

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
        min_liquidity: float = MIN_LIQUIDITY_USD,
        exclude_only: bool = True,  # NEW: Only exclude crypto/sports, don't filter by target categories
    ) -> List[Market]:
        """Fetch markets with filtering.

        By default, scans ALL markets except crypto and sports.
        Set exclude_only=False to also prioritize target categories.

        Args:
            limit: Maximum markets to return after filtering
            min_volume: Minimum volume in USD
            min_liquidity: Minimum liquidity in USD
            exclude_only: If True, only exclude crypto/sports (scan ALL else)

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
                # Skip excluded categories (crypto/sports)
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

            # Safety limit - increased to scan more markets
            if offset > 2000:
                break

        return filtered[:limit]

    def fetch_all_categories(
        self,
        limit: int = 200,
        min_volume: float = 5000.0,  # Lower threshold to catch more
        min_liquidity: float = 2000.0,
    ) -> List[Market]:
        """Fetch ALL tradeable markets except crypto/sports.

        This is the broadest scan - catches everything.
        Use this for comprehensive market discovery.

        Args:
            limit: Maximum markets to return
            min_volume: Minimum volume in USD
            min_liquidity: Minimum liquidity in USD

        Returns:
            List of all eligible markets
        """
        return self.fetch_filtered_markets(
            limit=limit,
            min_volume=min_volume,
            min_liquidity=min_liquidity,
            exclude_only=True,
        )

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
