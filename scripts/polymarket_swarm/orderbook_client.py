"""Order book client for Polymarket CLOB API.

Fetches real-time order book data for liquidity and slippage analysis.
Based on Polymarket CLOB documentation.
"""
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


# CLOB API endpoints
CLOB_BASE_URL = "https://clob.polymarket.com"
BOOK_ENDPOINT = f"{CLOB_BASE_URL}/book"
MIDPOINT_ENDPOINT = f"{CLOB_BASE_URL}/midpoint"
PRICE_ENDPOINT = f"{CLOB_BASE_URL}/price"


@dataclass
class OrderLevel:
    """Single price level in the order book."""
    price: float
    size: float  # Total size at this level

    @property
    def value(self) -> float:
        """Dollar value at this level."""
        return self.price * self.size


@dataclass
class OrderBookSide:
    """One side (bids or asks) of the order book."""
    levels: List[OrderLevel] = field(default_factory=list)

    @property
    def total_size(self) -> float:
        """Total size across all levels."""
        return sum(level.size for level in self.levels)

    @property
    def total_value(self) -> float:
        """Total dollar value across all levels."""
        return sum(level.value for level in self.levels)

    @property
    def best_price(self) -> Optional[float]:
        """Best price (first level)."""
        if self.levels:
            return self.levels[0].price
        return None

    @property
    def depth_5_levels(self) -> float:
        """Size available in top 5 levels."""
        return sum(level.size for level in self.levels[:5])

    def size_at_price(self, target_price: float, is_bid: bool = True) -> float:
        """Calculate total size available at or better than target price.

        For bids: sum all levels >= target_price
        For asks: sum all levels <= target_price
        """
        total = 0.0
        for level in self.levels:
            if is_bid:
                if level.price >= target_price:
                    total += level.size
            else:
                if level.price <= target_price:
                    total += level.size
        return total


@dataclass
class OrderBook:
    """Complete order book for a market."""
    token_id: str
    market_id: Optional[str] = None
    timestamp: str = ""

    # Order book sides
    bids: OrderBookSide = field(default_factory=OrderBookSide)
    asks: OrderBookSide = field(default_factory=OrderBookSide)

    # Computed metrics
    midpoint: Optional[float] = None
    spread: Optional[float] = None
    spread_percent: Optional[float] = None

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids.best_price

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks.best_price

    @property
    def is_liquid(self) -> bool:
        """Check if order book has reasonable liquidity."""
        return (
            self.bids.total_size > 100 and
            self.asks.total_size > 100 and
            self.spread_percent is not None and
            self.spread_percent < 5.0  # Less than 5% spread
        )

    def compute_metrics(self):
        """Compute derived metrics from order book."""
        if self.best_bid and self.best_ask:
            self.midpoint = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid
            if self.midpoint > 0:
                self.spread_percent = (self.spread / self.midpoint) * 100

    def estimate_slippage(self, size: float, side: str = "buy") -> Dict[str, float]:
        """Estimate slippage for a given order size.

        Args:
            size: Order size in shares
            side: "buy" or "sell"

        Returns:
            Dict with average_price, slippage_percent, filled_size
        """
        levels = self.asks.levels if side == "buy" else self.bids.levels

        if not levels:
            return {
                "average_price": 0.0,
                "slippage_percent": 100.0,
                "filled_size": 0.0,
                "unfilled_size": size,
            }

        remaining = size
        total_cost = 0.0
        filled = 0.0

        for level in levels:
            if remaining <= 0:
                break

            fill_size = min(remaining, level.size)
            total_cost += fill_size * level.price
            filled += fill_size
            remaining -= fill_size

        if filled == 0:
            return {
                "average_price": 0.0,
                "slippage_percent": 100.0,
                "filled_size": 0.0,
                "unfilled_size": size,
            }

        average_price = total_cost / filled

        # Calculate slippage from best price
        best_price = levels[0].price if levels else 0
        if best_price > 0:
            slippage = abs(average_price - best_price) / best_price * 100
        else:
            slippage = 0.0

        return {
            "average_price": average_price,
            "best_price": best_price,
            "slippage_percent": slippage,
            "filled_size": filled,
            "unfilled_size": remaining,
        }


@dataclass
class OrderBookAnalysis:
    """Analysis result for order book."""
    token_id: str
    market_question: str = ""

    # Book metrics
    best_bid: float = 0.0
    best_ask: float = 0.0
    midpoint: float = 0.0
    spread: float = 0.0
    spread_percent: float = 0.0

    # Liquidity metrics
    bid_depth_5: float = 0.0  # Size in top 5 bid levels
    ask_depth_5: float = 0.0  # Size in top 5 ask levels
    total_bid_size: float = 0.0
    total_ask_size: float = 0.0

    # Slippage estimates for common sizes
    slippage_100: float = 0.0  # Slippage for 100 share order
    slippage_500: float = 0.0  # Slippage for 500 share order
    slippage_1000: float = 0.0 # Slippage for 1000 share order

    # Quality scores
    liquidity_score: float = 0.0  # 0-1 score
    tradability_score: float = 0.0  # 0-1 overall score

    # Warnings
    warnings: List[str] = field(default_factory=list)

    @property
    def is_tradeable(self) -> bool:
        """Check if market is reasonably tradeable."""
        return (
            self.spread_percent < 10.0 and
            self.liquidity_score > 0.3 and
            len(self.warnings) < 3
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "market_question": self.market_question,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "spread_percent": self.spread_percent,
            "bid_depth_5": self.bid_depth_5,
            "ask_depth_5": self.ask_depth_5,
            "total_bid_size": self.total_bid_size,
            "total_ask_size": self.total_ask_size,
            "slippage_100": self.slippage_100,
            "slippage_500": self.slippage_500,
            "slippage_1000": self.slippage_1000,
            "liquidity_score": self.liquidity_score,
            "tradability_score": self.tradability_score,
            "is_tradeable": self.is_tradeable,
            "warnings": self.warnings,
        }


class OrderBookClient:
    """Client for fetching and analyzing Polymarket order books."""

    def __init__(self, timeout: float = 15.0):
        """Initialize order book client.

        Args:
            timeout: Request timeout in seconds
        """
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "PolymarketSwarmTrader/2.0"
            }
        )

    def fetch_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book for a specific token.

        Args:
            token_id: CLOB token ID (from market.clobTokenIds)

        Returns:
            OrderBook object or None if failed
        """
        try:
            response = self.client.get(
                BOOK_ENDPOINT,
                params={"token_id": token_id}
            )
            response.raise_for_status()
            data = response.json()

            return self._parse_order_book(token_id, data)

        except httpx.HTTPError as e:
            print(f"Error fetching order book for {token_id}: {e}")
            return None
        except Exception as e:
            print(f"Error parsing order book: {e}")
            return None

    def _parse_order_book(self, token_id: str, data: Dict[str, Any]) -> OrderBook:
        """Parse CLOB API response into OrderBook object."""
        book = OrderBook(
            token_id=token_id,
            market_id=data.get("market"),
            timestamp=datetime.now().isoformat(),
        )

        # Parse bids (sorted high to low)
        bids_data = data.get("bids", [])
        bid_levels = []
        for bid in bids_data:
            try:
                price = float(bid.get("price", 0))
                size = float(bid.get("size", 0))
                if price > 0 and size > 0:
                    bid_levels.append(OrderLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue

        # Sort bids high to low
        bid_levels.sort(key=lambda x: x.price, reverse=True)
        book.bids = OrderBookSide(levels=bid_levels)

        # Parse asks (sorted low to high)
        asks_data = data.get("asks", [])
        ask_levels = []
        for ask in asks_data:
            try:
                price = float(ask.get("price", 0))
                size = float(ask.get("size", 0))
                if price > 0 and size > 0:
                    ask_levels.append(OrderLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue

        # Sort asks low to high
        ask_levels.sort(key=lambda x: x.price)
        book.asks = OrderBookSide(levels=ask_levels)

        # Compute metrics
        book.compute_metrics()

        return book

    def fetch_midpoint(self, token_id: str) -> Optional[float]:
        """Fetch just the midpoint price for a token.

        Args:
            token_id: CLOB token ID

        Returns:
            Midpoint price or None
        """
        try:
            response = self.client.get(
                MIDPOINT_ENDPOINT,
                params={"token_id": token_id}
            )
            response.raise_for_status()
            data = response.json()
            return float(data.get("mid", 0))
        except Exception:
            return None

    def analyze_order_book(
        self,
        token_id: str,
        market_question: str = "",
    ) -> OrderBookAnalysis:
        """Fetch and analyze order book for a token.

        Args:
            token_id: CLOB token ID
            market_question: Market question for context

        Returns:
            OrderBookAnalysis with all metrics
        """
        analysis = OrderBookAnalysis(
            token_id=token_id,
            market_question=market_question,
        )

        # Fetch order book
        book = self.fetch_order_book(token_id)

        if not book:
            analysis.warnings.append("Failed to fetch order book")
            return analysis

        # Basic metrics
        analysis.best_bid = book.best_bid or 0.0
        analysis.best_ask = book.best_ask or 0.0
        analysis.midpoint = book.midpoint or 0.0
        analysis.spread = book.spread or 0.0
        analysis.spread_percent = book.spread_percent or 0.0

        # Depth metrics
        analysis.bid_depth_5 = book.bids.depth_5_levels
        analysis.ask_depth_5 = book.asks.depth_5_levels
        analysis.total_bid_size = book.bids.total_size
        analysis.total_ask_size = book.asks.total_size

        # Slippage estimates
        slippage_100 = book.estimate_slippage(100, "buy")
        slippage_500 = book.estimate_slippage(500, "buy")
        slippage_1000 = book.estimate_slippage(1000, "buy")

        analysis.slippage_100 = slippage_100["slippage_percent"]
        analysis.slippage_500 = slippage_500["slippage_percent"]
        analysis.slippage_1000 = slippage_1000["slippage_percent"]

        # Calculate liquidity score (0-1)
        # Based on depth and spread
        depth_score = min(1.0, (analysis.bid_depth_5 + analysis.ask_depth_5) / 2000)
        spread_score = max(0.0, 1.0 - (analysis.spread_percent / 10))
        analysis.liquidity_score = (depth_score + spread_score) / 2

        # Calculate tradability score
        slippage_score = max(0.0, 1.0 - (analysis.slippage_500 / 5))
        analysis.tradability_score = (
            analysis.liquidity_score * 0.5 +
            spread_score * 0.3 +
            slippage_score * 0.2
        )

        # Generate warnings
        if analysis.spread_percent > 5:
            analysis.warnings.append(f"Wide spread: {analysis.spread_percent:.1f}%")

        if analysis.bid_depth_5 < 100:
            analysis.warnings.append(f"Low bid depth: {analysis.bid_depth_5:.0f} shares")

        if analysis.ask_depth_5 < 100:
            analysis.warnings.append(f"Low ask depth: {analysis.ask_depth_5:.0f} shares")

        if analysis.slippage_500 > 2:
            analysis.warnings.append(f"High slippage (500): {analysis.slippage_500:.1f}%")

        if analysis.total_bid_size < 500 or analysis.total_ask_size < 500:
            analysis.warnings.append("Thin order book - use limit orders")

        return analysis

    def analyze_market_order_books(
        self,
        clob_token_ids: List[str],
        market_question: str = "",
    ) -> Dict[str, OrderBookAnalysis]:
        """Analyze order books for all tokens in a market.

        Polymarket markets typically have 2 tokens: YES and NO.

        Args:
            clob_token_ids: List of token IDs (usually [YES_token, NO_token])
            market_question: Market question for context

        Returns:
            Dict mapping "YES"/"NO" to their analysis
        """
        results = {}

        if not clob_token_ids:
            return results

        # First token is YES, second is NO
        if len(clob_token_ids) >= 1:
            yes_analysis = self.analyze_order_book(
                clob_token_ids[0],
                market_question
            )
            results["YES"] = yes_analysis

        if len(clob_token_ids) >= 2:
            no_analysis = self.analyze_order_book(
                clob_token_ids[1],
                market_question
            )
            results["NO"] = no_analysis

        return results

    def get_market_summary(
        self,
        clob_token_ids: List[str],
        market_question: str = "",
    ) -> Dict[str, Any]:
        """Get a summary of order book health for a market.

        Args:
            clob_token_ids: List of token IDs
            market_question: Market question

        Returns:
            Summary dict with key metrics
        """
        analyses = self.analyze_market_order_books(clob_token_ids, market_question)

        if not analyses:
            return {
                "healthy": False,
                "reason": "Could not fetch order books",
                "yes_analysis": None,
                "no_analysis": None,
            }

        yes_analysis = analyses.get("YES")
        no_analysis = analyses.get("NO")

        # Determine overall health
        healthy = True
        issues = []

        if yes_analysis:
            if not yes_analysis.is_tradeable:
                healthy = False
                issues.append(f"YES side issues: {', '.join(yes_analysis.warnings)}")

        if no_analysis:
            if not no_analysis.is_tradeable:
                healthy = False
                issues.append(f"NO side issues: {', '.join(no_analysis.warnings)}")

        # Calculate combined metrics
        avg_spread = 0.0
        avg_liquidity = 0.0

        if yes_analysis and no_analysis:
            avg_spread = (yes_analysis.spread_percent + no_analysis.spread_percent) / 2
            avg_liquidity = (yes_analysis.liquidity_score + no_analysis.liquidity_score) / 2
        elif yes_analysis:
            avg_spread = yes_analysis.spread_percent
            avg_liquidity = yes_analysis.liquidity_score
        elif no_analysis:
            avg_spread = no_analysis.spread_percent
            avg_liquidity = no_analysis.liquidity_score

        return {
            "healthy": healthy,
            "issues": issues,
            "average_spread_percent": avg_spread,
            "average_liquidity_score": avg_liquidity,
            "yes_analysis": yes_analysis.to_dict() if yes_analysis else None,
            "no_analysis": no_analysis.to_dict() if no_analysis else None,
            "recommendation": self._get_trading_recommendation(yes_analysis, no_analysis),
        }

    def _get_trading_recommendation(
        self,
        yes_analysis: Optional[OrderBookAnalysis],
        no_analysis: Optional[OrderBookAnalysis],
    ) -> str:
        """Generate trading recommendation based on order book health."""

        if not yes_analysis and not no_analysis:
            return "AVOID - No order book data"

        # Check for serious issues
        all_warnings = []
        if yes_analysis:
            all_warnings.extend(yes_analysis.warnings)
        if no_analysis:
            all_warnings.extend(no_analysis.warnings)

        if len(all_warnings) >= 4:
            return "AVOID - Multiple liquidity issues"

        # Check spread
        spreads = []
        if yes_analysis:
            spreads.append(yes_analysis.spread_percent)
        if no_analysis:
            spreads.append(no_analysis.spread_percent)

        avg_spread = sum(spreads) / len(spreads) if spreads else 100

        if avg_spread > 10:
            return "AVOID - Spread too wide"
        elif avg_spread > 5:
            return "CAUTION - Wide spread, use limit orders"
        elif avg_spread > 2:
            return "OK - Moderate spread, consider limit orders"
        else:
            return "GOOD - Tight spread, market orders acceptable"

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    from polymarket_client import PolymarketClient

    print("=" * 70)
    print("ORDER BOOK CLIENT TEST")
    print("=" * 70)

    # Get some markets
    with PolymarketClient() as pm:
        markets = pm.fetch_tradeable_markets(limit=5)

    if not markets:
        print("No markets found")
        exit(1)

    # Test order book client
    with OrderBookClient() as ob_client:
        for market in markets[:3]:
            print(f"\n{'='*60}")
            print(f"Market: {market.question[:60]}...")
            print(f"Token IDs: {market.clobTokenIds}")
            print(f"{'='*60}")

            if not market.clobTokenIds:
                print("No CLOB tokens - skipping")
                continue

            summary = ob_client.get_market_summary(
                market.clobTokenIds,
                market.question or ""
            )

            print(f"\nOrder Book Summary:")
            print(f"  Healthy: {summary['healthy']}")
            print(f"  Avg Spread: {summary['average_spread_percent']:.2f}%")
            print(f"  Liquidity Score: {summary['average_liquidity_score']:.2f}")
            print(f"  Recommendation: {summary['recommendation']}")

            if summary['issues']:
                print(f"  Issues: {summary['issues']}")

            if summary['yes_analysis']:
                yes = summary['yes_analysis']
                print(f"\n  YES Order Book:")
                print(f"    Best Bid: {yes['best_bid']:.3f}")
                print(f"    Best Ask: {yes['best_ask']:.3f}")
                print(f"    Spread: {yes['spread_percent']:.2f}%")
                print(f"    Depth (5 lvls): {yes['bid_depth_5']:.0f} bid / {yes['ask_depth_5']:.0f} ask")
                print(f"    Slippage (500): {yes['slippage_500']:.2f}%")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
