#!/usr/bin/env python3
"""Optimized Polymarket Scanner - Two-Phase Approach.

Phase 1 (FREE): Scan 200+ markets using only price/volume data
Phase 2 (PAID): Deep LLM analysis only on markets with potential edge

This saves API costs by only using expensive LLM calls on promising markets.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from datetime import datetime
from polymarket_client import PolymarketClient
from orchestrator_v3 import EnhancedOrchestratorV3


def phase1_prefilter(max_markets: int = 200, min_potential: float = 5.0):
    """Phase 1: FREE pre-filter using only market data.

    Args:
        max_markets: How many markets to scan (free)
        min_potential: Minimum potential edge score to pass filter

    Returns:
        List of (market, potential_edge, reasons) tuples
    """
    print()
    print("=" * 70)
    print("PHASE 1: FREE PRE-FILTER")
    print(f"Scanning up to {max_markets} markets using price/volume analysis")
    print("=" * 70)
    print()

    client = PolymarketClient()
    markets = client.fetch_all_categories(
        limit=max_markets,
        min_volume=5000,
        min_liquidity=2000,
    )

    print(f"Fetched {len(markets)} markets from Polymarket")
    print(f"Minimum potential edge to pass: {min_potential}%")
    print()

    promising = []

    for market in markets:
        yes_price = market.yes_price
        volume = market.get_volume_safe()
        liquidity = market.get_liquidity_safe()
        question = market.question[:65] if market.question else "Unknown"

        potential = 0.0
        reasons = []

        # SWEET SPOT: Prices 3-20% or 80-97% have upside potential
        # (Very extreme 0-2% or 98-100% are usually correctly priced)
        if 0.03 <= yes_price <= 0.20:
            potential += 15
            reasons.append(f"low ({yes_price:.0%})")
        elif 0.80 <= yes_price <= 0.97:
            potential += 15
            reasons.append(f"high ({yes_price:.0%})")

        # MID-RANGE with good liquidity = tradeable opportunities
        if 0.25 <= yes_price <= 0.75:
            if liquidity > 20000:
                potential += 12
                reasons.append(f"liquid mid ({yes_price:.0%})")
            elif liquidity > 10000:
                potential += 8
                reasons.append(f"mid-range ({yes_price:.0%})")

        # HIGH VOLUME relative to liquidity = active price discovery
        if liquidity > 0 and volume / liquidity > 15:
            potential += 8
            reasons.append("very active")
        elif liquidity > 0 and volume / liquidity > 8:
            potential += 4
            reasons.append("active")

        # GOOD LIQUIDITY bonus (makes trades executable)
        if liquidity > 50000:
            potential += 5
            reasons.append("high liq")

        if potential >= min_potential:
            promising.append((market, potential, reasons))

    # Sort by potential
    promising.sort(key=lambda x: x[1], reverse=True)

    print(f"{'MARKET':<55} {'PRICE':>8} {'POTENTIAL':>10}")
    print("-" * 75)

    for market, potential, reasons in promising:
        q = market.question[:52] + "..." if len(market.question) > 55 else market.question
        print(f"{q:<55} {market.yes_price:>7.0%} {potential:>8.0f}% ({', '.join(reasons)})")

    print()
    print(f"Phase 1 Complete: {len(promising)}/{len(markets)} markets passed pre-filter")

    return promising


def phase2_deep_analysis(promising_markets, min_edge=8.0, min_confidence=0.6):
    """Phase 2: PAID deep analysis with LLM swarm.

    Args:
        promising_markets: List from Phase 1
        min_edge: Minimum edge for final recommendations
        min_confidence: Minimum confidence score

    Returns:
        List of trading opportunities
    """
    if not promising_markets:
        print("No markets to analyze in Phase 2.")
        return []

    print()
    print("=" * 70)
    print(f"PHASE 2: PAID DEEP ANALYSIS ({len(promising_markets)} markets)")
    print("Using: 6 LLMs + Web Search + Order Book Analysis")
    print("=" * 70)
    print()

    with EnhancedOrchestratorV3(
        use_web_search=True,
        use_llm_swarm=True,
        use_orderbook=True,
        use_hybrid_scoring=True,
        track_performance=True,
    ) as orch:
        opportunities = []

        for i, (market, potential, reasons) in enumerate(promising_markets):
            question = market.question[:50] if market.question else "Unknown"
            print(f"[{i+1}/{len(promising_markets)}] {question}...")

            try:
                opp = orch.analyze_market(market, deep_analysis=True)

                if opp and opp.edge_percent >= min_edge and opp.confidence >= min_confidence:
                    opportunities.append(opp)
                    print(f"  -> FOUND: {opp.recommendation.value} ({opp.edge_percent:.1f}% edge)")
                elif opp:
                    print(f"  -> Below threshold (edge: {opp.edge_percent:.1f}%)")
                else:
                    print(f"  -> SKIP")

            except Exception as e:
                print(f"  -> Error: {e}")

        opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        if opportunities:
            print()
            print(orch.generate_report(opportunities))

        return opportunities


def main():
    """Run optimized two-phase scanner."""
    print()
    print("=" * 70)
    print("POLYMARKET SWARM TRADER v3.0 - OPTIMIZED SCANNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # PHASE 1: Free pre-filter (scan 2000 markets - this is FREE!)
    promising = phase1_prefilter(
        max_markets=2000,     # Scan 2000 markets (FREE API calls)
        min_potential=15.0,   # Only pass markets with 15%+ potential score
    )

    if not promising:
        print("\nNo promising markets found. Exiting.")
        return

    # Ask user before spending on LLM calls
    print()
    print(f"Found {len(promising)} promising markets.")
    print(f"Phase 2 will use ~{len(promising) * 6} LLM API calls.")

    proceed = input("\nProceed with Phase 2 deep analysis? [Y/n]: ").strip().lower()
    if proceed == 'n':
        print("Skipping Phase 2. Exiting.")
        return

    # PHASE 2: Paid deep analysis
    opportunities = phase2_deep_analysis(
        promising,
        min_edge=8.0,         # Final edge threshold
        min_confidence=0.6,   # Final confidence threshold
    )

    print()
    print("=" * 70)
    print(f"SCAN COMPLETE: {len(opportunities)} opportunities found")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
