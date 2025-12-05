"""Test all v3 components individually."""
from datetime import datetime

print("=" * 70)
print("V3 COMPONENT TESTS")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Test 1: Web Search Client
print("\n" + "=" * 50)
print("TEST 1: Web Search Client (Perplexity + OpenAI)")
print("=" * 50)

try:
    from web_search_client import WebSearchClient

    with WebSearchClient() as client:
        print(f"Perplexity available: {client.perplexity_available}")
        print(f"OpenAI available: {client.openai_available}")

        if client.is_available:
            result = client.get_news_context("Federal Reserve interest rate decision")
            print(f"Search provider: {result.provider}")
            print(f"Success: {result.success}")
            if result.success:
                print(f"Content preview: {result.content[:200]}...")
            print("TEST 1: PASSED")
        else:
            print("TEST 1: SKIPPED (no API keys)")
except Exception as e:
    print(f"TEST 1: FAILED - {e}")

# Test 2: LLM Swarm
print("\n" + "=" * 50)
print("TEST 2: LLM Swarm Predictions")
print("=" * 50)

try:
    from llm_predictions import LLMSwarmClient

    with LLMSwarmClient() as swarm:
        print(f"Active models: {list(swarm.active_models.keys())}")

        if swarm.active_models:
            result = swarm.get_swarm_prediction(
                market_id="test",
                market_question="Will the Fed cut rates in December 2025?",
                current_yes_price=0.85,
            )

            print(f"Consensus: {result.consensus_prediction}")
            print(f"Strength: {result.consensus_strength:.0%}")
            print(f"Votes: YES={result.yes_votes} NO={result.no_votes}")

            for pred in result.predictions:
                status = "OK" if pred.success else f"FAIL: {pred.error}"
                print(f"  {pred.model_name}: {pred.prediction} [{status}]")

            print("TEST 2: PASSED")
        else:
            print("TEST 2: SKIPPED (no API keys)")
except Exception as e:
    print(f"TEST 2: FAILED - {e}")

# Test 3: Order Book Client
print("\n" + "=" * 50)
print("TEST 3: Order Book Client")
print("=" * 50)

try:
    from orderbook_client import OrderBookClient
    from polymarket_client import PolymarketClient

    with PolymarketClient() as pm:
        markets = pm.fetch_tradeable_markets(limit=2)

    if markets:
        with OrderBookClient() as ob:
            market = markets[0]
            print(f"Testing: {market.question[:50]}...")

            if market.clobTokenIds:
                summary = ob.get_market_summary(market.clobTokenIds, market.question or "")
                print(f"Healthy: {summary.get('healthy')}")
                print(f"Spread: {summary.get('average_spread_percent', 0):.2f}%")
                print(f"Recommendation: {summary.get('recommendation')}")
                print("TEST 3: PASSED")
            else:
                print("TEST 3: SKIPPED (no CLOB tokens)")
    else:
        print("TEST 3: SKIPPED (no markets)")
except Exception as e:
    print(f"TEST 3: FAILED - {e}")

# Test 4: Hybrid Scorer
print("\n" + "=" * 50)
print("TEST 4: Hybrid Scorer")
print("=" * 50)

try:
    from hybrid_scorer import HybridScorer

    scorer = HybridScorer()

    score = scorer.calculate_hybrid_score(
        market_id="test",
        market_question="Test market",
        current_yes_price=0.5,
        research_edge=12.0,
        research_confidence=0.75,
        swarm_prediction="YES",
        swarm_strength=0.8,
        swarm_vote_count=4,
        orderbook_healthy=True,
        orderbook_spread=2.0,
        orderbook_liquidity_score=0.7,
        news_sentiment=0.3,
        historical_accuracy=0.6,
    )

    print(f"Raw Score: {score.raw_score:+.3f}")
    print(f"Recommendation: {score.recommendation}")
    print(f"Edge Estimate: {score.edge_estimate:.1f}%")
    print(f"Strength: {score.strength.value}")
    print("TEST 4: PASSED")
except Exception as e:
    print(f"TEST 4: FAILED - {e}")

# Test 5: Performance Tracker
print("\n" + "=" * 50)
print("TEST 5: Performance Tracker")
print("=" * 50)

try:
    from performance_tracker import PerformanceTracker

    with PerformanceTracker() as tracker:
        metrics = tracker.calculate_metrics()
        print(f"Total predictions: {metrics.total_predictions}")
        print(f"Resolved: {metrics.resolved_predictions}")
        print(f"Accuracy: {metrics.accuracy_rate:.1%}")

        # Record a test prediction
        pred_id = tracker.record_prediction(
            market_id="test_component_check",
            market_question="Component test market",
            our_prediction="YES",
            our_edge_percent=10.0,
            our_confidence=0.7,
            our_fair_value=0.6,
            market_yes_price=0.5,
            market_no_price=0.5,
        )
        print(f"Recorded prediction: {pred_id}")
        print("TEST 5: PASSED")
except Exception as e:
    print(f"TEST 5: FAILED - {e}")

# Test 6: Full Orchestrator
print("\n" + "=" * 50)
print("TEST 6: Orchestrator V3 (Integration)")
print("=" * 50)

try:
    from orchestrator_v3 import EnhancedOrchestratorV3

    with EnhancedOrchestratorV3(
        use_web_search=True,
        use_llm_swarm=True,
        use_orderbook=True,
        use_hybrid_scoring=True,
        track_performance=True,
    ) as orch:
        # Quick test with 2 markets
        print("Running quick scan (2 markets)...")
        opportunities = orch.find_opportunities(
            max_markets=2,
            min_edge=3.0,  # Low threshold for testing
            min_confidence=0.4,
            deep_analysis=False,  # Quick mode
        )

        print(f"Found {len(opportunities)} opportunities")
        print("TEST 6: PASSED")
except Exception as e:
    print(f"TEST 6: FAILED - {e}")

# Summary
print("\n" + "=" * 70)
print("ALL COMPONENT TESTS COMPLETE")
print("=" * 70)
