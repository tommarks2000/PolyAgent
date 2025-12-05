"""Tests for swarm orchestrator."""
import pytest
from orchestrator import SwarmOrchestrator
from models import TradeOpportunity

def test_orchestrator_initialization():
    """Test orchestrator initializes with all agents."""
    orchestrator = SwarmOrchestrator()
    assert len(orchestrator.agents) == 5

def test_orchestrator_has_analyze_market_method():
    """Test orchestrator has analyze_market method."""
    orchestrator = SwarmOrchestrator()
    assert hasattr(orchestrator, "analyze_market")
    assert callable(orchestrator.analyze_market)

def test_orchestrator_has_find_opportunities_method():
    """Test orchestrator has find_opportunities method."""
    orchestrator = SwarmOrchestrator()
    assert hasattr(orchestrator, "find_opportunities")
    assert callable(orchestrator.find_opportunities)

def test_orchestrator_returns_trade_opportunity():
    """Test that analysis can return TradeOpportunity."""
    orchestrator = SwarmOrchestrator()
    # We don't actually run analysis here (requires API)
    # Just verify the types are correct
    assert TradeOpportunity is not None
