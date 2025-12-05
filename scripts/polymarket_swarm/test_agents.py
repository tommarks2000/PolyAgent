"""Tests for swarm agent definitions."""
import pytest
from agents import (
    AGENT_REGISTRY,
    MarketScannerAgent,
    NewsResearcherAgent,
    ProbabilityAnalystAgent,
    RiskValidatorAgent,
    CoordinatorAgent
)
from models import CognitivePattern, Recommendation

def test_all_agents_defined():
    """Test all 5 agents are defined."""
    assert len(AGENT_REGISTRY) == 5

def test_market_scanner_has_convergent_pattern():
    """Test MarketScanner uses convergent cognitive pattern."""
    agent = MarketScannerAgent()
    assert agent.cognitive_pattern == CognitivePattern.CONVERGENT

def test_news_researcher_has_divergent_pattern():
    """Test NewsResearcher uses divergent cognitive pattern."""
    agent = NewsResearcherAgent()
    assert agent.cognitive_pattern == CognitivePattern.DIVERGENT

def test_probability_analyst_has_critical_pattern():
    """Test ProbabilityAnalyst uses critical cognitive pattern."""
    agent = ProbabilityAnalystAgent()
    assert agent.cognitive_pattern == CognitivePattern.CRITICAL

def test_risk_validator_has_systems_pattern():
    """Test RiskValidator uses systems cognitive pattern."""
    agent = RiskValidatorAgent()
    assert agent.cognitive_pattern == CognitivePattern.SYSTEMS

def test_coordinator_has_adaptive_pattern():
    """Test Coordinator uses adaptive cognitive pattern."""
    agent = CoordinatorAgent()
    assert agent.cognitive_pattern == CognitivePattern.ADAPTIVE

def test_agents_have_analyze_method():
    """Test all agents have analyze method."""
    for name, agent_class in AGENT_REGISTRY.items():
        agent = agent_class()
        assert hasattr(agent, "analyze")
        assert callable(agent.analyze)

def test_agent_weights_are_positive():
    """Test all agent weights are positive."""
    for name, agent_class in AGENT_REGISTRY.items():
        agent = agent_class()
        assert agent.weight > 0
