"""Multi-Model LLM Predictions for Polymarket.

Queries multiple LLM providers in parallel to get consensus predictions.
Inspired by Moon Dev's swarm approach but integrated with our edge-based system.
"""
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

from config import OPENAI_API_KEY, PERPLEXITY_API_KEY


@dataclass
class LLMPrediction:
    """Single model's prediction."""
    model_name: str
    provider: str
    prediction: str  # YES, NO, or SKIP
    confidence: float  # 0-1
    reasoning: str
    response_time: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class SwarmPrediction:
    """Aggregated prediction from all models."""
    market_id: str
    market_question: str
    timestamp: str

    # Individual predictions
    predictions: List[LLMPrediction] = field(default_factory=list)

    # Aggregated results
    yes_votes: int = 0
    no_votes: int = 0
    skip_votes: int = 0
    total_responses: int = 0

    # Consensus
    consensus_prediction: str = "SKIP"
    consensus_strength: float = 0.0  # 0-1
    average_confidence: float = 0.0

    def calculate_consensus(self):
        """Calculate consensus from individual predictions."""
        self.yes_votes = sum(1 for p in self.predictions if p.prediction == "YES" and p.success)
        self.no_votes = sum(1 for p in self.predictions if p.prediction == "NO" and p.success)
        self.skip_votes = sum(1 for p in self.predictions if p.prediction == "SKIP" and p.success)
        self.total_responses = sum(1 for p in self.predictions if p.success)

        if self.total_responses == 0:
            self.consensus_prediction = "SKIP"
            self.consensus_strength = 0.0
            return

        # Determine consensus
        max_votes = max(self.yes_votes, self.no_votes, self.skip_votes)

        if self.yes_votes == max_votes and self.yes_votes > self.no_votes:
            self.consensus_prediction = "YES"
            self.consensus_strength = self.yes_votes / self.total_responses
        elif self.no_votes == max_votes and self.no_votes > self.yes_votes:
            self.consensus_prediction = "NO"
            self.consensus_strength = self.no_votes / self.total_responses
        else:
            self.consensus_prediction = "SKIP"
            self.consensus_strength = self.skip_votes / self.total_responses

        # Average confidence
        confidences = [p.confidence for p in self.predictions if p.success and p.confidence > 0]
        self.average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "market_question": self.market_question,
            "timestamp": self.timestamp,
            "predictions": [
                {
                    "model": p.model_name,
                    "provider": p.provider,
                    "prediction": p.prediction,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning[:200] if p.reasoning else "",
                    "response_time": p.response_time,
                    "success": p.success,
                }
                for p in self.predictions
            ],
            "yes_votes": self.yes_votes,
            "no_votes": self.no_votes,
            "skip_votes": self.skip_votes,
            "total_responses": self.total_responses,
            "consensus_prediction": self.consensus_prediction,
            "consensus_strength": self.consensus_strength,
            "average_confidence": self.average_confidence,
        }


# Model configurations: (enabled, provider, model_id, api_key_env)
LLM_MODELS = {
    "gpt4o": (True, "openai", "gpt-4o", "OPENAI_API_KEY"),
    "gpt4o_mini": (True, "openai", "gpt-4o-mini", "OPENAI_API_KEY"),
    "claude_sonnet": (True, "anthropic", "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "deepseek": (True, "deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "perplexity": (True, "perplexity", "sonar", "PERPLEXITY_API_KEY"),
}


class LLMSwarmClient:
    """Client for querying multiple LLMs in parallel."""

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

        # Load API keys
        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", ""),
            "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY", PERPLEXITY_API_KEY),
        }

        # Filter to only enabled models with valid API keys
        self.active_models = {}
        for name, (enabled, provider, model_id, key_env) in LLM_MODELS.items():
            if enabled and self.api_keys.get(key_env):
                self.active_models[name] = (provider, model_id, self.api_keys[key_env])

        print(f"LLM Swarm initialized with {len(self.active_models)} models: {list(self.active_models.keys())}")

    def _build_prediction_prompt(
        self,
        market_question: str,
        current_yes_price: float,
        market_description: Optional[str] = None,
        news_context: Optional[str] = None,
        key_factors: Optional[List[str]] = None,
    ) -> str:
        """Build the prediction prompt for LLMs."""

        prompt = f"""You are a prediction market analyst. Analyze this market and provide your prediction.

MARKET: {market_question}

CURRENT MARKET PRICE: YES = {current_yes_price:.1%} / NO = {1-current_yes_price:.1%}
"""

        if market_description:
            prompt += f"\nDESCRIPTION: {market_description[:500]}\n"

        if news_context:
            prompt += f"\nRECENT NEWS/CONTEXT:\n{news_context[:1000]}\n"

        if key_factors:
            prompt += "\nKEY FACTORS:\n"
            for factor in key_factors[:5]:
                prompt += f"- {factor}\n"

        prompt += """
INSTRUCTIONS:
1. Analyze the market question and available context
2. Consider if the current market price is accurate or mispriced
3. Provide your prediction

RESPOND IN THIS EXACT FORMAT:
PREDICTION: [YES/NO/SKIP]
CONFIDENCE: [0.0-1.0]
REASONING: [One sentence explaining your prediction]

Only respond with YES if you believe the probability should be HIGHER than the current price.
Only respond with NO if you believe the probability should be LOWER than the current price.
Respond with SKIP if you're uncertain or the price seems fair.
"""
        return prompt

    def _parse_prediction(self, response: str) -> Tuple[str, float, str]:
        """Parse model response into prediction, confidence, reasoning."""
        prediction = "SKIP"
        confidence = 0.5
        reasoning = ""

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("PREDICTION:"):
                pred = line.split(":", 1)[1].strip().upper()
                if pred in ["YES", "NO", "SKIP"]:
                    prediction = pred
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    conf = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, conf))
                except ValueError:
                    pass
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return prediction, confidence, reasoning

    def _query_openai(self, model_id: str, api_key: str, prompt: str) -> str:
        """Query OpenAI API."""
        response = self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _query_anthropic(self, model_id: str, api_key: str, prompt: str) -> str:
        """Query Anthropic API."""
        response = self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def _query_deepseek(self, model_id: str, api_key: str, prompt: str) -> str:
        """Query DeepSeek API (OpenAI compatible)."""
        response = self.client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _query_perplexity(self, model_id: str, api_key: str, prompt: str) -> str:
        """Query Perplexity API."""
        response = self.client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _query_single_model(
        self,
        model_name: str,
        provider: str,
        model_id: str,
        api_key: str,
        prompt: str,
    ) -> LLMPrediction:
        """Query a single model and return prediction."""
        start_time = time.time()

        try:
            # Route to appropriate provider
            if provider == "openai":
                response = self._query_openai(model_id, api_key, prompt)
            elif provider == "anthropic":
                response = self._query_anthropic(model_id, api_key, prompt)
            elif provider == "deepseek":
                response = self._query_deepseek(model_id, api_key, prompt)
            elif provider == "perplexity":
                response = self._query_perplexity(model_id, api_key, prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")

            # Parse response
            prediction, confidence, reasoning = self._parse_prediction(response)

            return LLMPrediction(
                model_name=model_name,
                provider=provider,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                response_time=time.time() - start_time,
                success=True,
            )

        except Exception as e:
            return LLMPrediction(
                model_name=model_name,
                provider=provider,
                prediction="SKIP",
                confidence=0.0,
                reasoning="",
                response_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def get_swarm_prediction(
        self,
        market_id: str,
        market_question: str,
        current_yes_price: float,
        market_description: Optional[str] = None,
        news_context: Optional[str] = None,
        key_factors: Optional[List[str]] = None,
    ) -> SwarmPrediction:
        """Get predictions from all active models in parallel.

        Args:
            market_id: Market identifier
            market_question: The prediction market question
            current_yes_price: Current YES price (0-1)
            market_description: Optional market description
            news_context: Optional news/research context
            key_factors: Optional list of key factors

        Returns:
            SwarmPrediction with consensus
        """
        # Build prompt
        prompt = self._build_prediction_prompt(
            market_question=market_question,
            current_yes_price=current_yes_price,
            market_description=market_description,
            news_context=news_context,
            key_factors=key_factors,
        )

        # Create swarm prediction
        swarm = SwarmPrediction(
            market_id=market_id,
            market_question=market_question,
            timestamp=datetime.now().isoformat(),
        )

        # Query all models in parallel
        with ThreadPoolExecutor(max_workers=len(self.active_models)) as executor:
            futures = {
                executor.submit(
                    self._query_single_model,
                    name, provider, model_id, api_key, prompt
                ): name
                for name, (provider, model_id, api_key) in self.active_models.items()
            }

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    prediction = future.result()
                    swarm.predictions.append(prediction)
                except Exception as e:
                    swarm.predictions.append(LLMPrediction(
                        model_name=model_name,
                        provider="unknown",
                        prediction="SKIP",
                        confidence=0.0,
                        reasoning="",
                        response_time=0.0,
                        success=False,
                        error=str(e),
                    ))

        # Calculate consensus
        swarm.calculate_consensus()

        return swarm

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("LLM SWARM PREDICTION TEST")
    print("=" * 70)

    with LLMSwarmClient() as swarm:
        if not swarm.active_models:
            print("No models available - check API keys")
            exit(1)

        # Test with a sample market
        result = swarm.get_swarm_prediction(
            market_id="test_123",
            market_question="Will the Federal Reserve cut interest rates in December 2025?",
            current_yes_price=0.85,
            news_context="Recent economic data shows cooling inflation. Fed officials have signaled openness to rate cuts.",
            key_factors=[
                "Inflation trending down to 2.5%",
                "Labor market cooling",
                "Fed officials dovish commentary",
            ],
        )

        print(f"\nMarket: {result.market_question}")
        print(f"\nIndividual Predictions:")
        for pred in result.predictions:
            status = "OK" if pred.success else f"FAIL: {pred.error}"
            print(f"  {pred.model_name}: {pred.prediction} ({pred.confidence:.0%}) [{status}]")
            if pred.reasoning:
                print(f"    -> {pred.reasoning[:80]}...")

        print(f"\nConsensus:")
        print(f"  Prediction: {result.consensus_prediction}")
        print(f"  Strength: {result.consensus_strength:.1%}")
        print(f"  Votes: YES={result.yes_votes} NO={result.no_votes} SKIP={result.skip_votes}")

        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
