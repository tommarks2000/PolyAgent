"""Historical Performance Tracking for Polymarket Predictions.

Tracks predictions, evaluates resolved markets, and calculates accuracy metrics.
Provides feedback loop for continuous improvement.
"""
import os
import json
import csv
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import httpx

from config import GAMMA_MARKETS_ENDPOINT


# Data directory for persistence
DATA_DIR = Path(__file__).parent / "data" / "performance"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = DATA_DIR / "predictions.json"
RESULTS_FILE = DATA_DIR / "results.json"
METRICS_FILE = DATA_DIR / "metrics.json"


@dataclass
class PredictionRecord:
    """Record of a single prediction."""
    prediction_id: str
    market_id: str
    market_question: str
    timestamp: str

    # Our prediction
    our_prediction: str  # YES or NO
    our_edge_percent: float
    our_confidence: float
    our_fair_value: float  # Our estimated probability

    # Market state at prediction time
    market_yes_price: float
    market_no_price: float
    market_volume: float
    market_liquidity: float

    # Additional context
    swarm_consensus: Optional[str] = None
    swarm_strength: Optional[float] = None
    orderbook_healthy: bool = True
    research_sources: List[str] = field(default_factory=list)

    # Resolution (filled in later)
    resolved: bool = False
    resolution_timestamp: Optional[str] = None
    actual_outcome: Optional[str] = None  # YES or NO
    prediction_correct: Optional[bool] = None
    profit_loss_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionRecord":
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_predictions: int = 0
    resolved_predictions: int = 0
    pending_predictions: int = 0

    # Accuracy
    correct_predictions: int = 0
    incorrect_predictions: int = 0
    accuracy_rate: float = 0.0

    # By prediction type
    yes_predictions: int = 0
    yes_correct: int = 0
    yes_accuracy: float = 0.0

    no_predictions: int = 0
    no_correct: int = 0
    no_accuracy: float = 0.0

    # Edge analysis
    avg_edge_claimed: float = 0.0
    avg_edge_captured: float = 0.0
    edge_capture_rate: float = 0.0

    # Confidence calibration
    high_confidence_accuracy: float = 0.0  # >80% confidence
    medium_confidence_accuracy: float = 0.0  # 60-80%
    low_confidence_accuracy: float = 0.0  # <60%

    # Swarm consensus analysis
    consensus_accuracy: float = 0.0
    split_vote_accuracy: float = 0.0

    # Time analysis
    last_updated: str = ""
    oldest_prediction: str = ""
    newest_prediction: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """Tracks and evaluates prediction performance."""

    def __init__(self):
        self.predictions: Dict[str, PredictionRecord] = {}
        self.metrics = PerformanceMetrics()
        self.client = httpx.Client(timeout=30.0)

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing predictions from disk."""
        if PREDICTIONS_FILE.exists():
            try:
                with open(PREDICTIONS_FILE, "r") as f:
                    data = json.load(f)
                    for pred_data in data:
                        pred = PredictionRecord.from_dict(pred_data)
                        self.predictions[pred.prediction_id] = pred
                print(f"Loaded {len(self.predictions)} historical predictions")
            except Exception as e:
                print(f"Error loading predictions: {e}")

        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r") as f:
                    data = json.load(f)
                    self.metrics = PerformanceMetrics(**data)
            except Exception as e:
                print(f"Error loading metrics: {e}")

    def _save_data(self):
        """Save predictions and metrics to disk."""
        # Save predictions
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(
                [p.to_dict() for p in self.predictions.values()],
                f,
                indent=2,
                default=str
            )

        # Save metrics
        with open(METRICS_FILE, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

    def record_prediction(
        self,
        market_id: str,
        market_question: str,
        our_prediction: str,
        our_edge_percent: float,
        our_confidence: float,
        our_fair_value: float,
        market_yes_price: float,
        market_no_price: float,
        market_volume: float = 0.0,
        market_liquidity: float = 0.0,
        swarm_consensus: Optional[str] = None,
        swarm_strength: Optional[float] = None,
        orderbook_healthy: bool = True,
        research_sources: Optional[List[str]] = None,
    ) -> str:
        """Record a new prediction.

        Args:
            market_id: Polymarket market ID
            market_question: The market question
            our_prediction: Our prediction (YES or NO)
            our_edge_percent: Our calculated edge
            our_confidence: Our confidence score
            our_fair_value: Our estimated probability
            market_yes_price: Current YES price
            market_no_price: Current NO price
            market_volume: Market volume
            market_liquidity: Market liquidity
            swarm_consensus: LLM swarm consensus (optional)
            swarm_strength: Swarm consensus strength (optional)
            orderbook_healthy: Order book health status
            research_sources: List of research sources used

        Returns:
            Prediction ID
        """
        prediction_id = f"{market_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        record = PredictionRecord(
            prediction_id=prediction_id,
            market_id=str(market_id),
            market_question=market_question,
            timestamp=datetime.now().isoformat(),
            our_prediction=our_prediction,
            our_edge_percent=our_edge_percent,
            our_confidence=our_confidence,
            our_fair_value=our_fair_value,
            market_yes_price=market_yes_price,
            market_no_price=market_no_price,
            market_volume=market_volume,
            market_liquidity=market_liquidity,
            swarm_consensus=swarm_consensus,
            swarm_strength=swarm_strength,
            orderbook_healthy=orderbook_healthy,
            research_sources=research_sources or [],
        )

        self.predictions[prediction_id] = record
        self._save_data()

        return prediction_id

    def check_market_resolution(self, market_id: str) -> Optional[str]:
        """Check if a market has resolved.

        Args:
            market_id: Polymarket market ID

        Returns:
            Resolution outcome (YES/NO) or None if not resolved
        """
        try:
            url = f"{GAMMA_MARKETS_ENDPOINT}/{market_id}"
            response = self.client.get(url)
            response.raise_for_status()
            data = response.json()

            # Check if closed/resolved
            if data.get("closed") or data.get("archived"):
                # Determine outcome from final prices
                outcome_prices = data.get("outcomePrices", [])
                if outcome_prices and len(outcome_prices) >= 2:
                    # Parse if string
                    if isinstance(outcome_prices, str):
                        outcome_prices = json.loads(outcome_prices)
                    outcome_prices = [float(p) for p in outcome_prices]

                    # If YES price is near 1, YES won; if near 0, NO won
                    if outcome_prices[0] > 0.95:
                        return "YES"
                    elif outcome_prices[0] < 0.05:
                        return "NO"

            return None

        except Exception as e:
            print(f"Error checking market {market_id}: {e}")
            return None

    def update_resolved_predictions(self) -> int:
        """Check all pending predictions for resolution.

        Returns:
            Number of newly resolved predictions
        """
        resolved_count = 0

        for pred_id, pred in self.predictions.items():
            if pred.resolved:
                continue

            outcome = self.check_market_resolution(pred.market_id)

            if outcome:
                pred.resolved = True
                pred.resolution_timestamp = datetime.now().isoformat()
                pred.actual_outcome = outcome

                # Check if correct
                pred.prediction_correct = (pred.our_prediction == outcome)

                # Calculate profit/loss
                if pred.our_prediction == "YES":
                    if outcome == "YES":
                        # We bet YES, it resolved YES
                        pred.profit_loss_percent = (1.0 - pred.market_yes_price) / pred.market_yes_price * 100
                    else:
                        # We bet YES, it resolved NO
                        pred.profit_loss_percent = -100.0
                else:  # We predicted NO
                    if outcome == "NO":
                        # We bet NO, it resolved NO
                        pred.profit_loss_percent = (1.0 - pred.market_no_price) / pred.market_no_price * 100
                    else:
                        # We bet NO, it resolved YES
                        pred.profit_loss_percent = -100.0

                resolved_count += 1
                print(f"Resolved: {pred.market_question[:50]}... -> {outcome} (Correct: {pred.prediction_correct})")

        if resolved_count > 0:
            self._save_data()
            self.calculate_metrics()

        return resolved_count

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Returns:
            Updated PerformanceMetrics
        """
        resolved = [p for p in self.predictions.values() if p.resolved]
        pending = [p for p in self.predictions.values() if not p.resolved]

        self.metrics.total_predictions = len(self.predictions)
        self.metrics.resolved_predictions = len(resolved)
        self.metrics.pending_predictions = len(pending)

        if not resolved:
            return self.metrics

        # Basic accuracy
        correct = [p for p in resolved if p.prediction_correct]
        self.metrics.correct_predictions = len(correct)
        self.metrics.incorrect_predictions = len(resolved) - len(correct)
        self.metrics.accuracy_rate = len(correct) / len(resolved) if resolved else 0.0

        # By prediction type
        yes_preds = [p for p in resolved if p.our_prediction == "YES"]
        no_preds = [p for p in resolved if p.our_prediction == "NO"]

        self.metrics.yes_predictions = len(yes_preds)
        self.metrics.yes_correct = len([p for p in yes_preds if p.prediction_correct])
        self.metrics.yes_accuracy = self.metrics.yes_correct / len(yes_preds) if yes_preds else 0.0

        self.metrics.no_predictions = len(no_preds)
        self.metrics.no_correct = len([p for p in no_preds if p.prediction_correct])
        self.metrics.no_accuracy = self.metrics.no_correct / len(no_preds) if no_preds else 0.0

        # Edge analysis
        edges_claimed = [p.our_edge_percent for p in resolved]
        self.metrics.avg_edge_claimed = sum(edges_claimed) / len(edges_claimed) if edges_claimed else 0.0

        # Actual edge captured (only for correct predictions)
        edges_captured = [p.profit_loss_percent for p in correct if p.profit_loss_percent]
        self.metrics.avg_edge_captured = sum(edges_captured) / len(edges_captured) if edges_captured else 0.0

        self.metrics.edge_capture_rate = (
            self.metrics.avg_edge_captured / self.metrics.avg_edge_claimed
            if self.metrics.avg_edge_claimed > 0 else 0.0
        )

        # Confidence calibration
        high_conf = [p for p in resolved if p.our_confidence >= 0.8]
        med_conf = [p for p in resolved if 0.6 <= p.our_confidence < 0.8]
        low_conf = [p for p in resolved if p.our_confidence < 0.6]

        self.metrics.high_confidence_accuracy = (
            len([p for p in high_conf if p.prediction_correct]) / len(high_conf)
            if high_conf else 0.0
        )
        self.metrics.medium_confidence_accuracy = (
            len([p for p in med_conf if p.prediction_correct]) / len(med_conf)
            if med_conf else 0.0
        )
        self.metrics.low_confidence_accuracy = (
            len([p for p in low_conf if p.prediction_correct]) / len(low_conf)
            if low_conf else 0.0
        )

        # Swarm consensus analysis
        consensus_preds = [p for p in resolved if p.swarm_consensus]
        if consensus_preds:
            consensus_agreed = [p for p in consensus_preds if p.swarm_consensus == p.our_prediction]
            self.metrics.consensus_accuracy = (
                len([p for p in consensus_agreed if p.prediction_correct]) / len(consensus_agreed)
                if consensus_agreed else 0.0
            )

            split_preds = [p for p in consensus_preds if p.swarm_strength and p.swarm_strength < 0.6]
            self.metrics.split_vote_accuracy = (
                len([p for p in split_preds if p.prediction_correct]) / len(split_preds)
                if split_preds else 0.0
            )

        # Time info
        self.metrics.last_updated = datetime.now().isoformat()

        timestamps = [p.timestamp for p in self.predictions.values()]
        if timestamps:
            self.metrics.oldest_prediction = min(timestamps)
            self.metrics.newest_prediction = max(timestamps)

        self._save_data()
        return self.metrics

    def get_performance_report(self) -> str:
        """Generate a human-readable performance report.

        Returns:
            Formatted report string
        """
        m = self.calculate_metrics()

        report = f"""
{'='*70}
POLYMARKET PREDICTION PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

OVERVIEW
--------
Total Predictions: {m.total_predictions}
Resolved: {m.resolved_predictions}
Pending: {m.pending_predictions}

ACCURACY
--------
Overall Accuracy: {m.accuracy_rate:.1%} ({m.correct_predictions}/{m.resolved_predictions})

By Prediction Type:
  YES Predictions: {m.yes_accuracy:.1%} ({m.yes_correct}/{m.yes_predictions})
  NO Predictions:  {m.no_accuracy:.1%} ({m.no_correct}/{m.no_predictions})

EDGE ANALYSIS
-------------
Avg Edge Claimed: {m.avg_edge_claimed:.1f}%
Avg Edge Captured: {m.avg_edge_captured:.1f}%
Edge Capture Rate: {m.edge_capture_rate:.1%}

CONFIDENCE CALIBRATION
----------------------
High Confidence (>80%):   {m.high_confidence_accuracy:.1%}
Medium Confidence (60-80%): {m.medium_confidence_accuracy:.1%}
Low Confidence (<60%):    {m.low_confidence_accuracy:.1%}

SWARM CONSENSUS
---------------
When Consensus Agreed: {m.consensus_accuracy:.1%}
Split Vote Accuracy:   {m.split_vote_accuracy:.1%}

{'='*70}
"""
        return report

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export predictions to CSV.

        Args:
            filepath: Output file path (optional)

        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = str(DATA_DIR / f"predictions_export_{datetime.now().strftime('%Y%m%d')}.csv")

        fieldnames = [
            "prediction_id", "market_id", "market_question", "timestamp",
            "our_prediction", "our_edge_percent", "our_confidence", "our_fair_value",
            "market_yes_price", "market_no_price", "market_volume",
            "swarm_consensus", "swarm_strength",
            "resolved", "actual_outcome", "prediction_correct", "profit_loss_percent"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for pred in self.predictions.values():
                row = {k: getattr(pred, k, "") for k in fieldnames}
                writer.writerow(row)

        return filepath

    def get_recent_predictions(self, days: int = 7) -> List[PredictionRecord]:
        """Get predictions from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent predictions
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = []

        for pred in self.predictions.values():
            try:
                pred_time = datetime.fromisoformat(pred.timestamp)
                if pred_time >= cutoff:
                    recent.append(pred)
            except ValueError:
                continue

        return sorted(recent, key=lambda p: p.timestamp, reverse=True)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    print("=" * 70)
    print("PERFORMANCE TRACKER TEST")
    print("=" * 70)

    with PerformanceTracker() as tracker:
        # Show current metrics
        print(tracker.get_performance_report())

        # Check for resolved markets
        print("\nChecking for resolved markets...")
        resolved = tracker.update_resolved_predictions()
        print(f"Newly resolved: {resolved}")

        # Show recent predictions
        recent = tracker.get_recent_predictions(days=30)
        print(f"\nRecent predictions (last 30 days): {len(recent)}")

        for pred in recent[:5]:
            status = "Resolved" if pred.resolved else "Pending"
            correct = f"({'CORRECT' if pred.prediction_correct else 'WRONG'})" if pred.resolved else ""
            print(f"  {pred.our_prediction} on {pred.market_question[:40]}... [{status}] {correct}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
