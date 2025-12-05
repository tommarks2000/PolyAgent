#!/usr/bin/env python3
"""Quick runner for Polymarket Swarm analysis."""
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import SwarmOrchestrator


def main():
    print("=" * 70)
    print("POLYMARKET SWARM TRADER v2.0")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("Target: Politics, Economy, Technology, Entertainment, World")
    print("Excluded: Crypto, Sports")
    print("Minimum Edge: 10% | Minimum Confidence: 70%")
    print()

    with SwarmOrchestrator(use_rag=False) as orchestrator:
        # Run analysis
        opportunities = orchestrator.find_opportunities(max_markets=30)

        # Generate and print report
        print()
        print(orchestrator.generate_report(opportunities))

        # Save results
        if opportunities:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"polymarket_opportunities_{timestamp}.json"

            with open(output_file, 'w') as f:
                f.write(orchestrator.to_json(opportunities))

            print(f"\nResults saved to: {output_file}")

    return opportunities


if __name__ == "__main__":
    main()
