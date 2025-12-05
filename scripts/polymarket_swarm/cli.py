"""Command-line interface for Polymarket Swarm Trader v2.0."""
import argparse
import json
import sys
from datetime import datetime

from orchestrator import SwarmOrchestrator
from mcp_integration import MCPSwarmIntegration
from config import MIN_EDGE_PERCENT, MIN_CONFIDENCE


def cmd_scan(args):
    """Scan markets for opportunities."""
    print(f"Scanning up to {args.max_markets} markets...")
    print(f"Minimum edge: {args.min_edge}%")
    print(f"Minimum confidence: {args.min_confidence}")
    if args.query:
        print(f"Search query: {args.query}")
    print()

    with SwarmOrchestrator(use_rag=bool(args.query)) as orchestrator:
        opportunities = orchestrator.find_opportunities(
            max_markets=args.max_markets,
            search_query=args.query,
            min_edge=args.min_edge,
            min_confidence=args.min_confidence,
        )

        print()
        print(orchestrator.generate_report(opportunities))

        if args.output:
            with open(args.output, 'w') as f:
                f.write(orchestrator.to_json(opportunities))
            print(f"\nSaved to {args.output}")

    return opportunities


def cmd_analyze(args):
    """Analyze a specific market."""
    with SwarmOrchestrator(use_rag=False) as orchestrator:
        market = orchestrator.polymarket.get_market_by_id(args.market_id)

        if not market:
            print(f"Market not found: {args.market_id}")
            return None

        print(f"Analyzing: {market.question}")
        print(f"Current price: {market.yes_price:.1%} YES / {market.no_price:.1%} NO")
        print()

        opportunity = orchestrator.analyze_market(market)

        if opportunity:
            print(f"Recommendation: {opportunity.recommendation.value}")
            print(f"Edge: {opportunity.edge_percent:.1f}%")
            print(f"Confidence: {opportunity.confidence:.1%}")
            print(f"Fair YES price: {opportunity.estimated_fair_yes_price:.1%}")
            print(f"Reasoning: {opportunity.reasoning}")
            print(f"Agent Votes: {dict((k, v.value) for k, v in opportunity.agent_votes.items())}")
            print(f"Risk Factors: {opportunity.risk_factors}")
            print(f"Executable: {'YES' if opportunity.is_executable else 'NO'}")
        else:
            print("No trading opportunity identified.")

        return opportunity


def cmd_mcp_setup(args):
    """Print MCP swarm setup commands."""
    with MCPSwarmIntegration() as mcp:
        mcp.print_mcp_setup_instructions()


def cmd_categories(args):
    """Search by category."""
    categories = {
        "politics": "presidential election government policy voting",
        "economy": "federal reserve interest rates inflation GDP",
        "tech": "artificial intelligence AI technology companies",
        "world": "international geopolitics foreign affairs",
        "entertainment": "movies awards celebrities media",
    }

    if args.category not in categories:
        print(f"Unknown category: {args.category}")
        print(f"Available: {', '.join(categories.keys())}")
        return

    query = categories[args.category]
    print(f"Searching category '{args.category}' with query: {query}")
    print()

    # Use the scan command with the category query
    args.query = query
    args.max_markets = args.limit
    args.min_edge = MIN_EDGE_PERCENT
    args.min_confidence = MIN_CONFIDENCE
    args.output = None

    return cmd_scan(args)


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Swarm Trader v2.0 - Find high-probability trades"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan markets for opportunities")
    scan_parser.add_argument(
        "--max-markets", "-m",
        type=int,
        default=50,
        help="Maximum markets to analyze"
    )
    scan_parser.add_argument(
        "--min-edge", "-e",
        type=float,
        default=MIN_EDGE_PERCENT,
        help="Minimum edge percentage"
    )
    scan_parser.add_argument(
        "--min-confidence", "-c",
        type=float,
        default=MIN_CONFIDENCE,
        help="Minimum confidence score"
    )
    scan_parser.add_argument(
        "--query", "-q",
        type=str,
        help="Semantic search query (enables RAG)"
    )
    scan_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze specific market")
    analyze_parser.add_argument("market_id", type=int, help="Market ID to analyze")

    # Category command
    cat_parser = subparsers.add_parser("category", help="Search by category")
    cat_parser.add_argument(
        "category",
        choices=["politics", "economy", "tech", "world", "entertainment"],
        help="Category to search"
    )
    cat_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum markets to analyze"
    )

    # MCP setup command
    mcp_parser = subparsers.add_parser("mcp-setup", help="Print MCP swarm setup commands")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "category":
        cmd_categories(args)
    elif args.command == "mcp-setup":
        cmd_mcp_setup(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
