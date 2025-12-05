# Polymarket Swarm Trader v2.0

AI-powered trading intelligence for Polymarket prediction markets using a swarm of agents with diverse cognitive patterns.

## What's New in v2.0

Based on critical review of [Polymarket/agents](https://github.com/Polymarket/agents):

- **Pydantic Models**: Type-safe data handling
- **httpx Client**: Async-capable HTTP client
- **RAG Market Selection**: Semantic search with ChromaDB
- **enableOrderBook Filter**: Only tradeable markets
- **CLOB Token Tracking**: Ready for order execution
- **Superforecaster Methodology**: Proven probability estimation

## Features

- **5 Specialized Agents**: Market Scanner, News Researcher, Probability Analyst, Risk Validator, Coordinator
- **Cognitive Diversity**: Each agent uses different thinking patterns (convergent, divergent, critical, systems, adaptive)
- **Real-time Data**: Integrates Polymarket Gamma API and news sources
- **Smart Filtering**: Focuses on politics, economy, tech - excludes crypto/sports
- **MCP Integration**: Can run as distributed swarm via ruv-swarm

## Quick Start

```bash
# Install dependencies
pip install -r requirements-polymarket.txt

# Set up environment variables
cp .env.example .env
# Add: OPENAI_API_KEY, NEWSAPI_API_KEY (optional), FINNHUB_API_KEY (optional)

# Run a scan
cd scripts/polymarket_swarm
python run.py

# Or use CLI
python cli.py scan --max-markets 50 --min-edge 10

# Search by category
python cli.py category politics --limit 20

# Analyze specific market
python cli.py analyze 123456

# Get MCP swarm setup commands
python cli.py mcp-setup
```

## Architecture

```
┌─────────────────────────────────────────┐
│           Coordinator Agent             │
│        (Adaptive Thinking)              │
│   - Synthesizes all agent results       │
│   - Weighted consensus voting           │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───┴───┐   ┌─────┴─────┐   ┌───┴───┐
│Market │   │Probability│   │ Risk  │
│Scanner│   │ Analyst   │   │Validat│
│(conv.)│   │ (critical)│   │(syst.)│
└───┬───┘   └─────┬─────┘   └───┬───┘
    │             │             │
    │       ┌─────┴─────┐       │
    │       │   News    │       │
    │       │Researcher │       │
    │       │ (diverg.) │       │
    │       └───────────┘       │
    └─────────────┴─────────────┘
```

## Agent Cognitive Patterns

| Agent | Pattern | Focus |
|-------|---------|-------|
| Market Scanner | Convergent | Narrows to specific metrics (volume, liquidity, price) |
| News Researcher | Divergent | Expands to find diverse news sources |
| Probability Analyst | Critical | Questions assumptions, calculates edge |
| Risk Validator | Systems | Sees interconnections, feedback loops |
| Coordinator | Adaptive | Adjusts based on agent inputs |

## Data Sources

| Source | Purpose |
|--------|---------|
| Polymarket Gamma API | Market data, prices, CLOB tokens |
| NewsAPI | News articles for sentiment |
| Finnhub | Financial news backup |
| WebSearch | Breaking news verification |

## Output

Opportunities are reported with:
- **Recommendation**: YES or NO
- **Edge**: Estimated % advantage over market
- **Confidence**: Swarm consensus score
- **Reasoning**: Why this trade
- **Risk Factors**: Identified concerns
- **Executable**: Whether trade can be placed

## MCP Swarm Integration

For distributed processing with ruv-swarm:

```bash
python cli.py mcp-setup
```

This outputs the MCP commands to:
1. Initialize swarm topology
2. Create DAA agents with cognitive patterns
3. Set up analysis workflow
4. Execute distributed analysis

## Configuration

Edit `config.py` to adjust:
- `MIN_EDGE_PERCENT`: Minimum edge to recommend (default: 10%)
- `MIN_CONFIDENCE`: Minimum confidence score (default: 0.7)
- `MIN_VOLUME_USD`: Minimum market volume (default: $10,000)
- `EXCLUDED_CATEGORIES`: Markets to skip
- `TARGET_CATEGORIES`: Markets to prioritize
- `AGENT_WEIGHTS`: Relative importance of each agent

## File Structure

```
scripts/polymarket_swarm/
├── __init__.py           # Package init
├── config.py             # Configuration settings
├── models.py             # Pydantic data models
├── polymarket_client.py  # Gamma API client
├── market_rag.py         # RAG market selector
├── news_client.py        # News research client
├── agents.py             # 5 swarm agents
├── orchestrator.py       # Swarm orchestrator
├── mcp_integration.py    # MCP/DAA integration
├── cli.py                # Command-line interface
├── run.py                # Quick runner
├── test_*.py             # Test files
└── README.md             # This file
```

## Running Tests

```bash
cd scripts/polymarket_swarm
python -m pytest -v
```

## License

MIT
