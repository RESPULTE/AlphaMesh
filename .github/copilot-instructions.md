# Copilot Instructions for AlphaMesh

## Project Overview
AlphaMesh is a modular Python application for financial data analysis, leveraging SEC Edgar filings, local SQLite storage, and LLM-powered agents (LangChain v1.0). The architecture is designed for extensibility, clarity, and efficient data workflows.

## Architecture & Key Components
- **app/**: Frontend and user-facing logic (e.g., `app.py`, `auth.py`).
- **core/**: Core business logic and services.
  - `core/utils/get_financial_data.py`: Handles SEC Edgar data fetching, SQLite storage, and DataFrame transformations.
  - `core/services.py`: Centralized service manager for LLM, embeddings, Neo4j, and Chroma vector store. Use `service_manager` singleton for dependency injection.
  - `core/agents/`: Contains agent implementations (e.g., `fundamental_analysis_agent.py`) orchestrating LLM workflows and tool calls.
- **data/**: Persistent storage (e.g., Chroma DB, SQLite files).
- **testing/**, **tests/**: Test data and test scripts.

## Developer Workflows
- **Run Agents**: Execute agent scripts directly (e.g., `python core/agents/fundamental_analysis_agent.py`). Agents use LangChain's `AgentExecutor` and tool system.
- **Database**: Financial data is stored in SQLite (`financial_data.db`). The system auto-fetches missing years from Edgar.
- **LLM Integration**: All LLM calls are managed via `service_manager.get_llm()`. Do not instantiate LLMs directly.
- **Testing**: Place test scripts in `tests/`. Use pytest for running tests (`pytest tests/`).

## Patterns & Conventions
- **Modularity**: Use classes for data access (`FinancialDatabase`), analysis (`FinancialAnalyzer`), and orchestration (agents).
- **LangChain v1.0**: Agents are built with `create_tool_calling_agent`, tools use the `@tool` decorator, and prompts use `ChatPromptTemplate`.
- **Error Handling**: All data fetch and analysis functions catch exceptions and return user-friendly error messages.
- **Extensibility**: Add new tools by defining functions with `@tool` and registering them in agent constructors.
- **Logging**: Use Python's `logging` module for query and analysis steps.

## Integration Points
- **External APIs**: SEC Edgar (via `edgar` Python package), Google Generative AI, Neo4j, Chroma DB.
- **Cross-Component Communication**: Agents call tools, which interact with database and service manager. Data flows from Edgar → SQLite → DataFrame → Analysis → LLM output.

## Examples
- To fetch and analyze financials: see `core/agents/fundamental_analysis_agent.py`.
- To add a new financial metric: extend `FinancialAnalyzer` and update agent tools.
- To use LLM: always call `service_manager.get_llm()`.

## Quick Reference
- **Key files**: `core/utils/get_financial_data.py`, `core/services.py`, `core/agents/fundamental_analysis_agent.py`
- **Run agent**: `python core/agents/fundamental_analysis_agent.py`
- **Test**: `pytest tests/`

---
*Update this file as architecture or workflows evolve. Feedback welcome!*
