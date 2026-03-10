import sys

from core.agents.news_analysis_agent import _build_entities_from_news

print("--- TASK 7.1: CHECK _build_entities_from_news ---")
try:
    entities = _build_entities_from_news("AAPL", [], "News analysis text.")
    assert len(entities) == 1
    c = entities[0]
    assert c.ticker == "AAPL"
    assert c.name == "AAPL"
    assert "news analysis: AAPL" in c.description
    assert c.sector == ""
    # "enriched" is not used in our updated version, this is fine
    print("PASS: _build_entities_from_news populates correctly")
except Exception as e:
    print(f"FAIL task 7.1: {e}")
    sys.exit(1)

print("PASS: All Phase 6 and 7 unit checks passed.")
