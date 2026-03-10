from core.agents.fundamental_analysis_agent import FundamentalAnalysisOutput
from core.agents.models import BaseAgentOutput
from core.agents.news_analysis_agent import NewsAnalysisOutput

print("--- TASK 1.1: CHECK entities_enriched FIELD ---")
try:
    fields = BaseAgentOutput.model_fields
    assert "entities_enriched" in fields, "entities_enriched field missing"
    assert (
        fields["entities_enriched"].default_factory is not None
    ), "Must have default_factory=list"
    print("PASS: BaseAgentOutput.entities_enriched field present")
except Exception as e:
    print(f"FAIL task 1.1: {e}")

print("--- TASK 1.2: CHECK EXISTING AGENTS INSTANTIATE ---")
try:
    f = FundamentalAnalysisOutput(analysis="test", financial_data=None)
    assert f.entities_enriched == [], "entities_enriched should default to []"
    assert f.agent_name == "fundamentals_agent"

    n = NewsAnalysisOutput(analysis="test", sources=[])
    assert n.entities_enriched == [], "entities_enriched should default to []"
    assert n.agent_name == "news_agent"

    print("PASS: existing agent outputs instantiate correctly")
except Exception as e:
    print(f"FAIL task 1.2: {e}")
