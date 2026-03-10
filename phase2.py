print("--- TASK 2.1: CHECK LEAN_SUMMARY_SYSTEM_PROMPT ---")
try:
    from core.memory.prompts import LEAN_SUMMARY_SYSTEM_PROMPT

    assert "NO_FINANCIAL_DATA" in LEAN_SUMMARY_SYSTEM_PROMPT
    assert len(LEAN_SUMMARY_SYSTEM_PROMPT) < 500, "Prompt should be terse"
    print("PASS: LEAN_SUMMARY_SYSTEM_PROMPT present and terse")
except Exception as e:
    print(f"FAIL task 2.1: {e}")

print("--- TASK 2.2: CHECK SYNTHESISER_WRITEBACK_SYSTEM_PROMPT ---")
try:
    from core.memory.prompts import SYNTHESISER_WRITEBACK_SYSTEM_PROMPT

    required_strings = [
        "<relationships>",
        "<response>",
        "AFFECTS",
        "CAUSED_BY",
        "INCREASES",
        "DECREASES",
        "CORRELATED_WITH",
        "MITIGATES",
        "EXPOSES_TO",
        "confidence",
        "high | low",
        "Do not output anything outside these two blocks",
    ]
    for s in required_strings:
        assert (
            s in SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
        ), f"Missing required string: {s!r}"
    print("PASS: SYNTHESISER_WRITEBACK_SYSTEM_PROMPT contains all required elements")
except Exception as e:
    print(f"FAIL task 2.2: {e}")

print("--- TASK 2.3: CHECK EXPORTS ---")
try:
    from core.memory import (
        LEAN_SUMMARY_SYSTEM_PROMPT,
        SYNTHESISER_WRITEBACK_SYSTEM_PROMPT,
    )

    print("PASS: new prompts exported from core.memory")
except Exception as e:
    print(f"FAIL task 2.3: {e}")
