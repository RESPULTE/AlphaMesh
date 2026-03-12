import asyncio
import logging
import os
import re
import threading
import traceback
from typing import Any, Union

import pandas as pd
import streamlit as st

# Silence gRPC internal logging
os.environ["GRPC_VERBOSITY"] = "ERROR"
# Optional: Force gRPC to use a specific polling strategy (mostly for linux, but helps consistency)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

from core.agents.orchestrator_agent import FinalResponse, OrchestratorAgent
from core.config import settings
from core.services import service_manager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaMeshUI")


def format_financial_value(val: Any) -> str:
    """Formats large numbers into readable K, M, B, T suffixes."""
    if not isinstance(val, (int, float)) or pd.isna(val):
        return str(val)

    abs_val = abs(val)
    if abs_val >= 1e12:
        return f"{val/1e12:,.2f}T"
    if abs_val >= 1e9:
        return f"{val/1e9:,.2f}B"
    if abs_val >= 1e6:
        return f"{val/1e6:,.2f}M"
    if abs_val >= 1e3:
        return f"{val/1e3:,.2f}K"
    return f"{val:,.2f}"


class AsyncLoopThread:
    _loop = None
    _thread = None

    @classmethod
    def get_loop(cls):
        if cls._loop is None:
            # Ensure we are using the default policy for Windows (Proactor)
            try:
                # This ensures the loop is created with the correct Windows policy
                cls._loop = asyncio.new_event_loop()
            except AttributeError:
                cls._loop = asyncio.SelectorEventLoop()

            def start_loop(loop):
                # Critical: Set the loop for this specific background thread
                asyncio.set_event_loop(loop)
                loop.run_forever()

            cls._thread = threading.Thread(
                target=start_loop, args=(cls._loop,), daemon=True
            )
            cls._thread.start()
        return cls._loop

    @classmethod
    def run_coroutine(cls, coro) -> Any:
        # Use the established loop
        future = asyncio.run_coroutine_threadsafe(coro, cls.get_loop())
        return future.result()


class AlphaMeshUI:
    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "orchestrator" not in st.session_state:

            async def init():
                service_manager.get_neo4j_adapter()
                await service_manager.get_nodeset_manager().get_global_financial_events_id()
                chroma_adapter = service_manager.get_chroma_adapter()
                await chroma_adapter.get_or_create_collection(
                    settings.CHROMA_COLLECTION_NEWS
                )
                service_manager.get_ingestor()
                return OrchestratorAgent()

            st.session_state.orchestrator = AsyncLoopThread.run_coroutine(init())

    def inject_custom_css(self):
        """Adds styles for tooltips and table layout."""
        st.markdown(
            """
            <style>
            .citation-link {
                color: #007bff;
                text-decoration: none;
                font-weight: bold;
                cursor: help;
                position: relative;
                display: inline-block;
            }
            .citation-link:hover { text-decoration: underline; }
            [data-testid="stMetricValue"] { font-size: 1.8rem; }
            </style>
        """,
            unsafe_allow_html=True,
        )

    def render_narrative_with_citations(self, text: str, sources: list):
        """
        Replaces [n] citations with HTML links containing tooltips.
        """
        source_map = {str(src.source_id): src for src in sources}

        def replace_with_tooltip(match):
            citation_id = match.group(1)
            if citation_id in source_map:
                src = source_map[citation_id]
                # HTML title attribute creates a native browser tooltip
                return f'<a href="{src.url}" target="_blank" class="citation-link" title="Source {citation_id}: {src.title}">[{citation_id}]</a>'
            return f"[{citation_id}]"

        # Regex to find [1], [2] etc.
        processed_text = re.sub(r"\[(\d+)\]", replace_with_tooltip, text)
        st.markdown(processed_text.replace("\n", "  \n"), unsafe_allow_html=True)

    def render_response(self, response: Union[str, FinalResponse]):
        if isinstance(response, str):
            st.markdown(response)
            return

        # 1. Narrative with Interactive Citations
        self.render_narrative_with_citations(response.summary, response.sources)

        # 2. Professional Table
        if (
            response.fundamental_data is not None
            and not response.fundamental_data.empty
        ):
            st.markdown("### 📊 Fundamental Data")

            # Create a formatted copy of the dataframe
            formatted_df = response.fundamental_data.copy()
            for col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(format_financial_value)

            # Use st.dataframe for an interactive, auto-width table
            st.dataframe(
                formatted_df,
                width="content",
                column_config={
                    "index": st.column_config.Column("Metric", width="medium")
                },
            )

        # 3. Reference List
        if response.sources:
            st.markdown("---")
            st.markdown("### 📚 References")
            for src in response.sources:
                st.markdown(f"**[{src.source_id}]** [{src.title}]({src.url})")

    def run(self):
        st.set_page_config(page_title="AlphaMesh AI", layout="wide", page_icon="📈")
        self.inject_custom_css()

        st.title("🎼 AlphaMesh Orchestrator")
        st.caption("Professional Multi-Agent Financial Intelligence")

        with st.sidebar:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        # Display Chat
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                self.render_response(msg["content"])

        if prompt := st.chat_input("Analyze NVIDIA's recent revenue growth..."):
            # 1. Append the new user prompt to session state
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("🧠 Orchestrating...", expanded=True)
                try:
                    # 2. CONVERT session_state.messages to LangChain format
                    from langchain_core.messages import AIMessage, HumanMessage

                    langchain_history = []
                    for m in st.session_state.messages:
                        if m["role"] == "user":
                            langchain_history.append(HumanMessage(content=m["content"]))
                        else:
                            # If content is FinalResponse object, extract summary for context
                            content = (
                                m["content"].summary
                                if hasattr(m["content"], "summary")
                                else str(m["content"])
                            )
                            langchain_history.append(AIMessage(content=content))

                    # 3. Pass the full history instead of just prompt
                    res = AsyncLoopThread.run_coroutine(
                        st.session_state.orchestrator.run(langchain_history)
                    )

                    status.update(label="✅ Success", state="complete", expanded=False)
                    self.render_response(res)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": res}
                    )
                except Exception as e:
                    status.update(label="❌ Analysis Failed", state="error")
                    st.error(f"Error: {str(e)}")
                    st.code(traceback.format_exc())


def main():
    AlphaMeshUI().run()
