import asyncio
import logging
import threading
import traceback
from typing import Any, Union

import streamlit as st

from core.agents.orchestrator_agent import FinalResponse, OrchestratorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaMeshUI")


class AsyncLoopThread:
    _loop = None
    _thread = None

    @classmethod
    def get_loop(cls):
        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()
            cls._thread = threading.Thread(target=cls._loop.run_forever, daemon=True)
            cls._thread.start()
        return cls._loop

    @classmethod
    def run_coroutine(cls, coro) -> Any:
        import asyncio

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
                return OrchestratorAgent()

            st.session_state.orchestrator = AsyncLoopThread.run_coroutine(init())

    def render_response(self, response: Union[str, FinalResponse]):
        """Custom renderer for structured response objects."""
        if isinstance(response, str):
            st.markdown(response)
            return

        # 1. Narrative Text
        st.markdown(response.summary)

        # 2. Tables (Quantitative Data)
        if (
            response.fundamental_data is not None
            and not response.fundamental_data.empty
        ):
            st.markdown("### 📊 Financial Metrics")
            st.dataframe(
                response.fundamental_data,
                use_container_width=True,
                column_config={"index": "Metric"},
            )

        # 3. References (Qualitative Data)
        if response.sources:
            st.markdown("---")
            st.markdown("### 📚 Sources & Citations")
            for src in response.sources:
                with st.expander(f"[{src.source_id}] {src.title}"):
                    st.markdown(f"**URL:** {src.url}")
                    st.write(src.page_content)

    def setup_page(self):
        st.set_page_config(page_title="AlphaMesh AI", layout="wide", page_icon="📈")
        st.title("🎼 AlphaMesh Orchestrator")
        st.markdown("---")

    def run(self):
        self.setup_page()

        # Sidebar for management
        with st.sidebar:
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

        # Display Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                self.render_response(msg["content"])

        # Input
        if prompt := st.chat_input("Analyze NVIDIA's recent performance..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("🧠 Processing...", expanded=True)
                try:
                    response = AsyncLoopThread.run_coroutine(
                        st.session_state.orchestrator.run(prompt)
                    )
                    status.update(
                        label="✅ Analysis Complete", state="complete", expanded=False
                    )

                    self.render_response(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(str(e))
                    st.code(traceback.format_exc())


def main():
    ui = AlphaMeshUI()
    ui.run()
