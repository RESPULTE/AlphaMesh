import asyncio
import logging
import threading
import traceback
from concurrent.futures import Future
from datetime import datetime
from typing import Any

import streamlit as st

# Import the Orchestrator
from core.agents.orchestrator_agent import OrchestratorAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaMeshUI")


class AsyncLoopThread:
    """
    Manages a single persistent background thread running an asyncio event loop.
    This prevents the 'Event loop is closed' or 'Attached to a different loop' errors
    common with gRPC/Google Gemini in Streamlit.
    """

    _loop: asyncio.AbstractEventLoop = None
    _thread: threading.Thread = None

    @classmethod
    def get_loop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()
            cls._thread = threading.Thread(target=cls._loop.run_forever, daemon=True)
            cls._thread.start()
            logger.info("✅ Persistent background event loop started.")
        return cls._loop

    @classmethod
    def run_coroutine(cls, coro) -> Any:
        """Submits a coroutine to the background loop and waits for the result."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, cls.get_loop())
        return future.result()


class ChatManager:
    """Handles the storage, retrieval, and modification of chat history."""

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def add_message(self, role: str, content: str):
        st.session_state.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    def delete_message(self, index: int):
        if 0 <= index < len(st.session_state.messages):
            st.session_state.messages.pop(index)
            st.rerun()

    def edit_message(self, index: int, new_content: str):
        if 0 <= index < len(st.session_state.messages):
            st.session_state.messages[index]["content"] = new_content
            st.rerun()

    def clear_history(self):
        st.session_state.messages = []
        st.rerun()


class AlphaMeshUI:
    """Main UI Class to handle rendering and agent interaction."""

    def __init__(self):
        self.chat_manager = ChatManager()
        self._init_session_state()

    def _init_session_state(self):
        """Initializes the Agent within the background loop thread."""
        if "orchestrator" not in st.session_state:
            with st.spinner("🤖 Initializing Orchestrator on background loop..."):
                # We MUST initialize the agent inside the background loop
                # so that its gRPC channels are bound to that loop correctly.
                async def init_agent():
                    return OrchestratorAgent()

                agent = AsyncLoopThread.run_coroutine(init_agent())
                st.session_state.orchestrator = agent

    def setup_page(self):
        st.set_page_config(page_title="AlphaMesh AI", layout="wide", page_icon="📈")
        st.title("🎼 AlphaMesh Orchestrator")
        st.caption("Synchronized Multi-Agent Financial Intelligence")
        st.markdown("---")

    def render_sidebar(self):
        with st.sidebar:
            st.header("🛠️ Chat Management")
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                self.chat_manager.clear_history()

            st.divider()
            st.subheader("📝 Edit History")
            messages = st.session_state.messages

            if not messages:
                st.info("No messages in history.")

            for i, msg in enumerate(messages):
                with st.expander(
                    f"{i}: {msg['role'].upper()} - {msg['timestamp'][-8:]}"
                ):
                    new_txt = st.text_area(
                        "Content", value=msg["content"], key=f"tx_{i}"
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("Save", key=f"sv_{i}"):
                        self.chat_manager.edit_message(i, new_txt)
                    if c2.button("Delete", key=f"dl_{i}", type="primary"):
                        self.chat_manager.delete_message(i)

    def render_chat(self):
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input logic
        if prompt := st.chat_input("Analyze NVIDIA's performance this quarter..."):
            # 1. User Message
            self.chat_manager.add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Assistant Message
            with st.chat_message("assistant"):
                status_box = st.status("🧠 Orchestrating Agents...", expanded=True)

                try:
                    # Run the agent in the persistent background loop
                    response = AsyncLoopThread.run_coroutine(
                        st.session_state.orchestrator.run(prompt)
                    )

                    status_box.update(
                        label="✅ Analysis Complete", state="complete", expanded=False
                    )
                    st.markdown(response)
                    self.chat_manager.add_message("assistant", response)

                except Exception as e:
                    status_box.update(
                        label="❌ Orchestrator Error", state="error", expanded=True
                    )
                    st.error(f"**Error:** {str(e)}")

                    # Provide full traceback for debugging
                    with st.expander("🔍 View Debugging Traceback"):
                        st.code(traceback.format_exc())


def main():
    """Application entry point."""
    ui = AlphaMeshUI()
    ui.setup_page()
    ui.render_sidebar()
    ui.render_chat()


if __name__ == "__main__":
    main()
