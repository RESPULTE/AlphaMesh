from typing import List, Type

from core.agents.base_agent import AbstractAgent, AgentOutput
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.news_analysis_agent import NewsAnalysisAgent
from core.services import service_manager
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# LangChain imports for Tool handling
from langchain_core.tools import StructuredTool

# 1. Agent Registration
AVAILABLE_AGENTS: List[Type[AbstractAgent]] = [
    FundamentalAnalysisAgent,
    NewsAnalysisAgent,
]


class OrchestratorAgent:
    """
    Refactored Orchestrator using Native Tool Calling (Solution 2).
    This ensures input schemas are strictly enforced by the LLM.
    """

    def __init__(self):
        self._llm = service_manager.get_agent(temperature=0)
        # Convert your Agent classes into LangChain Tools
        self._tools = [
            self._create_tool_for_agent(agent_cls) for agent_cls in AVAILABLE_AGENTS
        ]
        # Bind the tools to the LLM immediately
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        # Map tool names back to the original Agent instances for execution logic if needed
        self._agent_map = {agent().name: agent for agent in AVAILABLE_AGENTS}

    def _create_tool_for_agent(self, agent_cls: Type[AbstractAgent]) -> StructuredTool:
        """
        Wraps an AbstractAgent into a LangChain StructuredTool.
        Crucially, this passes the Pydantic 'input_schema' directly to the LLM.
        """
        agent_instance = agent_cls()

        def agent_wrapper(**kwargs):
            """
            The function the LLM will 'call'.
            We instantiate the specific input model from the kwargs provided by the LLM.
            """
            # Validate input using the agent's strict Pydantic model
            input_model = agent_instance.get_input_schema_class()(**kwargs)
            return agent_instance.run(input_model)

        return StructuredTool.from_function(
            func=agent_wrapper,
            name=agent_instance.name,
            description=agent_instance.description,
            # This is the fix: It passes the full schema (types + descriptions) to the LLM
            args_schema=agent_instance.get_input_schema_class(),
        )

    def _run_synthesis(self, query: str, agent_outputs: List[AgentOutput]) -> str:
        """Synthesizes the outputs from various agents into a single response."""
        if not agent_outputs:
            return "No relevant information could be gathered from the agents."

        print("--- [Orchestrator] Aggregating Responses ---")
        formatted_data = "\n\n".join(
            [
                f"## Report from {out.agent_name.upper()}\n{out.output}"
                for out in agent_outputs
            ]
        )

        prompt_template = (
            "You are a Financial Research Lead. You have received reports from your sub-agents.\n"
            "Synthesize these partial reports into a single, cohesive answer to the user's original question.\n\n"
            "**Original User Question:** {original_input}\n\n"
            "**Sub-Agent Reports:**\n{agent_data}\n\n"
            "**Requirements:**\n"
            "- Integrate findings seamlessly.\n"
            "- Connect qualitative and quantitative insights.\n"
            "- Provide a professional final answer."
        )

        prompt = ChatPromptTemplate.from_template(prompt_template)
        synthesis_chain = prompt | self._llm
        response = synthesis_chain.invoke(
            {"agent_data": formatted_data, "original_input": query}
        )
        return response.content

    def run(self, query: str) -> str:
        """Main entry point."""
        print(f"\n--- [Orchestrator] Analyzing query: '{query}' ---")

        # 1. Ask the LLM to decide which tools to call
        # We give it a system prompt to define its persona
        messages = [
            SystemMessage(
                content="You are a financial orchestrator. Analyze the user query and call the appropriate specialist agents to gather information."
            ),
            HumanMessage(content=query),
        ]

        # The LLM returns a message that may contain 'tool_calls'
        ai_msg = self._llm_with_tools.invoke(messages)

        agent_outputs = []

        # 2. Check if the LLM decided to call any tools
        if not ai_msg.tool_calls:
            print(
                "--- [Orchestrator] No tools selected. Returning direct response. ---"
            )
            return ai_msg.content

        print(f"--- [Orchestrator] Plan: Executing {len(ai_msg.tool_calls)} tasks. ---")

        # 3. Execute the tools
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]  # This is the dictionary extracted by the LLM

            print(
                f"\n>>> Calling Agent: {tool_name.upper()} with inputs: {tool_args} <<<"
            )

            # Find the wrapped tool logic
            selected_tool = next((t for t in self._tools if t.name == tool_name), None)

            if selected_tool:
                try:
                    # Execute the wrapper (which validates Pydantic and runs the agent)
                    # Note: StructuredTool.invoke handles the arg passing
                    result_output = selected_tool.invoke(tool_args)

                    # Ensure the result is in AgentOutput format for synthesis
                    # (Assuming the agent.run returns AgentOutput, but tool wrapper might return raw object.
                    # If your agent.run returns AgentOutput, we use it directly.)
                    if isinstance(result_output, AgentOutput):
                        agent_outputs.append(result_output)
                    else:
                        # Fallback if the tool wrapper returned something else
                        agent_outputs.append(
                            AgentOutput(agent_name=tool_name, output=str(result_output))
                        )

                except Exception as e:
                    agent_outputs.append(
                        AgentOutput(
                            agent_name=tool_name,
                            output=f"Error executing agent '{tool_name}': {str(e)}",
                        )
                    )
            else:
                agent_outputs.append(
                    AgentOutput(agent_name=tool_name, output="Tool not found.")
                )

        # 4. Synthesize results
        return self._run_synthesis(query, agent_outputs)


# Execution Block
if __name__ == "__main__":
    orchestrator = OrchestratorAgent()

    # This should now correctly populate inputs because the LLM sees the Pydantic schema
    print(orchestrator.run("Why did NVDA rise recently?"))
