"""Prompt builders for the dual-store retriever."""

from langchain_core.prompts import ChatPromptTemplate


def build_node_selection_prompt() -> ChatPromptTemplate:
    """Build the prompt used to select which graph nodes to expand."""
    system_message = (
        "You are a financial knowledge graph traversal agent. "
        "Your job is to review candidate neighboring entities and select which ones "
        "are most relevant to the user's query for further expansion. "
        "Select no more than {max_parallel_nodes} entities. "
        "If none of the candidates are relevant, return an empty list to terminate traversal. "
        "Prioritize entities with relationship types that are causally or temporally "
        "relevant to the query over entities that are merely co-mentioned. "
        "If the candidate set is mostly redundant with what has already been found, "
        "return an empty list."
    )

    human_message = (
        "Query:\n{query}\n\n"
        "Iteration: {iteration} / {max_iterations}\n"
        "Already retrieved chunks: {already_retrieved_count}\n\n"
        "Candidate neighbors:\n{candidate_neighbors}\n\n"
        "Return a list of selected entity IDs."
    )

    return ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", human_message)]
    )
