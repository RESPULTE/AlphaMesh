RAG_PROMPTS = {
    "rewrite_query": ChatPromptTemplate.from_template(
        "You are a helpful assistant that rewrites queries to be more effective for retrieval. \n"
        "Original query: {query} \n"
        "Output only the rewritten query, nothing else."
    ),
    "retrieval_grader": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document to a user question. \n"
                "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n"
                "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
            ),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {query}",
            ),
        ]
    ),
    "rag_generation": ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n"
        "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
        "Question: {query} \n"
        "Context: {context} \n"
        "Answer:"
    ),
    "hallucination_grader": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
                "Give a binary score 'yes' or 'no'. 'yes' means the answer is fully supported by the facts.",
            ),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    ),
    "answer_grader": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an answer addresses / resolves a question. \n"
                "Give a binary score 'yes' or 'no'. 'yes' means the answer resolves the question.",
            ),
            ("human", "User question: {query} \n\n LLM generation: {generation}"),
        ]
    ),
}
