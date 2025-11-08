from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def get_gemini_response(prompt: str) -> str:
    """Generate response from Gemini using LangChain wrapper."""
    response = llm.invoke(prompt)
    # .invoke() returns a ChatMessage or string-like object
    return response.content if hasattr(response, "content") else str(response)