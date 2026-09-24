from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY


llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def get_response(prompt):

    response = llm.invoke(prompt)

    return response.text