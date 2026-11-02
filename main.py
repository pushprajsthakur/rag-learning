from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version as pkg_version

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

print(f'LangChain Core Version: {pkg_version("langchain-core")}')
print(f'LangGraph Version: {pkg_version("langgraph")}')


def main():
    # GROQ Response
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    response = llm.invoke("Say 'Setup Complete!' in one word")
    print(f"Response from Groq: {response.content}")

    # GEMINI Response
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0
    )
    response = llm.invoke("Say 'Setup Complete!' in one word")
    print(f"Response from Google Generative AI: {response.content}")

    print('setup complete!')


if __name__ == '__main__':
    main()