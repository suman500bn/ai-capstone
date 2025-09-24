import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm_response(prompt):
    """
    Sends a prompt to the LLM via LangChain and returns the response.
    """
    try:
        # Get the API key from the loaded environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Initialize the LangChain model with your API key
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )

        # Invoke the model with the user's prompt as a HumanMessage
        response = model.invoke([HumanMessage(content=prompt)])

        return response.content
    except Exception as e:
        return f"An error occurred: {e}"


print("I'm a general purpose chatbot. Type 'exit' to quit.")
while True:
    user_query = input("Enter query: ")
    if user_query.lower() == 'exit':
        break
    response = get_llm_response(user_query)
    print("Response:", response)

print("---------------------------------")

