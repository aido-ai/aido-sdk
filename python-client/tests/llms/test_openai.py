from aido_client.llms.openai import ChatOpenAI


def test_invoke():
    chat_client = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key="sk-1234567890",
        openai_api_base="https://openai.aido.ai/v1"
    )
    response = chat_client.invoke("Hello, how are you?")
    print(response)
