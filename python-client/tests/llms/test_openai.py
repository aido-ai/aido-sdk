import logging
from typing import List, Optional
from aido_client.chains.llm import LLMChain
from aido_client.llms.openai import ChatOpenAI
from aido_client.prompts.base import BasePromptTemplate
from aido_client.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from aido_client.utils.utils import get_from_dict_or_env

logger = logging.getLogger(__file__)


LEEPY_SOLVER = """{context}
你是一个LeetCode的解题大师，你可以基于问题以及上面检索出的参考思路，给出一个最优的代码来解决问题，请使用python语言，问题是：{question}
"""

openai_api_key = get_from_dict_or_env(
    data={},
    key="openai_api_key",
    env_key="OPENAI_API_KEY",
)


def test_invoke():
    chat_client = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        openai_api_base="https://openai.aido.ai/v1"
    )
    response = chat_client.invoke("Hello, how are you?")
    logger.warning(response)


def test_chain():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        openai_api_base="https://openai.aido.ai/v1"
    )
    # 1. generate answer
    prompt = create_prompt(human_message=LEEPY_SOLVER, input_variables=["question", "context"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    result = llm_chain.predict(**{"question": "给定一个字符串，找到最长的回文子串。", "context": ""})
    logger.warning(f'给定一个字符串，找到最长的回文子串。:{result}')


def create_prompt(
    system_message: str = "",
    human_message: str = "",
    input_variables: Optional[List[str]] = None,
) -> BasePromptTemplate:
    if input_variables is None:
        input_variables = ["query"]
    messages = [
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(human_message),
    ]
    return ChatPromptTemplate(input_variables=input_variables, messages=messages)
