"""OpenAI chat wrapper."""
from __future__ import annotations

import logging
import sys
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Tuple, Type, Union

from pydantic import BaseModel, Extra, Field, root_validator
from aido_client.callbacks.base import BaseCallbackManager
from aido_client.llms.base import BaseChatModel

from aido_client.schemas.schema import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatGeneration,
    ChatGenerationChunk,
    ChatMessage,
    ChatMessageChunk,
    ChatResult,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from aido_client.utils.utils import get_from_dict_or_env

logger = logging.getLogger(__file__)


def recalculate(
        llm: ChatOpenAI, messages: List[BaseMessage], max_tokens: Optional[int]
) -> Optional[int]:
    if max_tokens is None:
        return None
    # TODO(ericjiang) config this
    model_limited_tokens = 8192
    message_tokens = llm.get_messages_tokens(messages)

    if message_tokens + max_tokens > model_limited_tokens:
        return max(model_limited_tokens - message_tokens, 16)
    else:
        return max_tokens


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_dict_to_message(_dict: dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    """Generate from a stream."""

    generation: Optional[ChatGenerationChunk] = None
    for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


async def agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    """Async generate from a stream."""

    generation: Optional[ChatGenerationChunk] = None
    async for chunk in stream:
        if generation is None:
            generation = chunk
        else:
            generation += chunk
    assert generation is not None
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


def message_chunk_to_message(chunk: BaseMessageChunk) -> BaseMessage:
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    # chunk classes always have the equivalent non-chunk class as their first parent
    return chunk.__class__.__mro__[1](
        **{k: v for k, v in chunk.__dict__.items() if k != "type"}
    )


class ChatOpenAI(BaseChatModel, BaseModel):
    """Wrapper around OpenAI Chat large language models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from aido import ChatOpenAI
            openai = ChatOpenAI(model_name="gpt-3.5-turbo")

    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_organization: Optional[str] = None
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=240, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or
        None."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.ignore

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}
        
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @root_validator
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if values["n"] < 1:
            raise ValueError("n must be at least 1.")
        if values["n"] > 1 and values["streaming"]:
            raise ValueError("n must be 1 when streaming.")

        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        openai_api_base = get_from_dict_or_env(
            values, "openai_api_base", "OPENAI_API_BASE"
        )
        try:
            import openai
            # fix 直接修改openai的全局变量存在问题，修改会影响同一个程序里面的其它地方
            # openai.api_key = openai_api_key
            # openai.api_base = openai_api_base
            values["openai_api_key"] = openai_api_key
            values["openai_api_base"] = openai_api_base
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        try:
            client_params = {
                "api_key": values["openai_api_key"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
                "default_headers": values["default_headers"],
                "default_query": values["default_query"],
                "http_client": values["http_client"],
            }
            if not values.get("client"):
                values["client"] = openai.OpenAI(**client_params).chat.completions
            if not values.get("async_client"):
                values["async_client"] = openai.AsyncOpenAI(
                    **client_params
                ).chat.completions
        except AttributeError:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`."
            )
        
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            yield chunk
            if streaming_channel:
                streaming_channel.on_llm_new_token(chunk.text, chunk=chunk)
            self.callback_manager.on_llm_new_token(
                chunk.text,
                verbose=self.verbose,
            )

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            for k, v in token_usage.items():
                if k in overall_token_usage:
                    overall_token_usage[k] += v
                else:
                    overall_token_usage[k] = v
        return {"token_usage": overall_token_usage, "model_name": self.model_name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        # recalculate max_tokens
        max_tokens = recalculate(self, messages, self.max_tokens)
        if max_tokens is not None:
            params['max_tokens'] = max_tokens

        should_stream = streaming if streaming is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, streaming_channel=streaming_channel, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": streaming} if streaming is not None else {}),
            **kwargs,
        }
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]], **kwargs: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**{"model": self.model_name}, **self._default_params}
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = {**params, **kwargs}
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        # TODO remove
        print(">>>>>>OpenAI_RAW_Response:", response)
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)
    
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await self.async_client.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            yield chunk
            if streaming_channel:
                await streaming_channel.on_llm_new_token(token=chunk.text, chunk=chunk)
            self.callback_manager.on_llm_new_token(
                chunk.text,
                verbose=self.verbose,
            )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> ChatResult:
        """
        function response
        {
            "choices": [
              {
                "finish_reason": "function_call",
                "index": 0,
                "message": {
                  "content": null,
                  "function_call": {
                    "arguments": "{\n  \"__arg1\": \"100\"\n}",
                    "name": "aido_genrand"
                  },
                  "role": "assistant"
                }
              }
            ],
            "created": 1689907093,
            "id": "chatcmpl-7eaJxcSce5r0lBfZ7csnuu1gzKn0W",
            "model": "gpt-3.5-turbo-0613",
            "object": "chat.completion",
            "usage": {
              "completion_tokens": 19,
              "prompt_tokens": 88,
              "total_tokens": 107
            }
        }
        """
        message_dicts, params = self._create_message_dicts(messages, stop, **kwargs)
        # recalculate max_tokens
        max_tokens = recalculate(self, messages, self.max_tokens)
        if max_tokens is not None:
            params['max_tokens'] = max_tokens

        should_stream = streaming if streaming is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, streaming_channel=streaming_channel, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": streaming} if streaming is not None else {}),
            **kwargs,
        }
        response = await self.async_client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}

    def get_messages_tokens(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in a list of messages.

        Args:
            messages: The messages to count the tokens of.

        Returns:
            The number of tokens in the messages.
        """
        tokens_per_message = 5
        tokens_per_request = 3

        message_tokens = tokens_per_request
        message_strs = ''
        for message in messages:
            message_strs += message.content
            message_tokens += tokens_per_message

        # calc once
        message_tokens += self.get_num_tokens(message_strs)
        return message_tokens

    def get_num_tokens(self, text: str) -> int:
        """Calculate num tokens with tiktoken package."""
        # tiktoken NOT supported for Python 3.8 or below
        if sys.version_info[1] <= 8:
            return super().get_num_tokens(text)
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )
        # create a GPT-3.5-Turbo encoder instance
        enc = tiktoken.encoding_for_model(self.model_name)

        # encode the text using the GPT-3.5-Turbo encoder
        tokenized_text = enc.encode(text)

        # calculate the number of tokens in the encoded text
        return len(tokenized_text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:  # noqa: C901
        """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

        Official documentation: https://github.com/openai/openai-cookbook/blob/
        main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb"""
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please it install it with `pip install tiktoken`."
            )

        model = self.model_name
        if model == "gpt-3.5-turbo":
            # gpt-3.5-turbo may change over time.
            # Returning num tokens assuming gpt-3.5-turbo-0301.
            model = "gpt-3.5-turbo-0301"
        elif model == "gpt-4":
            # gpt-4 may change over time.
            # Returning num tokens assuming gpt-4-0314.
            model = "gpt-4-0314"

        # Returns the number of tokens used by a list of messages.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo-0301":
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_message = 4
            # if there's a name, the role is omitted
            tokens_per_name = -1
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"get_num_tokens_from_messages() is not presently implemented "
                f"for model {model}."
                "See https://github.com/openai/openai-python/blob/main/chatml.md for "
                "information on how messages are converted to tokens."
            )
        num_tokens = 0
        messages_dict = [_convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        # every reply is primed with <im_start>assistant
        num_tokens += 2
        return num_tokens
