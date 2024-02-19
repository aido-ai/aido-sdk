from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, TypeVar, Union, cast
from pydantic import BaseModel, Extra, Field, validator
from aido_client.callbacks import get_callback_manager
from aido_client.callbacks.base import BaseCallbackManager
from aido_client.prompts.base import StringPromptValue
from aido_client.prompts.chat import ChatPromptValue

from aido_client.schemas.schema import BaseMessage, BaseMessageChunk, \
    ChatGeneration, ChatGenerationChunk, ChatResult, HumanMessage, LLMResult, PromptValue, get_buffer_string
from aido_client.utils.utils import _get_verbosity

logger = logging.getLogger(__file__)


LanguageModelInput = Union[PromptValue, str, List[BaseMessage]]
LanguageModelOutput = TypeVar("LanguageModelOutput")


class BaseLanguageModel(BaseModel, ABC):
    @abstractmethod
    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Take in a list of prompt values and return an LLMResult."""

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        # TODO: this method may not be exact.
        # TODO: this method may differ based on model (eg codex).
        try:
            from transformers import GPT2TokenizerFast
        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "This is needed in order to calculate get_num_tokens. "
                "Please install it with `pip install transformers`."
            )
        # create a GPT-3 tokenizer instance
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # tokenize the text using the GPT-3 tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # calculate the number of tokens in the tokenized text
        return len(tokenized_text)

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the message."""
        return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])


class BaseExtLanguageModel(BaseLanguageModel):
    model_name: str
    streaming: bool = False
    """Whether to stream the results or not."""

    def predict(
        self,
        text: str,
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> str:
        raise NotImplementedError

    async def apredict(
        self,
        text: str,
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> str:
        raise NotImplementedError

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        raise NotImplementedError

    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        raise NotImplementedError

    def get_messages_tokens(self, messages: List[BaseMessage]) -> int:
        """Get the number of tokens in the messages."""
        raise NotImplementedError


class BaseChatModel(BaseExtLanguageModel, ABC):
    verbose: bool = Field(default_factory=_get_verbosity)
    """Whether to print out response text."""
    callback_manager: BaseCallbackManager = Field(default_factory=get_callback_manager)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True, always=True)
    def set_callback_manager(
        cls, callback_manager: Optional[BaseCallbackManager]
    ) -> BaseCallbackManager:
        """If callback manager is None, set it.

        This allows users to pass in None as callback manager, which is a nice UX.
        """
        return callback_manager or get_callback_manager()

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return {}
    
    def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, list):
            return ChatPromptValue(messages=input)
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )
    
    def invoke(
        self,
        input: LanguageModelInput,
        *,
        streaming_channel: Optional[BaseCallbackManager] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        llm_result = self.generate_prompt(
            [self._convert_input(input)],
            stop=stop,
            streaming_channel=streaming_channel,
            **kwargs,
        )
        llm_result = cast(ChatGeneration, llm_result.generations[0][0])
        
        return llm_result.message

    async def ainvoke(
        self,
        input: LanguageModelInput,
        *,
        streaming_channel: Optional[BaseCallbackManager] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            streaming_channel=streaming_channel,
            **kwargs,
        )
        llm_result = cast(ChatGeneration, llm_result.generations[0][0])
        return llm_result.message
    
    def stream(
        self,
        input: LanguageModelInput,
        *,
        streaming_channel: Optional[BaseCallbackManager] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessageChunk]:
        if type(self)._stream == BaseChatModel._stream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, streaming_channel=streaming_channel, stop=stop, **kwargs)
            )
        else:
            messages = self._convert_input(input).to_messages()
            generation: Optional[ChatGenerationChunk] = None
            try:
                for chunk in self._stream(
                    messages, stop=stop, streaming_channel=streaming_channel, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                self.callback_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                self.callback_manager.on_llm_end(LLMResult(generations=[[generation]]))

    async def astream(
        self,
        input: LanguageModelInput,
        *,
        streaming_channel: Optional[BaseCallbackManager] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[BaseMessageChunk]:
        if type(self)._astream == BaseChatModel._astream:
            # model doesn't implement streaming, so use default implementation
            yield cast(
                BaseMessageChunk, self.invoke(input, streaming_channel=streaming_channel, stop=stop, **kwargs)
            )
        else:
            messages = self._convert_input(input).to_messages()
            generation: Optional[ChatGenerationChunk] = None
            try:
                async for chunk in self._astream(
                    messages, stop=stop, streaming_channel=streaming_channel, **kwargs
                ):
                    yield chunk.message
                    if generation is None:
                        generation = chunk
                    else:
                        generation += chunk
                assert generation is not None
            except BaseException as e:
                await self.callback_manager.on_llm_error(
                    e,
                    response=LLMResult(
                        generations=[[generation]] if generation else []
                    ),
                )
                raise e
            else:
                await self.callback_manager.on_llm_end(
                    LLMResult(generations=[[generation]]),
                )

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Top Level call"""
        results = [self._generate(m, stop=stop, streaming_channel=streaming_channel, **kwargs) for m in messages]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        return LLMResult(generations=generations, llm_output=llm_output)

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        """Top Level call"""
        results = await asyncio.gather(
            *[self._agenerate(m, stop=stop, streaming_channel=streaming_channel, **kwargs) for m in messages]
        )
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        return LLMResult(generations=generations, llm_output=llm_output)

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        prompt_strings = [p.to_string() for p in prompts]
        self.callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
        )
        try:
            output = self.generate(prompt_messages, stop=stop, streaming_channel=streaming_channel, **kwargs)
        except (KeyboardInterrupt, Exception) as e:
            self.callback_manager.on_llm_error(e, verbose=self.verbose)
            raise e
        self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output

    async def agenerate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        prompt_strings = [p.to_string() for p in prompts]
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
            )
        else:
            self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompt_strings, verbose=self.verbose
            )
        try:
            output = await self.agenerate(prompt_messages, stop=stop, streaming_channel=streaming_channel, **kwargs)
        except (KeyboardInterrupt, Exception) as e:
            if self.callback_manager.is_async:
                await self.callback_manager.on_llm_error(e, verbose=self.verbose)
            else:
                self.callback_manager.on_llm_error(e, verbose=self.verbose)
            raise e
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_end(output, verbose=self.verbose)
        else:
            self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError()

    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise NotImplementedError()
    
    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Top Level call"""

    @abstractmethod
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Top Level call"""

    def __call__(
        self,
        messages: Union[str, List[BaseMessage]],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        return self._generate(messages, stop=stop, streaming_channel=streaming_channel, **kwargs).generations[0].message
    
    async def _call_async(
        self,
        messages: Union[str, List[BaseMessage]],
        stop: Optional[List[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        result = await self._agenerate(messages, stop=stop, streaming_channel=streaming_channel, **kwargs)
        return result.generations[0].message
    
    def predict(
        self,
        text: str,
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        self.callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, [text], verbose=self.verbose
        )
        
        result = self([HumanMessage(content=text)], stop=_stop, streaming_channel=streaming_channel, **kwargs)
        
        self.callback_manager.on_llm_end(result, verbose=self.verbose)
        return result.content
    
    async def apredict(
        self,
        text: str,
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> str:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, [text], verbose=self.verbose
            )
        else:
            self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, [text], verbose=self.verbose
            )
            
        output = await self._call_async(
            [HumanMessage(content=text)], stop=_stop, streaming_channel=streaming_channel, **kwargs
        )
        
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_end(output, verbose=self.verbose)
        else:
            self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output.content

    def predict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        
        prompts = [m.content for m in messages]
        self.callback_manager.on_llm_start(
            {"name": self.__class__.__name__}, prompts, verbose=self.verbose
        )
        
        output = self(messages, stop=_stop, streaming_channel=streaming_channel, **kwargs)
        
        self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output
    
    async def apredict_messages(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[Sequence[str]] = None,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> BaseMessage:
        if stop is None:
            _stop = None
        else:
            _stop = list(stop)
        
        prompts = [m.content for m in messages]
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompts, verbose=self.verbose
            )
        else:
            self.callback_manager.on_llm_start(
                {"name": self.__class__.__name__}, prompts, verbose=self.verbose
            )
            
        output = await self._call_async(messages, stop=_stop, streaming_channel=streaming_channel, **kwargs)
        
        if self.callback_manager.is_async:
            await self.callback_manager.on_llm_end(output, verbose=self.verbose)
        else:
            self.callback_manager.on_llm_end(output, verbose=self.verbose)
        return output
