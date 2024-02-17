from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence
from pydantic import BaseModel


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