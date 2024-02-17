from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, root_validator


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example
        conversation.
    """
    
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class FunctionMessage(BaseMessage):
    """A Message for passing the result of executing a function back to a model."""

    name: str
    """The name of the function that was executed."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class ChatMessage(BaseMessage):
    """Type of message with arbitrary speaker."""

    role: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"


class ToolMessage(BaseMessage):
    """A Message for passing the result of executing a tool back to a model."""

    tool_call_id: str
    """Tool call that this message is responding to."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "tool"


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""


def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ChatMessage):
            role = m.role
        elif isinstance(m, FunctionMessage):
            role = "Function"
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)


def _merge_content(
    first_content: Union[str, List[Union[str, Dict]]],
    second_content: Union[str, List[Union[str, Dict]]],
) -> Union[str, List[Union[str, Dict]]]:
    """Merge two message contents.

    Args:
        first_content: The first content.
        second_content: The second content.

    Returns:
        The merged content.
    """
    # If first chunk is a string
    if isinstance(first_content, str):
        # If the second chunk is also a string, then merge them naively
        if isinstance(second_content, str):
            return first_content + second_content
        # If the second chunk is a list, add the first chunk to the start of the list
        else:
            return_list: List[Union[str, Dict]] = [first_content]
            return return_list + second_content
    # If both are lists, merge them naively
    elif isinstance(second_content, List):
        return first_content + second_content
    # If the first content is a list, and the second content is a string
    else:
        # If the last element of the first content is a string
        # Add the second content to the last element
        if isinstance(first_content[-1], str):
            return first_content[:-1] + [first_content[-1] + second_content]
        else:
            # Otherwise, add the second content as a new element of the list
            return first_content + [second_content]


class BaseMessageChunk(BaseMessage):
    """A Message chunk, which can be concatenated with other Message chunks."""

    def _merge_kwargs_dict(  # noqa: C901
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge additional_kwargs from another BaseMessageChunk into this one,
        handling specific scenarios where a key exists in both dictionaries
        but has a value of None in 'left'. In such cases, the method uses the
        value from 'right' for that key in the merged dictionary.
        Example:
        If left = {"function_call": {"arguments": None}} and
        right = {"function_call": {"arguments": "{\n"}}
        then, after merging, for the key "function_call",
        the value from 'right' is used,
        resulting in merged = {"function_call": {"arguments": "{\n"}}.
        """
        merged = left.copy()
        for k, v in right.items():
            if k not in merged:
                merged[k] = v
            elif merged[k] is None and v:
                merged[k] = v
            elif v is None:
                continue
            elif merged[k] == v:
                continue
            elif not isinstance(merged[k], type(v)):
                raise TypeError(
                    f'additional_kwargs["{k}"] already exists in this message,'
                    " but with a different type."
                )
            elif isinstance(merged[k], str):
                merged[k] += v
            elif isinstance(merged[k], dict):
                merged[k] = self._merge_kwargs_dict(merged[k], v)
            elif isinstance(merged[k], list):
                merged[k] = merged[k].copy()
                for i, e in enumerate(v):
                    if isinstance(e, dict) and isinstance(e.get("index"), int):
                        i = e["index"]
                    if i < len(merged[k]):
                        merged[k][i] = self._merge_kwargs_dict(merged[k][i], e)
                    else:
                        merged[k] = merged[k] + [e]
            else:
                raise TypeError(
                    f"Additional kwargs key {k} already exists in this message."
                )
        return merged

    def __add__(self, other: Any):  # type: ignore
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # concat into a single BaseMessageChunk

            return self.__class__(
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """A Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "AIMessageChunk"

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            return self.__class__(
                example=self.example,
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """A Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "HumanMessageChunk"


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """A System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "SystemMessageChunk"


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """A Tool Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ToolMessageChunk"

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                raise ValueError(
                    "Cannot concatenate ToolMessageChunks with different names."
                )

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """A Function Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "FunctionMessageChunk"

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                raise ValueError(
                    "Cannot concatenate FunctionMessageChunks with different names."
                )

            return self.__class__(
                name=self.name,
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """A Chat Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ChatMessageChunk"

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                raise ValueError(
                    "Cannot concatenate ChatMessageChunks with different roles."
                )

            return self.__class__(
                role=self.role,
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        elif isinstance(other, BaseMessageChunk):
            return self.__class__(
                role=self.role,
                content=_merge_content(self.content, other.content),
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        else:
            return super().__add__(other)


class Generation(BaseModel):
    """A single text generation output."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw response from the provider. May include things like the
        reason for finishing or token log probabilities.
    """
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "Generation"
    # TODO: add log probs as separate attribute


class ChatGeneration(Generation):
    """A single chat generation output."""

    text: str = ""
    """*SHOULD NOT BE SET DIRECTLY* The text contents of the output message."""
    message: BaseMessage
    """The message output by the chat model."""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ChatGeneration"

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Set the text attribute to be the contents of the message."""
        try:
            values["text"] = values["message"].content
        except (KeyError, AttributeError) as e:
            raise ValueError("Error while initializing ChatGeneration") from e
        return values


class GenerationChunk(Generation):
    """A Generation chunk, which can be concatenated with other Generation chunks."""

    def __add__(self, other: Any):
        if isinstance(other, GenerationChunk):
            generation_info = (
                {**(self.generation_info or {}), **(other.generation_info or {})}
                if self.generation_info is not None or other.generation_info is not None
                else None
            )
            return GenerationChunk(
                text=self.text + other.text,
                generation_info=generation_info,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )


class ChatGenerationChunk(ChatGeneration):
    """A ChatGeneration chunk, which can be concatenated with other
      ChatGeneration chunks.

    Attributes:
        message: The message chunk output by the chat model.
    """

    message: BaseMessageChunk
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    
    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ChatGenerationChunk"
    
    def __add__(self, other: Any):
        if isinstance(other, ChatGenerationChunk):
            generation_info = (
                {**(self.generation_info or {}), **(other.generation_info or {})}
                if self.generation_info is not None or other.generation_info is not None
                else None
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info,
            )
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
            )


class ChatResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChatGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class BaseOutputParser(BaseModel, ABC):
    """Class to parse the output of an LLM call.

    Output parsers help structure language model responses.
    """

    @abstractmethod
    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call.

        A method which takes in a string (assumed output of language model )
        and parses it into some structure.

        Args:
            text: output of language model

        Returns:
            structured output
        """

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Optional method to parse the output of an LLM call with a prompt.

        The prompt is largely provided in the event the OutputParser wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: output of language model
            prompt: prompt value

        Returns:
            structured output
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Union[Generation, ChatGeneration]]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""
