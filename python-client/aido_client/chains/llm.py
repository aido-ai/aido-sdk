from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Extra
from aido_client.callbacks.base import BaseCallbackManager
from aido_client.chains.base import Chain
from aido_client.llms.base import BaseExtLanguageModel
from aido_client.prompts.base import BasePromptTemplate
from aido_client.prompts.prompt import PromptTemplate
from aido_client.schemas.schema import LLMResult, PromptValue
from aido_client.utils.input import get_colored_text


class LLMChain(Chain, BaseModel):
    """Chain to run queries against LLMs.

    Example:
        .. code-block:: python

            from aido.core.chains.llm import LLMChain
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseExtLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> Dict[str, str]:
        return self.apply([inputs])[0]

    def generate(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list)
        return self.llm.generate_prompt(prompts, stop, streaming_channel=streaming_channel)

    async def agenerate(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = await self.aprep_prompts(input_list)
        return await self.llm.agenerate_prompt(prompts, stop, streaming_channel=streaming_channel)

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    async def aprep_prompts(
        self, input_list: List[Dict[str, Any]]
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        for inputs in input_list:
            selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if self.callback_manager.is_async:
                await self.callback_manager.on_text(
                    _text, end="\n", verbose=self.verbose
                )
            else:
                self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(prompt)
        return prompts, stop

    def apply(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = self.generate(input_list, streaming_channel=streaming_channel)
        return self.create_outputs(response)

    async def aapply(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> List[Dict[str, str]]:
        """Utilize the LLM generate method for speed gains."""
        response = await self.agenerate(input_list, streaming_channel=streaming_channel)
        return self.create_outputs(response)

    def create_outputs(self, response: LLMResult) -> List[Dict[str, str]]:
        """Create outputs from response."""
        return [
            # Get the text of the top generated string.
            {self.output_key: generation[0].text}
            for generation in response.generations
        ]

    async def _acall(
        self,
        inputs: Dict[str, Any],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> Dict[str, str]:
        return (await self.aapply([inputs], streaming_channel=streaming_channel))[0]

    def predict(
        self,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return self(kwargs, streaming_channel=streaming_channel)[self.output_key]

    async def apredict(
        self,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> str:
        """Format prompt with kwargs and pass to LLM.

        Args:
            **kwargs: Keys to pass to prompt template.

        Returns:
            Completion from LLM.

        Example:
            .. code-block:: python

                completion = llm.predict(adjective="funny")
        """
        return (await self.acall(kwargs, streaming_channel=streaming_channel))[self.output_key]

    def predict_and_parse(
        self,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> Union[str, List[str], Dict[str, str]]:
        """Call predict and then parse the results."""
        result = self.predict(**kwargs, streaming_channel=streaming_channel)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    async def apredict_and_parse(
        self,
        streaming_channel: Optional[BaseCallbackManager] = None,
        **kwargs: Any
    ) -> Union[str, List[str], Dict[str, str]]:
        """Call apredict and then parse the results."""
        result = await self.apredict(**kwargs, streaming_channel=streaming_channel)
        if self.prompt.output_parser is not None:
            return self.prompt.output_parser.parse(result)
        else:
            return result

    def apply_and_parse(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = self.apply(input_list, streaming_channel=streaming_channel)
        return self._parse_result(result)

    def _parse_result(
        self, result: List[Dict[str, str]]
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        if self.prompt.output_parser is not None:
            return [
                self.prompt.output_parser.parse(res[self.output_key]) for res in result
            ]
        else:
            return result

    async def aapply_and_parse(
        self,
        input_list: List[Dict[str, Any]],
        streaming_channel: Optional[BaseCallbackManager] = None
    ) -> Sequence[Union[str, List[str], Dict[str, str]]]:
        """Call apply and then parse the results."""
        result = await self.aapply(input_list, streaming_channel=streaming_channel)
        return self._parse_result(result)

    @property
    def _chain_type(self) -> str:
        return "llm_chain"

    @classmethod
    def from_string(cls, llm: BaseExtLanguageModel, template: str) -> Chain:
        """Create LLMChain from LLM and template."""
        prompt_template = PromptTemplate.from_template(template)
        return cls(llm=llm, prompt=prompt_template)
