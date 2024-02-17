

from typing import List
from aido_client.schemas.schema import BaseMessage, PromptValue, get_buffer_string


class ChatPromptValue(PromptValue):
    messages: List[BaseMessage]

    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.messages
