from aido_client.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from aido_client.callbacks.shared import SharedCallbackManager


def get_callback_manager() -> BaseCallbackManager:
    """Return the shared callback manager."""
    return SharedCallbackManager()


def set_handler(handler: BaseCallbackHandler) -> None:
    """Set handler."""
    callback = get_callback_manager()
    callback.set_handler(handler)
