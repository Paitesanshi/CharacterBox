from pydantic import BaseModel
from typing import Any, Dict, Optional, Tuple
class Message(BaseModel):
    """
    Message class for communication between backend and frontend.
    """
    content: str
    agent_id: int
    action: str

    @classmethod
    def from_dict(cls, message_dict):
        return cls(
            message_dict["agent_id"], message_dict["action"], message_dict["content"]
        )
    