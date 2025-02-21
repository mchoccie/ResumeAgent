from typing import List
from .models import Message, AgentResponse
from .config import Settings

class Agent:
    def __init__(self):
        self.settings = Settings()
        self.conversation_history: List[Message] = []

    async def process_message(self, message: Message) -> AgentResponse:
        self.conversation_history.append(message)
        
        # Add your agent logic here
        response = AgentResponse(
            response="Default response",
            confidence=1.0,
            metadata={"history_length": len(self.conversation_history)}
        )
        
        return response

    def clear_history(self) -> None:
        self.conversation_history = []
