from pydantic import BaseModel
from typing import List, Optional, Dict

class Message(BaseModel):
    content: str
    role: str = "user"
    metadata: Optional[Dict] = None

class AgentResponse(BaseModel):
    response: str
    confidence: float
    metadata: Optional[Dict] = None
