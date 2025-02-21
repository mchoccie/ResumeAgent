from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    API_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
