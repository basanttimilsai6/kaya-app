from pydantic_settings import BaseSettings  # ✅ instead of `from pydantic import BaseSettings`

class Settings(BaseSettings):
    app_name: str = "ImageAPI"
    app_version: str = "1.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
