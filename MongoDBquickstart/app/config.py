from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    database_hostname: str
    database_password: str
    database_name: str
    database_username: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    cloudinary_cloud_name: str
    cloudinary_api_key: str
    cloudinary_api_secret: str

    class Config:
        env_file = os.path.join(os.getcwd(), '.env')

settings = Settings()