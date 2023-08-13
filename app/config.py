import os
from pydantic import Field
from pydantic_settings import BaseSettings

os.environ['CQLENG_ALLOW_SCHEMA_MANAGEMENT'] = '1'

class Settings(BaseSettings):
    astra_db_client_id: str = Field(..., env="ASTRA_DB_CLIENT_ID")
    astra_db_client_secret: str = Field(..., env="ASTRA_DB_CLIENT_SECRET")
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    bucket_name: str = Field(..., env="BUCKET_NAME")
    endpoint_url: str = Field(..., env="ENDPOINT_URL")
    region_name: str = Field(..., env="REGION_NAME")

    class Config:
        env_file = '.env'



def get_settings():
    return Settings()