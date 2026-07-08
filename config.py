import os
from dotenv import load_dotenv

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not WATSONX_API_KEY:
    raise EnvironmentError("WATSONX_API_KEY is not set. Please copy .env.example to .env and fill in your credentials.")
if not WATSONX_PROJECT_ID:
    raise EnvironmentError("WATSONX_PROJECT_ID is not set. Please copy .env.example to .env and fill in your credentials.")
