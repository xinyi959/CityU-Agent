from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


model = ChatOpenRouter(
    model="stealth/ox-alpha",
    temperature=0,
)