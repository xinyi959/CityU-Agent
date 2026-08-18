from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()


model = ChatOpenRouter(
    model="deepseek/deepseek-v4-flash",
    temperature=0,
)