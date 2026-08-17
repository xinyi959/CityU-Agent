import os

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings



DATA_PATH = "../data/markdown"

VECTOR_PATH = "./vectorstore"



# ==========================
# 1. Load markdown files
# ==========================


documents = []


for filename in os.listdir(DATA_PATH):

    if filename.endswith(".md"):

        filepath = os.path.join(
            DATA_PATH,
            filename
        )


        loader = TextLoader(
            filepath,
            encoding="utf-8"
        )


        docs = loader.load()


        # 添加metadata

        for doc in docs:

            doc.metadata["programme_id"] = (
                filename.replace(".md","").upper()
            )


        documents.extend(docs)



print(
    f"Loaded {len(documents)} markdown files"
)



# ==========================
# 2. Split documents
# ==========================


splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)


chunks = splitter.split_documents(
    documents
)


print(
    f"Created {len(chunks)} chunks"
)



# ==========================
# 3. Embedding model
# ==========================


embedding = HuggingFaceEmbeddings(

    model_name=
    "BAAI/bge-large-en-v1.5"

)



# ==========================
# 4. Create Chroma
# ==========================


vectorstore = Chroma.from_documents(

    documents=chunks,

    embedding=embedding,

    persist_directory=VECTOR_PATH

)



print(
    "Vector database created!"
)