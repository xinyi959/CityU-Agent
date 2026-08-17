from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

import os

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VECTOR_PATH = os.path.join(
    BASE_DIR,
    "vectorstore"
)


embedding = HuggingFaceEmbeddings(

    model_name="BAAI/bge-large-en-v1.5"

)



vectorstore = Chroma(

    persist_directory=VECTOR_PATH,

    embedding_function=embedding

)



retriever = vectorstore.as_retriever(

    search_kwargs={
        "k":5
    }

)




def search_programmes(query:str):

    print("\n===== RETRIEVER QUERY =====")
    print(query)

    # docs = retriever.invoke(
    #     query
    # )

    docs = vectorstore.similarity_search_with_score(
        query,
        k=5
    )

    print(
        f"Retrieved docs: {len(docs)}"
    )

    results=[]


    for doc, score in docs:


        print("\n--- CHUNK ---")

        print(
            "Score:",
            score
        )

        print(
            doc.metadata
        )


        print(
            doc.page_content[:500]
        )


        results.append(
            f"""
Programme ID:
{doc.metadata.get("programme_id")}

Content:
{doc.page_content}
"""
        )


    return "\n".join(results)