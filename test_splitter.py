from langchain_text_splitters import MarkdownHeaderTextSplitter


with open(
    "data/markdown/p66.md",
    encoding="utf-8"
) as f:
    md = f.read()


headers = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]


splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers
)


docs = splitter.split_text(md)


print(
    "chunks:",
    len(docs)
)


for d in docs[:]:

    print("="*50)

    print(d.page_content[:])

    print(d.metadata)