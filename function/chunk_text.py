from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(raw_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks