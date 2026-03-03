from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main() -> None:
    persist_path = Path("./llamaindex_index_store")
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Index directory not found: {persist_path.resolve()}\n"
            "Please run 03_llamaindex_vector.py first."
        )

    # Keep the same embedding model used during indexing.
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
    index = load_index_from_storage(storage_context)

    # Pure similarity search: no LLM call, no OPENAI_API_KEY required.
    retriever = index.as_retriever(similarity_top_k=2)
    query = "LlamaIndex 是做什么的？"
    nodes = retriever.retrieve(query)

    print(f"Query: {query}")
    print("Top similar chunks:")
    for i, node in enumerate(nodes, start=1):
        score = node.score if node.score is not None else 0.0
        print(f"{i}. score={score:.4f}, text={node.node.get_content()}")


if __name__ == "__main__":
    main()
