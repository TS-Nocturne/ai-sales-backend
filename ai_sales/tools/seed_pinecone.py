"""
Seed the Pinecone vector index with the product catalog and FAQ.

Run once (or whenever the catalog/FAQ changes) to upsert embeddings
into Pinecone so that `search_knowledge_base` can perform semantic search.

Usage:
    poetry run python -m ai_sales.tools.seed_pinecone
"""

import os
import time
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

from ai_sales.config.vectorstore import get_embeddings, get_pinecone_index
from ai_sales.tools.catalog import get_product_catalog

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FAQ_PATH = os.path.join(_BASE_DIR, "documents", "CustomerFAQ.pdf")
_TECH_STANDARDS_PATH = os.path.join(_BASE_DIR, "documents", "technical_standards.txt")


def _product_to_text(product: dict) -> str:
    """Convert a product dict into a text blob for embedding."""
    return (
        f"Product Name: {product['name']} | "
        f"Category: {product['category']} | "
        f"Price: {product['price']} THB | "
        f"Description: {product['description']} | "
        f"Warranty: {product['warranty_period']}"
    )


def seed():
    """Embed and upsert the products and FAQs into Pinecone."""
    print("=" * 60)
    print("  Pinecone Knowledge Base Seeder (Mobile Accessories)")
    print("=" * 60)

    embeddings = get_embeddings()
    index = get_pinecone_index()

    texts = []
    ids = []
    metadatas = []

    # 1. Process Products
    catalog = get_product_catalog()
    print(f"  Loaded {len(catalog)} products from CSV.")
    
    for product in catalog:
        text = _product_to_text(product)
        texts.append(text)
        ids.append(f"prod_{product['id']}")
        metadatas.append({
            "source_type": "product",
            "id": product["id"],
            "name": product["name"],
            "price": product["price"],
            "category": product["category"],
            "description": product["description"],
            "stock": product["stock"],
            "warranty": product["warranty_period"],
            "text": text,
        })

    # 2. Process FAQ PDF
    print(f"  Loading FAQ from {_FAQ_PATH}...")
    try:
        loader = PyPDFLoader(_FAQ_PATH)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        faq_chunks = text_splitter.split_documents(documents)
        print(f"  Split FAQ into {len(faq_chunks)} chunks.")
        
        for i, chunk in enumerate(faq_chunks):
            texts.append(chunk.page_content)
            ids.append(f"faq_{uuid4().hex[:8]}")
            metadatas.append({
                "source_type": "faq",
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "CustomerFAQ.pdf"),
                "page": chunk.metadata.get("page", 0)
            })
    except Exception as e:
        print(f"  Warning: Failed to process FAQ PDF: {e}")

    # 3. Technical standards reference (IP, Bluetooth, etc.)
    print(f"  Loading technical reference from {_TECH_STANDARDS_PATH}...")
    try:
        with open(_TECH_STANDARDS_PATH, encoding="utf-8") as fh:
            tech_text = fh.read().strip()
        if tech_text:
            tech_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            tech_chunks = tech_splitter.split_text(tech_text)
            print(f"  Split technical reference into {len(tech_chunks)} chunks.")
            for i, chunk in enumerate(tech_chunks):
                texts.append(chunk)
                ids.append(f"tech_{uuid4().hex[:8]}")
                metadatas.append({
                    "source_type": "knowledge",
                    "text": chunk,
                    "title": "Technical Standards Reference",
                    "source": "technical_standards.txt",
                    "page": i,
                })
    except Exception as e:
        print(f"  Warning: Failed to process technical standards: {e}")

    if not texts:
        print("  No data to seed. Exiting.")
        return

    # Upsert in batches to avoid payload limits
    BATCH_SIZE = 100
    print(f"\n  Generating embeddings and upserting {len(texts)} vectors in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_metadatas = metadatas[i:i+BATCH_SIZE]
        
        vectors = embeddings.embed_documents(batch_texts)
        
        records = []
        for vec_id, vector, metadata in zip(batch_ids, vectors, batch_metadatas):
            records.append({
                "id": vec_id,
                "values": vector,
                "metadata": metadata,
            })
            
        index.upsert(vectors=records)
        print(f"    Upserted batch {i//BATCH_SIZE + 1} ({len(records)} items)")

    time.sleep(2)

    stats = index.describe_index_stats()
    print(f"\n  Index stats after upsert:")
    print(f"    Total vectors: {stats.get('total_vector_count', 'N/A')}")
    print(f"    Dimension:     {stats.get('dimension', 'N/A')}")

    print("\n  Seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    seed()
