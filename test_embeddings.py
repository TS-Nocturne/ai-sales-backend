from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ai_sales.config.llm import resolve_api_key

api_key = resolve_api_key()
models = ['models/text-embedding-004', 'models/gemini-embedding-001', 'models/embedding-001']

for m in models:
    try:
        emb = GoogleGenerativeAIEmbeddings(model=m, google_api_key=api_key)
        res = emb.embed_query('test')
        print(f'{m} works! Dimension: {len(res)}')
    except Exception as e:
        print(f'{m} failed: {e}')
