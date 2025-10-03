from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 加载文档
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块的大小
    chunk_overlap=50,  # 文本块之间的重叠部分
)
splits = text_splitter.split_documents(documents)

# 3. 向量化并存储
embeddings = OpenAIEmbeddings(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3-Embedding-0.6B",
)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化存储路径
)

print(f"成功将 {len(splits)} 个文本块存入向量数据库")
