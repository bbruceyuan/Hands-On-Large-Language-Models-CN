from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# 1. 加载已有的向量数据库
embeddings = OpenAIEmbeddings(
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3-Embedding-0.6B",
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. 用户提问
query = "什么是RAG？"

# 3. 检索相关文档（返回最相关的 3 个）
docs = vectorstore.similarity_search(query, k=3)

# 4. 将检索到的文档内容拼接成上下文
context = "\n\n".join([doc.page_content for doc in docs])

# 5. 构建 Prompt 模板
prompt_template = """
你是一个专业的问答助手。请根据以下参考文档回答用户的问题。
如果参考文档中没有相关信息，请诚实地说不知道，不要编造答案。

参考文档：
{context}

用户问题：{question}

回答：
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

# 6. 创建 LLM 并生成回答
llm = ChatOpenAI(
    model="THUDM/glm-4-9b-chat",
    temperature=0,
    max_retries=3,
    base_url="https://api.siliconflow.cn/v1",
)
final_prompt = prompt.format(context=context, question=query)

print(f"最终的 Prompt 内容：{final_prompt}")

messages = [HumanMessage(content=final_prompt)]
response = llm.invoke(messages)

# 7. 输出结果
print(f"问题: {query}")
print(f"回答: {response.content}")
print(f"\n参考文档数量: {len(docs)}")
