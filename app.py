import streamlit as st
import logging
import os
import tempfile
import shutil

from ollama import Client

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# Streamlit页面配置
st.set_page_config(
    page_title="Pdf问答示例",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    从ollama中获取模型名称

    参数:
        models_info (Dict[str, List[Dict[str, Any]]]): 字典中包含ollama返回的模型信息

    返回:
        Tuple[str, ...]: 模型名称的元组.
    """
    logger.info("从Ollama中获取模型名称")
    model_names = tuple(model["name"] for model in models_info["models"])
    embedding_model = os.getenv('embedding_model')
    model_names = [model for model in model_names if model!= embedding_model]
    logger.info(f"获取的模型名称为: {model_names}")
    return model_names


def create_vector_db(file_upload,provider) -> QdrantVectorStore:
    """
    根据上传的PDF文件创建向量数据库

    参数:
        file_upload (st.UploadedFile): 使用Streamlit文件上传组件.

    返回:
        QdrantVectorStore: 构建完成的向量数据库.
    """
    logger.info(f"正在上传文件: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"临时文件存储路径: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("文档分块完成")

    if provider == "云":
        embeddings = OpenAIEmbeddings(base_url=os.getenv('openai_url'))
    else:
        embeddings = OllamaEmbeddings(base_url=os.getenv('ollama_url'),model=os.getenv('embedding_model'), show_progress=True)
    
    vector_db = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url= os.getenv('qdrant_url'),
        prefer_grpc=True,
        force_recreate=True,
        collection_name="pdfchat",
    )
    logger.info("向量数据库构建完成")

    shutil.rmtree(temp_dir)
    logger.info(f"清除临时文件目录 {temp_dir}")
    return vector_db


def rag_process(question: str, vector_db: QdrantVectorStore, selected_model: str,temperature:float,provider: str) -> str:
    """
    基于向量检索的RAG过程

    参数:
        question (str): 用户输入的问题.
        vector_db (QdrantVectorStore): 向量数据库.
        selected_model (str): 选择的LLM.

    返回:
        str: RAG生成的答案.
    """
    logger.info(f"""用户的问题: {question} 选择的模型: {selected_model}""")
    if provider == "云":
        llm = ChatOpenAI(base_url=os.getenv('openai_url'),model=selected_model,temperature=temperature);
    else:
        llm = ChatOllama(base_url=os.getenv('ollama_url'),model=selected_model, temperature=temperature)
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""你是一名AI语言模型助理。您的任务是生成给定用户问题的3个不同版本中文问题，以从向量数据库中检索相关文档。通过对用户问题生成多个视角的问题，您的目标是帮助用户克服基于距离的相似性搜索的一些局限性。请提供这些用换行符分隔的备选问题。
        原始问题: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """仅根据以下上下文使用中文言简意赅回答问题：
    {context}
    问题: {question}
    如果你不知道答案，就说你不知道，不要试图编造答案。
    只提供{context}中的答案，其他什么都不提供。
    添加您用来回答问题的上下文片段。
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("LLM 生成完成")
    return response


def delete_vector_db(vector_db: Optional[QdrantVectorStore]) -> None:
    """
    删除向量数据库并清除相关会话状态.

    参数:
        vector_db (Optional[QdrantVectorStore]): 要删除的向量数据库.
    """
    logger.info("删除向量数据库")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("数据库表清理完成")
        logger.info("数据库表清理完成")
        st.rerun()
    else:
        st.error("没有找到可删除的向量数据库.")
        logger.warning("没有找到可删除的向量数据库.")


def main() -> None:
    """
    主程序，负责页面布局和交互逻辑
    """
    st.title("🦜🔗 Pdf问答", anchor=False)
    user_avator = "🧑‍💻"
    robot_avator = "🤖"

    
    col1, col2 = st.columns([3, 5])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None


    provider = col1.radio("模型服务", ["云","本地"],horizontal=True)
    if provider == '本地':
        try:
             client = Client(host=os.getenv('ollama_url'))
             models_info = client.list()
             available_models = extract_model_names(models_info)
             selected_model = col1.selectbox("模型选择", available_models)
        except Exception as e:
            print(e)
            col1.error("请检查ollama服务是否运行正确") 
    else:
        available_models = ('gpt-3.5-turbo', 'gpt-4-turbo','gpt-4o')
        selected_model = col1.selectbox("模型选择", available_models)
        api_key = col1.text_input("OpenAI_API_Key", "", type="password")


    temperature = col1.slider('温度(Temperature)', 0.0, 1.0, 0.0, step=0.1)
    file_upload = col1.file_uploader(
        "上传待问答的Pdf", type="pdf", accept_multiple_files=False
    )
    status_placeholder = col1.empty()

    if provider == "云" and (api_key is not None and api_key.startswith("sk-") and st.session_state.get('api_key') is None):
        os.environ["OPENAI_API_KEY"] = api_key
        status_placeholder.info("key设置完成")
        st.session_state["api_key"] = True
        
    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            status_placeholder.info("知识库构建中，请稍后...")
            st.session_state["vector_db"] = create_vector_db(file_upload,provider)
            status_placeholder.info("知识库就绪，可以提问了!")

    delete_collection = col1.button("删除连接", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = robot_avator if message["role"] == "assistant" else user_avator
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("输入你的问题..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar=user_avator).markdown(prompt)

                with message_container.chat_message("assistant", avatar=robot_avator):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None :
                            response = rag_process(
                                prompt, st.session_state["vector_db"], selected_model,temperature,provider
                            )
                            st.markdown(response)
                        else:
                            st.warning("你还没有上传文档呢!")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e)
                logger.error(f"Rag执行时发生错误: {e}")
        else:
            if provider == "云" and st.session_state.get('api_key') is None:
                status_placeholder.warning("还未设置OpenAI API key...")
            elif st.session_state["vector_db"] is None:
                status_placeholder.warning("文档还未上传...")



if __name__ == "__main__":
    main()
