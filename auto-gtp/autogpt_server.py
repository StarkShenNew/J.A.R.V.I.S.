import sys
import os
import gradio as gr
from utils import ArgumentParser, LOG


import random
import time

import os
os.environ["SERPAPI_API_KEY"] = "e65622355785aba531fe0f3733c6c429e3ec43457c916a0c3006e6f81d433369"
os.environ["OPENAI_API_KEY"]= "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"

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.embeddings import OpenAIEmbeddings

import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

#pip install google-search-results
#pip install faiss-cpu

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def translation(input_file, source_language, target_language):
    LOG.debug(f"[翻译任务]\n源文件: {input_file.name}\n源语言: {source_language}\n目标语言: {target_language}")

    output_file_path = Translator.translate_pdf(
        input_file.name, source_language=source_language, target_language=target_language)

    return output_file_path

def launch_gradio():

    iface = gr.Interface(
        fn=translation,
        title="OpenAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese")
        ],
        outputs=[
            gr.File(label="下载翻译文件")
        ],
        allow_flagging="never"
    )

    iface.launch(share=True, server_name="0.0.0.0")

def initialize_translator():
    # 解析命令行
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()

    # 初始化配置单例
    config = TranslationConfig()
    config.initialize(args)    
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    global Translator
    Translator = PDFTranslator(config.model_name)


if __name__ == "__main__":
    # 初始化 translator
#    initialize_translator()
    # 启动 Gradio 服务
    # launch_gradio()

    # 构造 AutoGPT 的工具集
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    embeddings_model = OpenAIEmbeddings()
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0),
        memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
    )
    # 打印 Auto-GPT 内部的 chain 日志
    agent.chain.verbose = True

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            
            bot_message = agent.run(["2023年大运会举办地在哪？"]) #random.choice(["How are you?", "I love you", "I'm very hungry"])
            chat_history.append((message, bot_message))
            time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.launch(share=True, server_name="0.0.0.0")


    




