from typing import List
from functools import lru_cache

# --------------------------------------------
# LangChain / LangGraph
# --------------------------------------------
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Vector Store
from langchain_qdrant import QdrantVectorStore

# LLMs
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangGraph 1.0+
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
# Config
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

#livekit
from livekit import agents
from livekit.plugins import aws, deepgram, langchain, silero
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)

from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("INFO")

# ============================================
# 1. Load Environment Configuration
# ============================================

load_dotenv("./.env", override=True)

class APIConfig(BaseSettings):

    DEEPGRAM_API_KEY: str

    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    GEMINI_API_KEY: str
    GEMINI_MODEL_NAME: str
    GEMINI_EMBEDDING_MODEL_NAME: str

    TWILIO_SIP_DOMAIN: str
    TWILIO_PHONE_NUMBER: str
    TWILIO_SIP_USERNAME: str
    TWILIO_SIP_PASSWORD: str

    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str
    LIVEKIT_URL: str

    QDRANT_URL: str
    QDRANT_API_KEY: str


config = APIConfig()

# ============================================
# 2. Ingest PDF + Initialize Chroma VectorStore
# ============================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=config.GEMINI_EMBEDDING_MODEL_NAME,
    google_api_key=config.GEMINI_API_KEY,
)

vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="financial_products_collection",
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )

# ============================================
# 3. Graph State Definition
# ============================================

class AgentState(MessagesState):
    standalone_question: str
    documents: List[Document]

# ============================================
# 4. Node Definitions
# ============================================

def contextualize_node(state: AgentState):
    print("\n--- Contextualizing Question ---")

    messages = state.get("messages", [])
    
    # Guard: Gemini requires at least one non-system message (HumanMessage/AIMessage).
    # If only system messages or no content is present, prepend a minimal HumanMessage.
    message = messages[-1] if messages else None
    history = messages[:-1] if len(messages) > 1 else []

    if len(messages)==1:
        return {"standalone_question": messages[0].content}
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            Given a chat history and the latest user question which might reference context in the chat history,
            formulate a standalone question which can be understood without the chat history.
            DO NOT answer the question, just reformulate it if needed and otherwise return it as is.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),

    ])

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0,
        streaming=False,
    )

    chain = prompt | llm | StrOutputParser()

    reformulated = chain.invoke({
        "chat_history": history,
        "question": message.content if message else "",
    })

    return {"standalone_question": reformulated}


async def retrieve_node(state: AgentState):
    print("\n--- Retrieving Documents ---")
    question = state["standalone_question"]
    docs = await vector_store.asimilarity_search(
        query=question,
        k=4,
    )
    logger.info("---------------------------------------------------------")
    logger.info(f"Retrieved {len(docs)} documents for question: {question}")
    logger.info("---------------------------------------------------------")


    return {"documents": docs}  


def answer_node(state: AgentState):
    print("\n--- Generating Answer ---")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            You're a friendly financial assistant helping customers over the phone.

            Answer using the context below. Keep it natural and brief.

            VOICE RULES:
            - Max 2-3 short sentences per response
            - Talk like a real person, not a robot
            - No bullet points or lists - speak naturally
            - Use simple words - avoid technical terms
            - If unsure, just say "I don't have that info right now"

            Context:
            {context}
            """
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.1,
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": "\n\n".join([doc.page_content for doc in state["documents"]]),
        "question": state["standalone_question"]
    })

    return {"messages": response}

# ============================================
# 5. Build Workflow Graph (Latest LangGraph)
# ============================================

def create_graph():

    workflow = StateGraph(state_schema=AgentState)
    workflow.add_node("contextualize", contextualize_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", answer_node)

    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

# ============================================
# Livekit Agent
# ============================================

class QualificationAgent(Agent):

    def __init__(self, runnable_graph):
        self.graph = runnable_graph

        super().__init__(
            llm=langchain.LLMAdapter(
                graph=self.graph,

            ),
            instructions="No instructions provided.",
        )

# ============================================
# Entry Point
# ============================================

async def entrypoint(ctx: agents.JobContext):
    """
    Entry point for the LiveKit Voice Agent job.
    """

    # Initialize Deepgram ASR
    stt = deepgram.STT(
        api_key=config.DEEPGRAM_API_KEY
    )

    # Initialize AWS Polly TTS
    tts = aws.TTS(
        voice="Joanna",
        region=config.AWS_DEFAULT_REGION,
        api_key=config.AWS_ACCESS_KEY_ID,
        api_secret=config.AWS_SECRET_ACCESS_KEY
    )

    # Initialize Silero VAD
    vad = silero.VAD.load()

    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    graph = create_graph()

    session = AgentSession(
        stt=stt,
        tts=tts,
    )

    agent = QualificationAgent(runnable_graph=graph)

    await session.start(room=ctx.room, agent=agent)

    greeting = """
    Hi! This is your financial assistant calling to help you with any questions about our financial products and services. How can I assist you today?
    """

    await session.say(text=greeting)

    logger.info("\nQualification call started.\n")

if __name__ == "__main__":
    # Pass hardcoded credentials here
    cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint,
                             api_key=config.LIVEKIT_API_KEY, 
                             api_secret=config.LIVEKIT_API_SECRET, 
                             ws_url=config.LIVEKIT_URL),
    )
        
    
