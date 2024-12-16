# product_review_agent.py
import os
import logging
from typing import Dict
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import shutil
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

anthro_api_key = os.environ['ANTHRO_KEY']           
os.environ['ANTHROPIC_API_KEY'] = anthro_api_key

api_key = os.environ['OA_API']           
os.environ['OPENAI_API_KEY'] = api_key

class ProductReviewAgent:
    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        self.llm = ChatAnthropic(model=model_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectorstore = None
        
        self.system_prompt = """
        Role and Capabilities:
        You are an AI customer service specialist for Amazon. You respond strictly based on the context provided and 
        from the previous chat history. Whenever user mentions Amazon, you refer strictly to local knowledge base. 
        Your primary functions are: 
        1. Providing accurate product information including cost, availability, features, top review or user rating. Treat top review, user rating, user feedback are all same request.
        2. Handling delivery-related queries
        3. Addressing product availability
        4. Offering technical support for electronics

        Core Instructions:
        1. Product Information:
           - Provide detailed specifications and features based only on the provided context.
           - Compare similar products when relevant only if they appear in the provided context.
           - Only discuss products found in the provided context.
           - Highlight key benefits and limitations found in the context.
           - Include top reviews or user ratings only if available in the context.

        2. Price & Availability:
           - Quote exact prices and stock availability directly from the provided context.
           - Explain any pricing variations or discounts only if stated in the context.
           - Provide clear stock availability information only if stated in the context.
           - Mention delivery timeframes only when available in the context.

        3. Query Handling:
           - Address the main query first, then provide additional relevant information from the context.
           - For multi-part questions, structure answers in bullet points
           - If information is missing from context, explicitly state this
           - Suggest alternatives when a product is unavailable

        Communication Guidelines:
        1. Response Structure:
           - Start with a direct answer to the query based solely on the provided context.
           - Provide supporting details and context from the provided information only.
           - End with a clear next step or call to action
           - Include standard closing: "Thank you for choosing Amazon. Is there anything else I can help you with?"

        2. Tone and Style:
           - Professional yet friendly
           - Clear and jargon-free language
           - Empathetic and patient
           - Concise but comprehensive

        Limitations and Restrictions:
        1. Provide information present only in the given context.
        2. Do not provide answers from memory; rely exclusively on the provided context.
        3. Clearly state when information is not available in the context.
        4. Never share personal or sensitive information
        5. Don't make promises about delivery times unless explicitly stated in context

        Error Handling:
        1. Out of Scope: "While I can't assist with [topic], I'd be happy to help you other products if you like."
        2. Technical Issues: "I apologize for any inconvenience. Could you please rephrase your question or provide more details?"

        Response Format:
        1. For product queries:
           - Product name and model
           - Price and availability
           - Key features
           - Top review or user rating
           - Comparison among similar products (example: cell phone with cell phone, not with cell phone accessories)
           - Recommendations if relevant

        2. For service queries:
           - Current status
           - Next steps
           - Timeline (if available)
           - Contact options

        Remember: Always verify information against the provided context or in the previous chat history before 
        responding. Don't make assumptions or provide speculative information.
        """
        self.initialize_vectorstore()

    def initialize_vectorstore(self, vectorstore_path: str = 'data/chroma/'):
        """Initialize vector store with product data"""
        try:
            file_path = 'data/cleaned_dataset_full.csv'
            dataframe = pd.read_csv(file_path)
            dataframe['combined'] = dataframe.apply(
                lambda row: ' '.join(f"{col}: {val}" for col, val in row.items()), 
                axis=1
            )
            
            documents = [Document(page_content=text) for text in dataframe['combined']]
            chunks = self._split_text(documents)
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
                self.vectorstore = Chroma(
                    persist_directory=vectorstore_path,
                    embedding_function=self.embeddings
                )
            else:
                shutil.rmtree(vectorstore_path, ignore_errors=True)
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=vectorstore_path
                )
                self.vectorstore.persist()
                
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise


    def _split_text(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len,
            add_start_index=True,
        )
        return splitter.split_documents(documents)


    def process_review_query(self, state: Dict, config: dict) -> Dict:
        """Process product review queries"""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            
            # Extract content properly based on message type
            if isinstance(last_message, (HumanMessage, AIMessage)):
                query = last_message.content
            elif isinstance(last_message, str):
                query = last_message
            else:
                query = str(last_message)

            thread_id = config["configurable"]["thread_id"]
            
            logger.info(f"Processing product_review query for thread {thread_id}")
            
            # Retrieve relevant documents
            retriever = self.vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={"k": 2, "fetch_k": 5}
            )
            results = retriever.invoke(query)
            
            if not results:
                response = "I apologize, but I couldn't find any relevant product information for your query."
                state["product_info"] = response
                state["messages"].append(AIMessage(content=response))
                return state
                
            context = "\n\n".join([doc.page_content for doc in results])
            
            # Format messages with system prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._format_review_prompt(query, context))
            ]

            response = self.llm.invoke(messages)

            if isinstance(response.content, str):
                response_content = response.content
            else:
                response_content = str(response.content)

            print('*** \nDebugging: response returned by llm in product_review agent : ', response_content)
            
            state["product_info"] = response_content
            state["messages"].append(AIMessage(content=response_content))
            
            # return {
            #     "review_response": response.content,
            #     "thread_id": thread_id
            # }
            return state
            
        except Exception as e:
            logger.error(f"Error processing review query: {e}")
            error_msg = "I apologize, but I encountered an error while processing your product request."
            state["product_info"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            return state


    def _format_review_prompt(self, query: str, context: str) -> str:
        """Format the prompt for review processing"""
        return f"""
        Context from product database:
        {context}
        
        User Query:
        {query}
        """

def setup_product_review_agent() -> ProductReviewAgent:
    """Setup and return the product review agent"""
    return ProductReviewAgent()

