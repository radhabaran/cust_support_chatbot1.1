# router_agent.py
from typing import Dict
import logging
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

anthro_api_key = os.environ['ANTHRO_KEY']           
os.environ['ANTHROPIC_API_KEY'] = anthro_api_key

# Initialize LLM
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm = ChatAnthropic(model="claude-3-haiku-20240307")


class RouterResponse:
    PRODUCT_REVIEW = "product_review"
    GENERIC = "generic"
    ORDER_TRACKING = "order_tracking"


def extract_current_message(state: Dict) -> str:
    """Extract the current message from state"""
    if "current_message" in state:
        return state["current_message"]
    
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            return last_message.content
    return ""


def planning_route_query(state: Dict, config: Dict) -> Dict:
    """Route the query based on content analysis"""
    try:
        current_message = extract_current_message(state)
        messages = state.get("messages", [])
        history = messages[:-1] if len(messages) > 1 else []

        prompt = f"""Analyze the following query and determine if it's related to product review or a generic query. Whenever
        user asks the availability of any product, understand that user is asking about the product in 
        Amazon product catalogue. Infer Amazon product catalogue even if nothing is mentioned about the
        source of the product.
        
        Product Review queries include:
        - Questions about product features, specifications, or capabilities
        - Question on any product model
        - Product prices and availability inquiries
        - Requests for product reviews or comparisons
        - Product warranty or guarantee questions
        - Product compatibility or dimension questions
        - Product recommendations
        
        Order Tracking queries include:
        - Order tracking inquiries
        - Order replacement request
        - Order refund request
        - Order cancellation request

        Generic queries include:
        - Customer happiness message
        - Customer grievance
        - Customer service inquiries
        - Account-related questions
        - Technical support issues
        - Website navigation help
        - Payment or billing queries
        - Return policy questions
        - Company information requests
        
        Chat History:
        {history}

        Current Query:
        {current_message}
        
        Return ONLY 'product_review' or 'order_tracking' or 'generic' as response."""
        
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages, config).content.lower().strip()
        
        # category = RouterResponse.PRODUCT_REVIEW if RouterResponse.PRODUCT_REVIEW in response else RouterResponse.GENERIC
        if RouterResponse.PRODUCT_REVIEW in response:
            category = RouterResponse.PRODUCT_REVIEW
        elif RouterResponse.ORDER_TRACKING in response:
            category = RouterResponse.ORDER_TRACKING
        else:
            category = RouterResponse.GENERIC

        # Log the routing decision
        logger.info(f"Routed message: '{current_message[:50]}...' to category: {category}")

        return {
            "router_response": category,
            "routing_metadata": {
                "routing_category": category,
                "original_message": current_message[:100]  # First 100 chars for context
            }
        }
    
    except Exception as e:
        logger.error(f"Error in planning_route_query: {e}")
        return {
            "router_response": RouterResponse.GENERIC,
            "routing_metadata": {
                "routing_category": RouterResponse.GENERIC,
                "error": str(e)
            }
        }