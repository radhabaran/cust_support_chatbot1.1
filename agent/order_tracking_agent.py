# order_tracking_agent.py

import pandas as pd
import logging
import re
import os
import json
from typing import Dict, Optional, Union, List
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OrderTrackingError(Exception):
    """Base exception class for OrderTracking errors"""
    pass


class CSVFileError(OrderTrackingError):
    """Exception raised for CSV file related errors"""
    pass


class APIError(OrderTrackingError):
    """Exception raised for API related errors"""
    pass


class ValidationError(OrderTrackingError):
    """Exception raised for validation errors"""
    pass

csv_path = "data\order_track.csv"


# Query Analysis Prompt
QUERY_ANALYSIS_PROMPT = """
Analyze the user query about order tracking and extract the following information:
User Query: {user_query}

1. Query Type (return one of these exact values):
   - track_order
   - refund_request
   - replacement_request
   - general_inquiry

2. Order/Tracking Number (if present):
   Extract any order number or tracking number mentioned.

3. Specific Details:
   Any specific requirements or concerns mentioned in the query.

Return the response in this exact JSON format:
{
    "query_type": "<type>",
    "order_track_number": "<extracted_number_or_none>",
    "order_details": "<specific_details>"
}
"""

# Order Response Prompt
ORDER_RESPONSE_PROMPT = """
Given the order details and query type, generate a helpful response for the customer.

Order Details: {order_details}
Query Type: {query_type}
Action Taken: {action_taken}

Your response must include these details in a clean, bulleted format:
• Order Number: [Extract from order details]
• Tracking Number: [Extract from order details]
• Product Name: [Extract from order details]
• Quantity Ordered: [Extract from order details]
• Order Date: [Extract from order details]
• Delivery Date: [Extract from order details]
• Order Status: [Extract from order details]
• Delivery Address: [Extract from order details]

Follow these formatting rules:
1. Use bullet points (•) for each detail
2. Bold the labels (e.g., **Order Number:**)
3. Format dates in DD-MMM-YYYY format
4. Add a line break after the bulleted list
5. Add a friendly closing message based on the order status

Response should be friendly but professional.
"""

class OrderTrackingAgent:
    def __init__(self, csv_path: str = "order_track.csv"):
        """Initialize the OrderTrackingAgent with CSV path and LLM setup"""
        try:
            self.csv_path = Path(csv_path)
            self._validate_csv_file()
            
            # Set up Anthropic API key
            self.anthro_api_key = os.environ.get('ANTHRO_KEY')
            if not self.anthro_api_key:
                raise APIError("Anthropic API key not found in environment variables")
            
            os.environ['ANTHROPIC_API_KEY'] = self.anthro_api_key
            
            # Initialize Claude model
            self.llm = ChatAnthropic(model="claude-3-haiku-20240307")
            
            # Initialize prompts
            self.query_prompt = PromptTemplate(
                template=QUERY_ANALYSIS_PROMPT,
                input_variables=["user_query"]
            )
            self.response_prompt = PromptTemplate(
                template=ORDER_RESPONSE_PROMPT,
                input_variables=["order_details", "query_type", "action_taken"]
            )
            
            logger.info("OrderTrackingAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OrderTrackingAgent: {str(e)}")
            raise OrderTrackingError(f"Failed to initialize OrderTrackingAgent: {str(e)}")


    def _validate_csv_file(self) -> None:
        """Validate CSV file existence and structure"""
        try:
            if not self.csv_path.exists():
                raise CSVFileError(f"CSV file not found: {self.csv_path}")
            
            df = pd.read_csv(self.csv_path)
                
        except pd.errors.EmptyDataError:
            raise CSVFileError("CSV file is empty")
        except pd.errors.ParserError:
            raise CSVFileError("Invalid CSV file format")
        except Exception as e:
            raise CSVFileError(f"Error validating CSV file: {str(e)}")
        

    def _validate_state(self, state: Dict) -> None:
        """Validate state dictionary structure"""
        if not isinstance(state, dict):
            raise ValidationError("State must be a dictionary")
        required_keys = {"messages", "current_order", "current_order_details"}
    
        if not state["messages"]:
            raise ValidationError("Messages list cannot be empty")
        

    def _get_order_details(self, df: pd.DataFrame, order_track_number: str) -> Optional[Dict]:
        """Get order details from DataFrame"""
        order = df[
            (df['Order number'] == str(order_track_number)) | 
            (df['Tracking number'] == str(order_track_number))
        ]
        return order.iloc[0].to_dict() if not order.empty else None
    

    def process_tracking_query(self, state: Dict, config: dict) -> Dict:
        """Main entry point for processing order tracking queries"""
        try:
            self._validate_state(state)
            last_message = state["messages"][-1].content
            
            # Analyze query using Claude
            query_analysis = self._analyze_query(last_message)
            order_track_number = query_analysis["order_track_number"]

            # Check if we're dealing with the same order
            if order_track_number and order_track_number != state.get("current_order"):
                # New order - fetch details from CSV
                df = self._read_order_data()
                order_details = self._get_order_details(df, order_track_number)
                if order_details:
                    state["current_order"] = order_track_number
                    state["current_order_details"] = order_details
            
            # Route to appropriate handler based on query type
            handlers = {
                "refund_request": self._handle_refund_request,
                "replacement_request": self._handle_replacement_request,
                "track_order": self._handle_tracking_request,
                "general_inquiry": self._handle_general_inquiry
            }
            
            handler = handlers.get(query_analysis["query_type"])
            if not handler:
                raise ValidationError(f"Invalid query type: {query_analysis['query_type']}")
                
            return handler(state, query_analysis)
                
        except OrderTrackingError as e:
            logger.error(f"Order tracking error: {str(e)}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            return self._create_error_response(state, error_msg)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            error_msg = "I apologize, but I encountered an unexpected error processing your request."
            return self._create_error_response(state, error_msg)
        

    def _create_error_response(self, state: Dict, error_msg: str) -> Dict:
        """Create standardized error response"""
        state["tracking_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        return state
    

    def _analyze_query(self, query: str) -> Dict:
        """Analyze user query using Claude"""
        try:
            chain = self.query_prompt | self.llm
            result = chain.invoke({"user_query": query})
            
            # Validate and parse the response
            analysis = json.loads(result.content)
            required_keys = {"query_type", "order_track_number", "order_details"}
            if not all(key in analysis for key in required_keys):
                raise ValidationError("Invalid query analysis response structure")
                
            return analysis
            
        except json.JSONDecodeError:
            raise ValidationError("Failed to parse query analysis response")
        except Exception as e:
            raise APIError(f"Error analyzing query: {str(e)}")
        

    def _generate_response(self, order_details: Dict, query_type: str, action_taken: str) -> str:
        """Generate formatted response using Claude"""
        try:
            formatted_details = {
                'order_details': {
                    'Order number': order_details.get('Order number', 'N/A'),
                    'Tracking number': order_details.get('Tracking number', 'N/A'),
                    'Product name': order_details.get('Product name', 'N/A'),
                    'Quantity': order_details.get('Quantity', 'N/A'),
                    'Order date': order_details.get('Order date', 'N/A'),
                    'Delivery date': order_details.get('Delivery date', 'N/A'),
                    'Order status': order_details.get('Order status', 'N/A'),
                    'Delivery address': order_details.get('Delivery address', 'N/A')
                },
                'query_type': query_type,
                'action_taken': action_taken
            }

            chain = self.response_prompt | self.llm
            return chain.invoke({
                "order_details": str(formatted_details['order_details']),
                "query_type": formatted_details['query_type'],
                "action_taken": formatted_details['action_taken']
            }).content
            
        except Exception as e:
            raise APIError(f"Error generating response: {str(e)}")
        

    def _read_order_data(self) -> pd.DataFrame:
        """Read and validate order data from CSV"""
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise CSVFileError(f"Error reading order data: {str(e)}")
        

    def _update_order_status(self, df: pd.DataFrame, order_number: str, new_status: str) -> None:
        """Update order status in CSV file"""
        try:
            df.loc[df['Order number'] == str(order_number), 'Order status'] = new_status
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            raise CSVFileError(f"Error updating order status: {str(e)}")


    def _handle_tracking_request(self, state: Dict, query_analysis: Dict) -> Dict:
        """Handle order tracking requests"""
        try:
            order_track_number = query_analysis["order_track_number"]
            if not order_track_number:
                return self._create_error_response(
                    state,
                    "I couldn't find an order or tracking number in your query. Please provide a valid number."
                )

            # Use cached order details if available
            if order_track_number == state.get("current_order"):
                order_details = state["current_order_details"]
            else:
                df = self._read_order_data()
                order_details = self._get_order_details(df, order_track_number)
                if not order_details:
                    return self._create_error_response(
                        state,
                        f"No order found with the provided number: {order_track_number}"
                    )
                state["current_order"] = order_track_number
                state["current_order_details"] = order_details

            response = self._generate_response(
                order_details,
                "track_order",
                "Retrieved order details"
            )

            state["tracking_response"] = response
            state["messages"].append(AIMessage(content=response))
            return state

        except Exception as e:
            logger.error(f"Error in handle_tracking_request: {str(e)}")
            raise


    def _handle_refund_request(self, state: Dict, query_analysis: Dict) -> Dict:
        """Handle refund requests"""
        try:
            order_track_number_ref = query_analysis["order_track_number"]
            if not order_track_number_ref:
                return self._create_error_response(
                    state,
                    "Please provide an order number or tracking number for your refund request."
                )

            # Use cached order details if available
            if order_track_number_ref == state.get("current_order"):
                order_details = state["current_order_details"]
            else:
                df = self._read_order_data()
                order_details = self._get_order_details(df, order_track_number_ref)
                if not order_details:
                    return self._create_error_response(
                        state,
                        f"No order found with order number: {order_track_number_ref}"
                    )
                state["current_order"] = order_track_number_ref
                state["current_order_details"] = order_details

            # Update status in CSV and cached details
            df = self._read_order_data()
            self._update_order_status(df, order_track_number_ref, 'Refund Requested')
            state["current_order_details"]["Order status"] = 'Refund Requested'

            response = self._generate_response(
                order_details,
                "refund_request",
                "Processed refund request"
            )

            state["tracking_response"] = response
            state["messages"].append(AIMessage(content=response))
            return state

        except Exception as e:
            logger.error(f"Error in handle_refund_request: {str(e)}")
            raise


    def _handle_replacement_request(self, state: Dict, query_analysis: Dict) -> Dict:
        """Handle replacement requests"""
        try:
            order_track_number_replace = query_analysis["order_track_number"]
            if not order_track_number_replace:
                return self._create_error_response(
                    state,
                    "Please provide an order number or trcking number for your replacement request."
                )

            # Use cached order details if available
            if order_track_number_replace == state.get("current_order"):
                order_details = state["current_order_details"]
            else:
                df = self._read_order_data()
                order_details = self._get_order_details(df, order_track_number_replace)
                if not order_details:
                    return self._create_error_response(
                        state,
                        f"No order found with order number: {order_track_number_replace}"
                    )
                state["current_order"] = order_track_number_replace
                state["current_order_details"] = order_details

            # Update status in CSV and cached details
            df = self._read_order_data()
            self._update_order_status(df, order_track_number_replace, 'Replacement Requested')
            state["current_order_details"]["Order status"] = 'Replacement Requested'

            response = self._generate_response(
                order_details,
                "replacement_request",
                "Processed replacement request"
            )

            state["tracking_response"] = response
            state["messages"].append(AIMessage(content=response))
            return state

        except Exception as e:
            logger.error(f"Error in handle_replacement_request: {str(e)}")
            raise


    def _handle_general_inquiry(self, state: Dict, query_analysis: Dict) -> Dict:
        """Handle general inquiries"""
        try:
            response = self._generate_response(
                {},
                "general_inquiry", 
                "Processed general inquiry"
            )
            state["tracking_response"] = response
            state["messages"].append(AIMessage(content=response))
            return state

        except Exception as e:
            logger.error(f"Error in handle_general_inquiry: {str(e)}")
            raise
