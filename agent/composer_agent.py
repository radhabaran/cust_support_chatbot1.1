# composer_agent.py

import logging
from typing import Dict, Union
from langchain_core.messages import AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compose_response(state: Dict, config: dict) -> Dict:
    """
    Process and enhance the final response
    
    Args:
        state: Current state dictionary containing messages and response_text
        config: Configuration dictionary
        
    Returns:
        Dict: Updated state with formatted response
    """
    try:
        logger.debug("Starting response composition")
        response_text = state.get("response_text", "")
        print('#' * 100)
        print('\nresponse_text received in composer agent : \n', response_text)
        print('#' * 100)
        if not response_text:
            formatted_response = "I apologize, but I couldn't find any response data to process."
        else:
            # Process the response
            formatted_response = process_response(response_text)
            
        # Update state with formatted response
        state["final_response"] = formatted_response
        state["messages"].append(AIMessage(content=formatted_response))
        
        return state
        
    except Exception as e:
        logger.error(f"Error in composition: {str(e)}")
        error_msg = "I apologize, but I encountered an error processing the response. Please try again."
        state["final_response"] = error_msg
        state["messages"].append(AIMessage(content=error_msg))
        return state


def process_response(text: str) -> str:
    """Process and format the text"""
    processed_text = remove_system_artifacts(text)
    processed_text = format_response(processed_text)
    
    return processed_text.strip()


def remove_system_artifacts(text: str) -> str:
    """Remove any system artifacts or unwanted patterns"""
    artifacts = ["Assistant:", "AI:", "Human:", "User:"]
    cleaned = text
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    # Remove double quotes
    cleaned = cleaned.replace('"', '').replace("'", "")  # Removes both double and single quotes
    
    return cleaned.strip()


def format_response(text: str) -> str:
    """Apply standard formatting"""
    # Add proper spacing
    formatted = text.replace("\n\n\n", "\n\n")
    
    # Ensure proper capitalization
    formatted = ". ".join(s.strip().capitalize() for s in formatted.split(". "))
    
    # Ensure proper ending punctuation
    if formatted and not formatted[-1] in ['.', '!', '?']:
        formatted += '.'
        
    return formatted