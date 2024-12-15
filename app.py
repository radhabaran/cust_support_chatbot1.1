from dotenv import load_dotenv
import logging
import uuid
from typing import Dict, Annotated, TypedDict, List, Tuple, NotRequired
from interface import create_interface
from agent.planning_agent import setup_agent_graph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: NotRequired[str]
    router_response: NotRequired[str]
    generic_response: NotRequired[str]
    product_info: NotRequired[str]
    final_response: NotRequired[str]
    current_order: NotRequired[str]
    current_order_details: NotRequired[str]
    tracking_response: NotRequired[str]


class AgentManager:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.graph, self.memory = setup_agent_graph(State)
        logger.info(f"Initialized AgentManager with session_id: {self.session_id}")
        self.config = {"configurable": {"thread_id": self.session_id}}

    def process_query(self, query: str, history: List[Tuple[str, str]], session_id: str=None) -> str:
        try:
                    
            # Create input state with just the new message
            input_state = {
                "messages": [HumanMessage(content=query)],
                "session_id" : self.session_id
            }
        
            # Debug print for input state
            print('*' * 100)
            print("\n\nBefore Graph Invoke: ")
            print("Messages in state:")
            for msg in input_state["messages"]:
                print(f"Type: {type(msg).__name__}")
                print(f"Content: {msg.content}")
            print('*' * 100)

            # Langgraph will automatically merge this with existing state
            result = self.graph.invoke(input_state, config=self.config)
            
            # Debug print for result
            print('@' * 100)
            print("\n\nAfter Graph Invoke: ")
            print(f"\nTotal messages in state: {len(result['messages'])}")
            print(f"\nFinal Response: {result.get('final_response')}")
            print(f"\nRouter Response: {result.get('router_response')}")
            print("\n\nMessages in result:")
            for msg in result.get("messages", []):
                print(f"Type: {type(msg).__name__}")
                print(f"Content: {msg.content}")
            print('@' * 100)

            return result["final_response"]
            
        except Exception as e:
            logger.error(f"Error processing query in app.py : {e}")
            return f"Error: {str(e)}"


    def clear_context(self, session_id: str) -> tuple[List, str]:
        """Clear the conversation context for a session"""
        try:
            return [], ""
        except Exception as e:
            logger.error(f"Error clearing context: {e}")
            return [], ""
              

def main():
    try:
        load_dotenv()              
        agent_manager = AgentManager()
        
        logger.info(f"Starting Gradio app")
        app = create_interface(
            process_query=agent_manager.process_query,
            agent_manager=agent_manager,
            session_id=agent_manager.session_id
        )
        app.queue()
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()