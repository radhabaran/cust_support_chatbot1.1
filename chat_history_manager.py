# chat_history_manager.py
import pandas as pd
import os
from typing import List
from langchain_core.messages import HumanMessage
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, csv_path: str = "chat_histories.csv"):
        self.csv_path = csv_path


    def _extract_human_messages(self, messages: List) -> str:
        """Extract and concatenate human messages from the chat history"""
        human_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                human_messages.append(message.content)
        return " ".join(human_messages)


    def save_chat_history(self, session_id: str, messages: List) -> None:
        """Save or update chat history for a session"""
        try:
            current_date = datetime.now().strftime("%d-%m-%Y")
            new_data = {
                'session_id': [session_id],
                'chat_messages': [self._extract_human_messages(messages)],
                'chat_captured_date': [current_date]
            }
            new_df = pd.DataFrame(new_data)

            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                df = pd.concat([df[df['session_id'] != session_id], new_df], ignore_index=True)
            else:
                df = new_df

            df.to_csv(self.csv_path, index=False)
            logger.info(f"Successfully saved chat history for session {session_id}")

        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            raise


    def delete_session_history(self, session_id: str) -> bool:
        """Delete chat history for a specific session"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning("No chat history file exists")
                return False

            df = pd.read_csv(self.csv_path)
            df = df[df['session_id'] != session_id]
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Successfully deleted chat history for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting chat history: {e}")
            return False


    def delete_all_history(self) -> bool:
        """Delete all chat history"""
        try:
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
                logger.info("Successfully deleted all chat history")
                return True
            return False

        except Exception as e:
            logger.error(f"Error deleting all chat history: {e}")
            return False