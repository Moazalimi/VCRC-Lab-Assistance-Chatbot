from document_loader import DocumentProcessor
from model_setup import ModelSetup
from chatbot import Chatbot
import os

def main():
    # Setup file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    file_path = os.path.join(data_dir, "lab_documents.txt")

    # Initialize document processor
    doc_processor = DocumentProcessor()
    
    # Load and process documents
    documents = doc_processor.load_documents(file_path)
    vectorstore = doc_processor.process_documents(documents)
    
    if vectorstore is None:
        print("Failed to process documents. Exiting...")
        return

    # Setup LLM
    model_setup = ModelSetup()
    llm = model_setup.setup_llm()

    # Create and run chatbot
    chatbot = Chatbot(llm, vectorstore)
    chatbot.chat_loop()

if __name__ == "__main__":
    main()
