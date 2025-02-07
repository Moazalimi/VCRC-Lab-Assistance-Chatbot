from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

class Chatbot:
    def __init__(self, llm, vectorstore):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            verbose=False,
            max_tokens_limit=2048
        )

    def get_response(self, question):
        try:
            response = self.qa_chain({"question": question})
            answer = response['answer'].strip()
            if not answer:
                return "I apologize, but I couldn't generate a meaningful response. Could you please rephrase your question?"
            return answer
        except Exception as e:
            return f"Error generating response: {e}"

    def chat_loop(self):
        print("\nChatbot is ready! Type 'quit' to exit.")
        print("="*50)
        while True:
            question = input("\nYou: ")
            if question.lower() == 'quit':
                print("\nGoodbye!")
                break
                
            response = self.get_response(question)
            print("\nBot:", response)
            print("-"*50)
