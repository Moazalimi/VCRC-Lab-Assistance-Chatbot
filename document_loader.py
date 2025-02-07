from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self):
        print("Initializing document processor...")
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        with tqdm(desc="Loading embeddings model", total=1) as pbar:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            pbar.update(1)

    def load_documents(self, file_path):
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None

    def process_documents(self, documents):
        if documents is None:
            return None
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        return vectorstore
