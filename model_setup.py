import warnings
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

class ModelSetup:
    def __init__(self, model_name="google/flan-t5-base"):
        self.model_name = model_name
        # Suppress warnings
        warnings.filterwarnings("ignore")
        transformers.logging.set_verbosity_error()

    def setup_llm(self):
        print("Loading language model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.2,
            min_length=30,
            do_sample=True,
            num_return_sequences=1
        )
        
        return HuggingFacePipeline(pipeline=pipe)
