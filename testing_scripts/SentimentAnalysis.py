import os
import json
import pandas as pd
import warnings
import numpy as np
import google.generativeai as genai

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features,SentimentOptions,EmotionOptions,KeywordsOptions

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_community.llms import HuggingFacePipeline

from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run
from langsmith import Client
from langsmith import traceable
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run

from transformers  import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from google.cloud import language_v1

from langchain_community.document_loaders import HuggingFaceDatasetLoader


from langchain.evaluation import ExactMatchStringEvaluator

from langchain.smith import RunEvalConfig, run_on_dataset

dataset_name = "imdb"
page_content_column = "text"


loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()
#print(data[:15])
warnings.filterwarnings("ignore")

ROBERTA_DICT = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}


OPENAI_API_KEY='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'
SMITH_LANGCHAIN_API_KEY='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'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Update to your API key
os.environ["OPENAI_API_KEY"] = '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'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['GOOGLE_API_KEY'] = 'AIzaSyApuMpOGmmRhrCHRqWNUEooxVIb9M1-sfk'



genai.configure(api_key="AIzaSyApuMpOGmmRhrCHRqWNUEooxVIb9M1-sfk")



IAM_KEY = 'l1Or8wSmhLkMZq_L5hYc3lYK9Z_t9-8z6EhuIkuq6t2X'
SERVICE_URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/fbd0ce3e-ca15-47d7-93d6-a86dc202fea6'

authenticator = IAMAuthenticator(IAM_KEY)
natural_language_understanding = NaturalLanguageUnderstandingV1(version='2020-08-01',authenticator=authenticator)
natural_language_understanding.set_service_url(SERVICE_URL)

class SentimentEvaluator(RunEvaluator):
    def __init__(self, llm):
        prompt = """Is the predominant sentiment in the following statement positive, negative or neutral?
            ---------
            Statement: {input}
            ---------
            Respond in one word: positive, negative, neutral.
            Sentiment:"""

        llm = llm 
        self.chain = LLMChain.from_string(llm=llm, template=prompt)

    def evaluate_run(self, text: str, example: Example) -> EvaluationResult:
        #input_dict = list(run.inputs.values())[0]
        input_str = text
        prediction = self.chain.run(input_str)
        
        prediction = prediction.strip()
        score = {"positive": 1, "negative": -1, "neutral": 0}.get(prediction)
        
        with tracing_v2_enabled():
            res = EvaluationResult(
                key="sentiment",
                value=prediction,
                score=score,
            )

        print(f"str: {input_str}")
        print(res)
        return res
    
    def evaluate_str(self, text: str, example: Example) -> EvaluationResult:
     #   input_dict = list(run.inputs.values())[0]
        input_str = str
        prediction = self.chain.run(input_str)
        
        prediction = prediction.strip()
        score = {"positive": 1, "negative": -1, "neutral": 0}.get(prediction)
        
        with tracing_v2_enabled():
            res = EvaluationResult(
                key="sentiment",
                value=prediction,
                score=score,
            )

        print(f"str: {input_str}")
        print(res)
        return res
    

def evalaute_ibm_nlu(input_str):
    f = Features(
                keywords=KeywordsOptions(sentiment=True, emotion=True),
                sentiment=SentimentOptions(document=True),
                emotion=EmotionOptions(document=True)
            )
    r = natural_language_understanding.analyze(
                text=input_str, features=f, language='en'
            ).get_result()

    print(r['sentiment']['document']['label'])

def evalaute_roberta(input_str):
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    return evaluate(model, tokenizer, input_str)

def evaluate(model, tokenizer, input_str):
    encoded_input = tokenizer(input_str, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()

    index_min = np.argmax(scores)

    print(ROBERTA_DICT[index_min])


def evaluate_google_ai(input_str):
    client = language_v1.LanguageServiceClient()
    document = language_v1.types.Document(
        content=input_str, type_=language_v1.types.Document.Type.PLAIN_TEXT
    )

    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    print(f"Sentiment: {sentiment.score}, {sentiment.magnitude}")

client = Client()


chat_gpt = SentimentEvaluator(ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0))
gemini = SentimentEvaluator(ChatGoogleGenerativeAI(model="gemini-pro"))

for run in client.list_runs(
    project_name="default",
    execution_order=1, 
):
        if run.name == 'handle_message':
            client.evaluate_run(run, chat_gpt)
            client.evaluate_run(run, gemini)
            
            input_dict = list(run.inputs.values())[0]
            input_str = input_dict['text']
            evalaute_roberta(input_str)
            evaluate_google_ai(input_str)
            evalaute_ibm_nlu(input_str)
          

