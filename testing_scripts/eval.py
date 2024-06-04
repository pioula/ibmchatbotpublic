import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import HuggingFacePipeline
from getpass import getpass
from langchain_experimental.llms import ChatLlamaAPI
from langchain import PromptTemplate
# watsonx_api_key = getpass()
# os.environ["WATSONX_APIKEY"] = watsonx_api_key

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"  # Update to your API key
os.environ["OPENAI_API_KEY"] = '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'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['GOOGLE_API_KEY'] = 'AIzaSyApuMpOGmmRhrCHRqWNUEooxVIb9M1-sfk'


# os.environ["WATSONX_URL"] = "your service instance url"
# os.environ["WATSONX_TOKEN"] = "your token for accessing the CPD cluster"
# os.environ["WATSONX_PASSWORD"] = "your password for accessing the CPD cluster"
# os.environ["WATSONX_USERNAME"] = "your username for accessing the CPD cluster"
# os.environ["WATSONX_INSTANCE_ID"] = "your instance_id for accessing the CPD cluster"

LLAMA_KEY = 'LL-LL-0pAUefP99qz9SA6UNI9HhKs8vZZ930n2HoykF7DVJLJL4gja6LOniAzXbZI1g6Vs'


ke = 'u1zsQFADpiISisQx-nWhZuJXvj79l9ZWLUOmD2mETaSF'
IAM_KEY = 'l1Or8wSmhLkMZq_L5hYc3lYK9Z_t9-8z6EhuIkuq6t2X'
SERVICE_URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/fbd0ce3e-ca15-47d7-93d6-a86dc202fea6'





genai.configure(api_key="AIzaSyApuMpOGmmRhrCHRqWNUEooxVIb9M1-sfk")
from langchain.chat_models import ChatOpenAI 
from langchain.evaluation import ExactMatchStringEvaluator
from langchain.chains import LLMChain 
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client 
import pandas as pd
from langsmith.evaluation import EvaluationResult, run_evaluator

from llamaapi import LlamaAPI
from langchain_ibm import WatsonxLLM


WX_API ="https://us-south.ml.cloud.ibm.com"
WX_PRJ = "c083f396-6855-452d-a0b4-67c48f1e16cd"
WX_KEY = "u1zsQFADpiISisQx-nWhZuJXvj79l9ZWLUOmD2mETaSF"
os.environ["WATSONX_APIKEY"] = WX_KEY
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 0,
    "temperature": 0,
    "top_k": 50,
    "top_p": 1,
}



llm = WatsonxLLM(
    model_id="ibm/granite-13b-chat-v2",
    url=WX_API,
    project_id=WX_PRJ,
   # params=parameters
)

# from langchain_huggingface import HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-70B-Instruct",
#     task="sentiment-analysis",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )

# llm = HuggingFacePipeline(pipeline=llm, model_kwargs={"temperature": 0})

# Replace 'Your_API_Token' with your actual API token
# llama = LlamaAPI(LLAMA_KEY)
# llm = ChatLlamaAPI(client=llama)
client = Client() 

#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) 
#llm = ChatGoogleGenerativeAI(model="gemini-pro")


def create_chain(): 
    # prompt = """
    #         You are an expert in sentiment analysis and you responding using only one word: positive or negative.
    #         Here are some examples:
    #         ---------
    #         Statement: I like you
    #         ---------
    #         Sentiment:positive
    #         --------
    #         Statement: I hate you
    #         ---------
    #         Sentiment:negative
    #         Is the predominant sentiment in the following statement positive, negative?
    #         ---------
    #         Statement: {input}
    #         ---------
    #         Respond in one word:positive,negative.
    #         Sentiment:"""
        
    prompt = """<|system|>
        Determine the sentiment of the user statement.Respond in one word: positive, negative.
        <|user|>
        {input}
        <|system|>"""
    return LLMChain.from_string(llm, prompt) 


csv_file = '/Users/pawelligeza/Studia/IBM/chatbot_ibm/IMDB Dataset.csv'
df = pd.read_csv(csv_file) 



# strr = """Is the predominant sentiment in the following statement positive, negative?
#             ---------
#             Statement: {input}
#             ---------
#             Respond in one word: positive, negative.
#             Sentiment:"""
    
# prompt = PromptTemplate(
#         input_variables=["input"],
#         template=strr,
#     )

# chain = LLMChain.from_string(llm=llm, template=strr)

@run_evaluator
def compare_label(run, example) -> EvaluationResult:
    # Custom evaluators let you define how "exact" the match ought to be
    # It also lets you flexibly pick the fields to compare

    prediction = run.outputs.get("text") or ""
    target = example.outputs.get("answer") or ""
    
    score = 0

    if target == 'positive' and target in prediction.lower() and 'negative' not in prediction.lower():
        score = 1
    if target == 'negative' and target in prediction.lower() and 'positive' not in prediction.lower():
        score = 1
    return EvaluationResult(key="matches_label", score=score)

dataset_name = "Smaller Imdb-dataset" 

evaluator = ExactMatchStringEvaluator(
    ignore_case=True,
    ignore_punctuation=True,
)


# chain = create_chain()
# res = chain.run("i like you")
# print(res)
# print('positive')

# print(evaluator.evaluate_strings(
#     prediction=res,
#     reference="positive",
# ))
 

# dataset = client.create_dataset( 
#     dataset_name=dataset_name, description="Smaller Imdb - sentiment", 
# ) 

 
# Total = 500
# ct = 0
# for input_prompt, output_answer in zip(df['review'], df['sentiment']):

#     client.create_example( 

#         inputs={"question": input_prompt}, 

#         outputs={"answer": output_answer}, 

#         dataset_id=dataset.id, 

#     )

#     if ct > Total:
#         break

#     ct += 1



evaluation_config = RunEvalConfig( 
    custom_evaluators=[compare_label],
) 

run_on_dataset( 

    client=client, 

    dataset_name=dataset_name, 

    llm_or_chain_factory=create_chain, 

    evaluation=evaluation_config, 

    project_name="Smaller ibm-better-test-imdb" 

) 