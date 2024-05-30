# main.py
import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
#from pyngrok import ngrok
import uvicorn
import json
from model import Model
import torch
import pandas as pd
from transformers import GenerationConfig, pipeline
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from langchain.agents.agent_types import AgentType
# from langchain.agents import AgentExecutor, create_react_agent, Tool
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain_community.callbacks import StreamlitCallbackHandler
# from langchain import LLMMathChain


# Logger configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


NGROK_TOKEN = "2aQUM6MDkhjcPEBbIFTiu4cZBBr_sMMei8h5yejFbxFeMFuQ"  # Replace with your NGROK token
#MODEL_NAME = "/opt/Llama-2-13B-chat-GPTQ"
#MODEL_NAME = "MediaTek-Research/Breeze-7B-Instruct-64k-v0.1"
MODEL_NAME = "/CodeLlama-7b-Instruct-hf"
PDF_PATH = "/opt/docs"
CSV_path = "/opt/Enno"

# os.system("nvidia-smi")
# logger.info("TORCH_CUDA", torch.cuda.is_available())


model_instance = Model(MODEL_NAME)
model_instance.load()

# df_hourly_m = pd.read_csv(CSV_path+'/UsagePerHourMerged.csv', delimiter=',')

# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# generation_config.max_new_tokens = 1024
# generation_config.temperature = 0.3
# generation_config.top_p = 0.9
# generation_config.do_sample = True
# generation_config.repetition_penalty = 1.15

# text_pipeline = pipeline(
#     "text-generation",
#     model=model_instance.model,
#     tokenizer=model_instance.tokenizer,
#     return_full_text=True,
#     generation_config=generation_config,
# )

# llm = HuggingFacePipeline(pipeline=text_pipeline)

# calculator = LLMMathChain.from_llm(llm=llm, verbose=True)

# pandas_df_agent = create_pandas_dataframe_agent(
#             llm=llm,
#             df=df_hourly_m,
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             return_intermediate_steps=True,
#             max_iterations=5,
#             handle_parsing_errors=True
#         )

# # General
# general_template = """
#   [INST] <<SYS>>
#   Act as a data analyst specializing in electric power consumption data analysis.
#   <</SYS>>
#   Instructions:
#   You are provided with a series of electric power consumption data recorded once per hour. Using this hourly data, please perform a detailed analysis focusing on the following aspects:
#   1. Identify and describe any clear trends in the power usage over time, considering the hourly intervals of data collection.
#   2. Highlight any unusual events or anomalies in the hourly data series. This includes any instances of significantly high or low power usage compared to the general pattern.
#   3. Discuss any periods of power usage that stand out as being particularly high or low, providing insights into possible reasons for these fluctuations, especially in the context of their hourly recording.
#   Ensure your analysis is concise, aiming for no more than 100 words.

#   Data series:
#   {data}

#   Based on your expertise, provide a comprehensive analysis that assists in understanding the power consumption behavior reflected in the hourly recorded data.
#   [/INST]"""
# general_chain = (PromptTemplate(template=general_template,input_variables=["data"]) | llm )


# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post("/v1/chat/completions")
async def completions(request: Request):
    try:
        # Parse request body as JSON
        request_body = await request.json()
        logger.info(f"Request: {request_body}")

        #model_name = request_body.get("model", "default-model")  # Fallback to a default model if not specified
        temperature = request_body.get("temperature", 0.1)  # Default temperature
        max_tokens = request_body.get("max_tokens", 6000)  # Default max_tokens

        # Process messages to extract the prompt or any other relevant info
        messages = request_body.get("messages", [])
        prompt = ""
        for message in messages:
            if message['role'] == "user":  # Assuming we only want the last user message
                prompt = prompt = message['content']  # Keeps only the last user message as prompt

        # Use extracted parameters in your model's prediction method
        result = model_instance.predict(prompt, temperature=temperature, max_length=max_tokens)
        logger.info(f"Result: {result}")

        # Extract only the relevant part of the assistant's response
        # Assuming 'result' returns full conversation, adjust the split to your implementation
        response_content = result.split('assistant:', 1)[-1].strip() if 'assistant:' in result else result.strip()
        response_content = response_content.replace(prompt, "").strip()

        # Prepare and return the response
        formatted_response = {
            "choices": [
                {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "logprobs": None,
                "finish_reason": "stop"
                }
            ]
        }
        return formatted_response
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": "An internal error occurred"}


# @app.post("/dataframe")
# async def predict_text(request: Request):
#     try:
#         # Parse request body as JSON
#         request_body = await request.json()

#         prompt = request_body.get("prompt", "")
#         # TODO: handle additional parameters like 'temperature' or 'max_tokens' if needed
#         #st_callback = StreamlitCallbackHandler(st.container())

#         #result = agent_executor.invoke({"input": prompt})
#         result = pandas_df_agent.invoke({"input": prompt})
#         logger.info(f"Result: {result}")
#         formatted_response = {
#             "choices": [
#                 {
#                     "message": {
#                         "content": "Result: " + str(result)
#                     }
#                 }
#             ]
#         }
#         return formatted_response
#     except json.JSONDecodeError:
#         return {"error": "Invalid JSON format"}

@app.post("/text_generation")
async def analyse(request: Request):
    try:
        # Parse request body as JSON
        request_body = await request.json()

        prompt = request_body.get("prompt", "")

        result = model_instance.predict(prompt, max_length=2048)
        logger.info(f"Result: {result}")
        formatted_response = {
            "choices": [
                {
                    "message": {
                        "content": "Result: " + str(result)
                    }
                }
            ]
        }
        torch.cuda.empty_cache()
        return formatted_response
    except json.JSONDecodeError:
      torch.cuda.empty_cache()
      return {"error": "Invalid JSON format"}

@app.get("/")
async def main(request:Request):
  return "main"


if __name__ == "__main__":

    # if NGROK_TOKEN is not None:
    #     ngrok.set_auth_token(NGROK_TOKEN)

    # ngrok_tunnel = ngrok.connect(8000)
    # public_url = ngrok_tunnel.public_url

    # print('Public URL:', public_url)
    # print("You can use {}/predict to get the assistant result.".format(public_url))
    # logger.info("You can use {}/predict to get the assistant result.".format(public_url))

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999, 
                ssl_keyfile="/app/server.key",
                ssl_certfile="/app/server.crt")

