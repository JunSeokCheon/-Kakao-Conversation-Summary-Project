import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_eval_data():
    with open('./res/eval_data.pickle', 'rb') as f:
        eval_data = pickle.load(f)
        
    return eval_data