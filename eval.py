import os
import pickle

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_eval_data():
    with open('./res/eval_data.pickle', 'rb') as f:
        eval_data = pickle.load(f)
        
    return eval_data

# 기존 mt-bench의 논문의 프롬프트에서 조금 수정을 가한 내용이다.
# question --> conversation, depth & creativity & level of detailedness 제외, instruction following 추가
# 정리하면 해당 대화의 요약의 결과의 질이 1~5점으로 보여주는 프롬프트이다.
def pointwise_eval(conversation, answer_a):
    client = OpenAI(api_key=OPENAI_API_KEY)
    eval_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user conversation displayed below. Your evaluation should consider factors
such as the helpfulness, relevance, and accuracy.
Begin your evaluation by providing a short explanation.The response should be
between 1 to 5 sentences. Be as objective as
possible. After providing your explanation, please rate the response on a scale of 1 to 10
by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".
[User Conversation]
{conversation}
[The Start of Assistant’s Answer]
{answer_a}
[The End of Assistant’s Answer]"""
    
    completion = client.chat.completions.create(
        model='gpt-4o-2024-05-13',
        messages=[{'role': 'user', 'content': eval_prompt}],
        temperature=0.0
    )

    return completion.choices[0].message.content