import os
import pickle
import time

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv['ANTHROPIC_API_KEY']
GOOGLE_API_KEY = os.getenv['GOOGLE_API_KEY']
OPENAI_API_KEY = os.getenv['OPENAI_API_KEY']
MAX_LEN = 3000

# 3000자가 넘으면 가장 최근 3000자만 반환하는 함수
def shorten_conv(conversation):
    shorten_len = len(conversation)
    lst = conversation.split('\n')
    # 제일 오래된 대화부터 한 대화씩 지우되, 지우고 난뒤 3000자보다 적거나 같은지 확인한다.
    for i, l in enumerate(lst):
        utterance_len = len(l)
        shorten_len -= utterance_len
        if shorten_len <= MAX_LEN:
            break
    
    # 최신 3000자 반영
    lst_shortened = lst[i+1:]
    conv_shortened = '\n'.join(lst_shortened)
    return conv_shortened

# 조건(글자수, 모델, 프롬프트 ..)에 따라 요약을 생성해주는 함수
def summarize(conversation, prompt, temperature=0.0, model='gpt-3.5-turbo'):
    if len(conversation) > MAX_LEN:
        conversation = shorten_conv(conversation)
    
    prompt = prompt + '\n\n' + conversation
    
    if 'gpt' in model:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role':'user', 'content':prompt}],
            temperature=temperature
        )
        
        return completion.choices[0].message.content
    elif 'gemini' in model:
        genai.configure(api_key=GOOGLE_API_KEY)
        client = genai.GenerativeModel(model)
        response = client.generate_content(
            contents=prompt,
            # 위험하지 않은데 위험하다고 판단 되는 경우가 있기 때문에 출력의 용이성을 위해 허용 처리
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        # 호출 횟수 제한 때문에 (분당 60회)
        time.sleep(1)
        
        return response.text
    elif 'claude' in model:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=model,
            messages=[{'role':'user', 'content':prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        
        return message.content[0].text
    
# 훈련 데이터 load
def get_train_data():
    with open('./res/train_data.pickle', 'rb') as f:
        train_data = pickle.load(f)
    
    return train_data