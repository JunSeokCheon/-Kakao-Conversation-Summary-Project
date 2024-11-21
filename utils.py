import os
import pickle
import time

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
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


def get_prompt():
    conv_train = """P01: 코로나가 좀 잠잠해지면 해외여행 중에 가고 싶은 곳 있어?
P02: 난 호주 한번 다시 다녀오고 싶엉 키키
P02: 아님... 원래 일본도 가보고 싶었는데 무서워서 안 갈래 키키
P01: 일본은 예전 노 재팬 한 이후로 사실 아직도 웬만하면 일본꺼 안 쓰거든 키키
P01: 나도 일본은 별로고 키키
P01: 나는 하와이 괌 이런 곳 가보고 싶어 키키
P02: 휴양지 좋지 키키
P02: 아아! 베트남 다낭 가보고 싶어~
P02: 좋다던데 키키
P01: 맞아 맞아 나 남편이랑 연애 초기에 남편 첫 직장 입사하고 그 해에 바로 거기로 해외 워크샵 갔는데 진짜 완전 질투나서 키키 연락하다 싸움 키키
P02: 키키 질투 났어? 키키
P02: 아 웃곀 키키
P02: 나중에 우리끼리 가자
P01: 키키 아니 사귄 지 얼마 안 됐는데 아 얼마 안된 건 아닌가 키키
P01: 무튼 연락도 안 되고 전화하면 혼자 신나 있고 난 연락 기다리느라 계속 신경은 거기로만 가있는대 얘는 아니니까 키키
P01: 흔한 그 연인들의 싸움이었지 키키
P01: 무튼 가보고 싶긴 해
P02: 키키 신나 있었어?
P02: 이런 키키
P02: 진짜 떠나고 싶당~
P01: 웅 키키 중딩 때 태권도에서 일본 간 거 말고 첫 해외여행이라고 매우 들떠있었지 키키
P01: 호텔도 엄청 좋은 데로 가고 키키
P01: 첫 직장에 첫 해외니 얼마나 좋았겠어 키키
P02: 엄청 좋지 키키
P02: 우리 대학교 졸업 여행 간 거 기억나?
P02: 태국 키키
P02: 난 그게 첫 해외여행이었어
P01: 기억나지 키키
P01: 내 얼굴에서만 플래쉬 터지고... 키키
P01: 아 재밌었지 웃기고 키키
P01: 태국 가보고 그다음에 간 게 호주 워홀이었던 건가?
P02: 키키 아 맞아~ 플래쉬 키키
P02: 옹옹~ 그 담이 호주였지~
P02: 대담해써 아주키키
P01: 그니까 용기 있었어 키키
P01: 대단해
P01: 돈 열심히 모아서 진짜 스위스 꼭 가봐
P01: 죽어도 여한 없을 것 같은 그런 느낌이야... 키키
P02: 신혼으로 스위스 갔지?
P02: 그때 사진 보고 넘 부러웠징~ 키키
P02: 여행 갈 나라가 너무 많다
P01: 그치 키키 가보고 싶은 나라가 많아
P01: 근데 사실 나는 아시아권 보단 유럽 가고 싶어 ㅠ
P02: 유럽은 뭔가 고급 지고 멋진 느낌이야 키키
P02: 죽기 전에는 가보겠지?
P01: 그치 키키 그냥 여유로운 느낌이 좋아
P01: 근데 뭐 유럽이라고 다 그렇지는 않더라 키키
P01: 좋은 사람도 있지만 나쁘고 위험한 사람도 많고 ㅠㅠ
P02: 맞아~ 위험하고 무서운 사람들 있지 ㅠㅠ
P02: 인도는 여자들 여행 가면 큰일 난대~ 없어져도 모른대~
P01: 그래 맞아 ㅠㅠ
P01: 내 친구는 유럽 어디더라 친구 두 명이랑 같이 여행 갔는데 짐을 통제로 소매치기 당해서 엄청 곤란한 상황이었다고 하더라고 ㅠ
P02: 어후...
P02: 여권도 잃어버렸을 거 아냐 ㅠㅠ
P02: 진짜 멘붕 됐겠다... 으악
P01: 엉 그니까 ㅠ 그래도 뭐 어찌어찌 위기는 넘긴 거 같더라 ㅠ
P01: 해외 여행 가면 가이드 안 끼고 가면 영어 실력 엄청 좋은 사람이랑 가는 거 아닐 땐 소통 문제도 좀 곤란할 때가 있는 거 같아
P02: 맞아~ 영어는 진짜... 잘하고 싶다 키키
P02: 왜 영어 공부를 열심히 안 했을까... 후회 키키"""

    prompt = f"""당신은 요약 전문가입니다. 사용자 대화들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 대화를 요약할 때는 다음 단계를 따라주세요:

1. 대화 참여자 파악: 대화에 참여하는 사람들의 수와 관계를 파악합니다.
2. 주제 식별: 대화의 주요 주제와 부차적인 주제들을 식별합니다.
3. 핵심 내용 추출: 각 주제에 대한 중요한 정보나 의견을 추출합니다.
4. 감정과 태도 분석: 대화 참여자들의 감정이나 태도를 파악합니다.
5. 맥락 이해: 대화의 전반적인 맥락과 배경을 이해합니다.
6. 특이사항 기록: 대화 중 특별히 눈에 띄는 점이나 중요한 사건을 기록합니다.
7. 요약문 작성: 위의 단계에서 얻은 정보를 바탕으로 간결하고 명확한 요약문을 작성합니다.
각 단계를 수행한 후, 최종적으로 전체 대화를 200자 내외로 요약해주세요.

아래는 예시 대화와 예시 요약 과정 및 결과 입니다.

예시 대화:
{conv_train}

예시 요약 과정
1. "우리 대학교 졸업 여행 간 거 기억나?"라는 언급과 전반적으로 친밀한 대화 톤을 사용하고 있는 것을 보았을 떄 두 사용자는 오랜 친구 사이로 보입니다.
대화의 시작 부분에서 "코로나가 좀 잠잠해지면 해외여행 중에 가고 싶은 곳 있어?"라고 묻고 있는 것을 보았을 때 코로나 이후 가고 싶은 해외 여행지에 대해 논의하고 있습니다.
따라서 다음과 같이 요약 할 수 있습니다:
최소 대학 생활부터 함께 한 매우 친밀한 사이의 두 사용자가 코로나가 잠잠해졌을 때 방문하고 싶은 해외 여행지에 대해 일상적이고 가벼운 톤으로 대화하고 있습니다.

2. 대화 중 호주, 일본, 하와이, 괌, 베트남 다낭, 스위스, 유럽들이 언급하고 있습니다.
남편의 첫 직장 워크샵, 대학교 졸업 여행, 호주 워킹홀리데이 등의 경험을 이야기하면서 과거 여행 경험을 공유하며 추억을 회상하고 있습니다.
따라서 다음과 같이 요약 할 수 있습니다:
여행지로는 하와이, 괌, 스위스, 호주, 베트남 다낭 등을 언급하며 남편과의 연락 관련 다툼이나 졸업여행 관련 추억을 회상합니다.

3. 소매치기, 여권 분실, 인도에서의 여성 여행자 위험 등을 언급하며 해외 여행의 위험성에 대해 우려를 표현하고 있습니다.
"해외 여행 가면 가이드 안 끼고 가면 영어 실력 엄청 좋은 사람이랑 가는 거 아닐 땐 소통 문제도 좀 곤란할 때가 있는 거 같아"라는 언급과 "왜 영어 공부를 열심히 안 했을까... 후회"라는 표현이 있는 것을 보았을 때 언어 장벽의 어려움을 인식하고 영어 실력 향상에 대한 욕구를 표현합니다.
따라서 다음과 같이 요약 할 수 있습니다:
또한 여행 중 발생하는 위험에 대한 우려도 표하고 있으며, 해외여행 시 언어 장벽의 어려움을 인식하고 영어 실력을 향상시키고 싶다는 마음을 가볍게 표현합니다.

예시 요약 결과
최소 대학 생활부터 함께 한 매우 친밀한 사이의 두 사용자가 코로나가 잠잠해졌을 때 방문하고 싶은 해외 여행지에 대해 일상적이고 가벼운 톤으로 대화하고 있습니다.
여행지로는 하와이, 괌, 스위스, 호주, 베트남 다낭 등을 언급하며 남편과의 연락 관련 다툼이나 졸업여행 관련 추억을 회상합니다.
또한 여행 중 발생하는 위험에 대한 우려도 표하고 있으며, 해외여행 시 언어 장벽의 어려움을 인식하고 영어 실력을 향상시키고 싶다는 마음을 가볍게 표현합니다.
    
아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""
    return prompt