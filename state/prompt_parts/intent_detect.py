from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import re

#quit response 세기
intent_detect="""
사용자의 응답을 보고 현재 대화에 대한 참여 의도를 파악하세요. 
출력은 오직 0 또는 1로만 하세요.
사용자가 '잘 모르겠다', '그만하고 싶다', '어렵다' 등과 같이 대화를 중단하고 싶어하는 표현을 사용할 경우 1을 출력하세요. 
반대로, 사용자가 대화를 계속 이어가고자 하는 의지가 보일 경우 0을 출력하세요. 
"""

example_prompt = PromptTemplate.from_template(
"""
user_response: {user_response}
intent_output: {intent_output}
"""
)

examples = [
{
    "user_response": "음... 잘 모르겠어요.",
    "intent_output": "1"
},
{
    "user_response":"이 노래 들을 때마다 뭔가 힘이 나요!",
    "intent_output":"0"
},
    {
    "user_response": "좀 어려운 것 같아요. 다음에 얘기해도 될까요?",
    "intent_output": "1"
},
{
    "user_response":"이 노래 들을 때마다 뭔가 힘이 나요!",
    "intent_output":"0"
},
    {
    "user_response": "저 이 노래 진짜 좋아해요. 특히 후렴 부분이요.",
    "intent_output": "0"
},

]

intent_suffix_prompt = """
user_response: {user_response}
intent_output:
"""

intent_dectect_prompt = FewShotPromptTemplate(
prefix=intent_detect,
suffix=intent_suffix_prompt,
example_prompt=example_prompt,
examples=examples,
input_variables=["user_response"]
)


def extract_binary_int(text):
    match = re.search(r'\b[01]\b', text)
    if match:
        return int(match.group())
    else:
        return None  # 또는 예외 처리