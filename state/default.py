from langchain.prompts import PromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from .prompt_parts.prefix import question_prefix_prompt, slot_prefix_prompt
from .prompt_parts.intent_detect import intent_dectect_prompt, extract_binary_int
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI



# 오늘 하루에 대한 기록 -> 음악감상이유, 기억에 남는 가사 + 이유, 기억에 남는 요소 + 이유
# 웹서칭 -> 음악에 대해 아는 척 
# task 변경할때 질문도 해달라고 해야함! 
class SlotFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""
    music_reason: Optional[str] = Field(default=None, description="오늘 이 노래를 들은 이유")
    music_today_relationship: Optional[str] = Field(default=None, description="이 노래가 오늘 하루의 일상이나 사건과 어떤 관계가 있었는지")
    impressive_lyrics_reason: Optional[str] = Field(default=None, description="특정 가사가 인상 깊은 이유")
    impressive_music_component: Optional[str] = Field(default=None, description="인상깊은 노래요소")
    impressive_music_component_reason: Optional[str] = Field(default=None, description="특정 노래요소가 인상 깊은 이유")

quit_threshold={'turn_min':7, 'turn_max':20, 'quit_intent':3}

def default(user_input, data, llm) -> str:
    flag=0
    option=None

    #intent detection
    intent_dectect_chain = intent_dectect_prompt | llm | StrOutputParser()
    intent_output = intent_dectect_chain.invoke({"user_response":user_input})

    intent=extract_binary_int(intent_output)
    if intent is not None:
        data['quit_response']+=intent
    
    memory_vars = data['memory'].load_memory_variables({})
    history = memory_vars.get("history", "")
    print('history\n',history)

    #slot 채우기
    structured_llm = llm.with_structured_output(schema=SlotFormat)
    slot_prompt = PromptTemplate(input_variables=["history"], template=slot_prefix_prompt + "\n" + "Chat history: {history}")
    final_prompt = slot_prompt.format(history=history)
    slot = structured_llm.invoke(final_prompt)


    if data['turn']!=0 and data['impressive_lyrics'] is None and slot.music_reason is not None and slot.music_today_relationship is not None:
        response="노래에서 인상 깊었던 구절은 무엇인가요?"
        option='option6'
        return response, data, flag,option
    

    today_question_response="""
    노래와 함께 오늘 하루에 대한 짧은 기록을 진행합니다.
    
    [Short Journaling Task]
    Task의 목적: 오늘 들은 {title}에 대해 이야기하며 그 음악과 관련된 오늘 하루에 대한 짧은 대화를 나눕니다. follow question도 진행하세요. 
    
    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "오늘 이 노래를 들은 이유가 무엇인가요?"
    - "오늘 있던 일과 이 노래는 어떤 관계가 있나요?"
    """

    impressive_question_response= """
    노래를 중심으로 하여 그 음악을 왜 들었는지 노래의 가사, 음악요소 중 인상 깊은 부분은 무엇인지, 왜 인상깊은지에 대해 물어봅니다.

    [Impressive Lyrics Task]
    Task의 목적: {title}에서 왜 {impressive_lyrics} 이라는 가사가 인상깊은지 이유를 물어보세요. 

    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "왜 기억에 남았나요?"
    - "이 구절이 왜 마음에 와닿았나요?"
    - "오늘따라 이 가사가 인상깊은 이유가 무엇인가요?"

    [Impressive Music Component Task]
    Task의 목적: {title}에서 인상깊은 노래요소를 묻고 이유를 물어보세요.

    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "이 노래에서 가장 좋아하는 노래 요소가 무엇인가요?"
    - "왜 이 노래요소를 좋아하나요?"

    - "노래에서 가장 좋아하는 부분이 어디인가요?"
    - "왜 그 부분을 좋아하나요?"
    """

    today_question_few_shot="""
    AI: 우효의 brave를 들으셨군요. 이 노래을 들으신 이유가 있을까요?
    Human: 마음이 편안해지고 싶어서요.
    AI: 왜 그렇게 생각했나요?
    Human: 이 노래의 가사가 저를 응원해주는 것 같아서 그렇게 생각했어요.
    """

    impressive_question_few_shot="""
    AI: 이 구절이 왜 마음에 와닿았나요?
    Human: 요즘 힘든 일이 많았는데, 그 문장을 듣는 순간 눈물이 나더라고요. 지금의 아픔도 언젠가는 의미 있는 무언가로 바뀔 수 있다는 말처럼 느껴졌어요.
    AI: 이 노래에서 가장 좋아하는 노래 요소가 무엇인가요?
    Human: 후렴에서 반복되는 "Wanna be brave" 부분이랑 그때 나오는 멜로디가요.
    AI: 왜 그 부분을 좋아하나요?
    Human: 점점 고조되는 리듬이 마치 제 안에 있던 용기를 끌어올려주는 것 같았어요. 우효의 목소리도 점점 단단해지면서 정말 "나도 용기 내고 싶다"는 생각이 들게 해요.
    """

    #impressive lyrics를 골랐는지 아닌지에 따라 prompt 변경
    if data['impressive_lyrics'] is None:
        question_response_prompt=today_question_response
        full_few_shot_dialogue=today_question_few_shot
    else:
        question_response_prompt=impressive_question_response
        full_few_shot_dialogue=impressive_question_few_shot


    question_prompt = PromptTemplate(
        input_variables=["user_message", "history", "pre_slot", "title", "impressive_lyrics"],
        template=
        question_prefix_prompt
        + "\n"
        + question_response_prompt
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "아래를 보고 참고하여 질문을 생성하세요."
        + "Previous Slot: {pre_slot}\n"
        + "Chat history: {history}\n"
        + "User said: {user_message}",
    )

   
    question_chain = question_prompt | llm | StrOutputParser()
    response = question_chain.invoke({"user_message": user_input, "history": history, "pre_slot": slot, "title": data['title'], "impressive_lyrics":data['impressive_lyrics']})
 
    data['memory'].save_context({"input": user_input}, {"output": response})

    #quit 
    if data['turn'] > quit_threshold['turn_min']:
        if data['quit_response']>=quit_threshold['quit_intent']:
            flag=1
            option='option2'
        
        checked_slot=slot.model_dump()
        if None not in checked_slot.values():
            flag=1
            option='option2'

        
    if data['turn'] >= quit_threshold['turn_max']:
        flag=1
        option='option2'

    return response, data, flag, option
