from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from .prompt_parts.prefix import question_prefix_prompt, slot_prefix_prompt, chat_state_prefix_prompt
from .prompt_parts.intent_detect import intent_dectect_prompt, extract_binary_int
from .music_generate import music_creation

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


class SlotFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""
    selected_music_tag_reason: Optional[str] = Field(default=None, description="특정 노래 요소를 선택한 이유")
    add_selected_music_component: Optional[str] = Field(default=None, description="추가적으로 선택하고 싶은 노래 요소")
    add_selected_music_component_reason: Optional[str] = Field(default=None, description="추가적으로 노래요소를 선택한 이유")
    
quit_threshold={'turn_min':5, 'turn_max':10, 'quit_intent':3}


def mid_music_selection(user_input, data, llm):
    flag=0
    option=None
    if data['turn']==0:
        option='option5'
        response=''
        return response, data, flag, option
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


    question_response_prompt="""
    인상 깊은 노래 요소나 노래에 새롭게 추가하고 싶은 노래요소를 파악하고 이유를 찾습니다. 

    [Music Tag Reason Task]
    Task의 목적: 선택한 {selected_music_tag}에 대해 이야기합니다.
    왜 선택했는지 왜 마음에 드는지를 알아내세요. 
    
    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "이 요소를 선택한 이유는 무엇인가요?"
    - "이 악기를 선택한 이유는 무엇인가요?"
    - "이런 요소들이 곡을 어떻게 느끼게 만들었나요?"  

    [Add New Music Component Task]
    Task의 목적: 앞서 선택한 {selected_music_tag}에 덧붙여 선택할 요소를 물어보세요.
    이것을 왜 추가적으로 선택하고 싶은지 알아내세요. 없을수도 있습니다. 

    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "이 요소를 추가적으로 선택한 이유는 무엇인가요?"
    - "이 악기를 추가적으로 선택한 이유는 무엇인가요?"
    - “딱히 더 추가하고 싶은 요소가 없다면, 그렇게 말씀해 주셔도 괜찮아요!”
    """

    full_few_shot_dialogue="""
    AI: 드럼을 선택하신 이유는 무엇인가요? 어떤 리듬이 특히 인상 깊었나요?
    Human: 반복되는 드럼 패턴이 중독성 있어서 계속 따라하게 됐어요.
    AI: 그 리듬에 어떤 요소를 더해보고 싶으신가요? 예를 들어 추가적인 악기나 분위기 같은 것들이요. 없으셔도 괜찮아요!
    Human: 약간의 전자음이 들어가면 더 흥겨울 것 같아요.
    """

    question_prompt = PromptTemplate(
    input_variables=["user_message", "history", "pre_slot", "selected_music_tag"],
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
    response = question_chain.invoke({"user_message": user_input, "history": history, "pre_slot": slot, "selected_music_tag": data['music_tag']})
 
    data['memory'].save_context({"input": user_input}, {"output": response})


    #quit 
    if data['turn'] > quit_threshold['turn_min']:
        if data['quit_response']>=quit_threshold['quit_intent']:
            response, data, flag, option = music_creation("", data, llm, slot)
        
        checked_slot=slot.model_dump()
        if None not in checked_slot.values():
            response, data, flag, option = music_creation("", data, llm, slot)
        
    if data['turn'] >= quit_threshold['turn_max']:
        response, data, flag, option = music_creation("", data, llm, slot)

    return response, data, flag, option

