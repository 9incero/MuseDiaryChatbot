from langchain.prompts import PromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from .prompt_parts.prefix import question_prefix_prompt, slot_prefix_prompt
from .prompt_parts.intent_detect import intent_dectect_prompt,extract_binary_int

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


class SlotFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""
    appreciation: Optional[str] = Field(default=None, description="만든 노래를 들은 감상평")
    feeling: Optional[str] = Field(default=None, description="노래 만든 소감")


   
quit_threshold={'turn_min':1, 'turn_max':5, 'quit_intent':3}

def discussion(user_input, data, llm) -> str:
    flag=0
    if data['turn']==0:
        data['option']='option2'
        response=''
        return response, data, flag

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



    discussion_question_response="""
    만든 노래에 대해 이야기를 나눕니다. 
    [Song Discussion]
    Task의 목적: 대화를 마무리하며 오늘 만든 노래에 대해서 이야기를 나눕니다.

    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - "오늘 노래를 만든 소감은 어떠신가요?"
    - "만든 노래를 들으니 어떤가요?"
    """


    full_few_shot_dialogue="""
    AI: 만든 노래를 들으니 어떠신가요?
    Human: 신기하고 재밌어요.
    AI: 저도 오늘 너무 좋은 경험이었습니다. 다음에 또 뵙겠습니다. 
    """


    question_prompt = PromptTemplate(
        input_variables=["user_message", "history", "pre_slot"],
        template=
        question_prefix_prompt
        + "\n"
        + discussion_question_response
        + "\n"
        + full_few_shot_dialogue
        + "\n"
        + "아래를 보고 참고하여 질문을 생성하세요."
        + "Previous Slot: {pre_slot}\n"
        + "Chat history: {history}\n"
        + "User said: {user_message}",
    )

   
    question_chain = question_prompt | llm | StrOutputParser()
    response = question_chain.invoke({"user_message": user_input, "history": history, "pre_slot": slot})
 
    data['memory'].save_context({"input": user_input}, {"output": response})

    #quit 
    if data['turn'] > quit_threshold['turn_min']:
        if data['quit_response']>=quit_threshold['quit_intent']:
            flag=1
        
        checked_slot=slot.model_dump()
        if None not in checked_slot.values():
            flag=1
        
    if data['turn'] >= quit_threshold['turn_max']:
        flag=1

    return response, data, flag



