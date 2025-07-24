from langchain.prompts import PromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from .prompt_parts.prefix import question_prefix_prompt, slot_prefix_prompt, chat_state_prefix_prompt
from .prompt_parts.intent_detect import intent_dectect_prompt, extract_binary_int

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI


class SlotFormat(BaseModel):
    """사용자의 응답에서 얻어내야하는 정보"""
    concept: Optional[str] = Field(default=None, description="The story or theme the user wants to express in the lyrics")
    concept_discussion: Optional[str] = Field(default=None, description="A summary of what the user shared about their concept during the [Concept Discussion Task]")
    lyric_keyword: Optional[str] = Field(default=None, description="The main keyword that comes to mind when expressing the intended theme")
    lyrics_content: Optional[str] = Field(default=None, description="Detailed sentences the user wants to include in the lyrics")

quit_threshold={'turn_min':5, 'turn_max':15, 'quit_intent':3}


def high_lyrics_concept(user_input, data, llm):
    flag=0
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
    [Making Concept Task]
    Task의 목적: 노래 가사에 대한 주제를 정합니다.
    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - 오늘 하루 중 기억에 남는 순간이 있다면, 그 장면을 음악으로 표현해볼까요? 그 장면을 설명해주세요. 
    - 노래 안에 어떤 이야기를 담고 싶나요?
    - 오늘 느낀 감정이나 상황을 음악으로 담고 싶나요?

    [Concept Discussion Task]
    Task의 목적: 사용자가 제시한 주제에 대해 더 깊이 이야기합니다.
    - 왜 이런 주제를 선택했는지, 어떤 감정이나 이유가 있는지를 물어보세요.
    - 이 이야기를 통해 사용자가 느낀 감정을 함께 들여다보세요.
    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - 왜 이런 주제에 대해 이야기를 하고 싶은가요?
    - 이 가사를 공유하고 싶은 사람이 있나요?
    - 이 이야기를 생각하며 어떤 감정을 느꼈나요?
    - 이 이야기를 생각하면 스스로에게 어떤 말을 해주고 싶은가요?

    [Making Lyrics Task]
    Task의 목적: 가사를 구성할 아이디어와 표현을 수집합니다.
    - 단어, 문장, 이미지 등 사용자 머릿속에 떠오르는 모든 것을 자유롭게 꺼낼 수 있도록 도와주세요.
    - 필요한 경우 선택지를 제시하면서, 현재 주제나 감정에 어울리는 표현을 이유와 함께 제안해주세요.
    질문 예시 (LLM이 사용자에게 직접 묻는 형식):
    - 주제를 표현하기 위해서 어떤 단어가 떠오르시나요?
    - 주제를 표현하기 위해서 어떤 느낌이 떠오르시나요?
    - 짧은 문장이나 단어라도 괜찮아요. 떠오르는 문구가 있다면 자유롭게 적어보세요.
    """

    full_few_shot_dialogue="""
    이 대화의 형식같이 대화를 진행하세요.  
    <예시 대화>
    AI: 오늘 하루 중 기억에 남는 순간이 있다면, 그 장면을 음악으로 표현해볼까요?
    Human: 오늘 퇴근길에 갑자기 소나기가 쏟아졌는데, 우산도 없이 비 맞으면서 걸었어요. 그런데 그게 이상하게도 기분이 나쁘진 않았어요.
    AI: 그 장면을 음악으로 담아본다면, 어떤 분위기의 곡이 될 것 같나요?
    AI: 왜 이런 주제에 대해 이야기를 하고 싶은가요?
    Human: 요즘 마음이 복잡했는데, 그날 비 맞으면서 걸었던 시간이 오히려 정리되는 느낌이었어요. 그래서 그 장면이 자꾸 생각나요.
    AI: 그 순간, 어떤 감정이 제일 강하게 느껴졌나요? 혹시 그 감정을 누군가와 나누고 싶다는 생각도 들었나요?
    Human: 약간의 외로움, 근데 동시에 괜찮다는 느낌도요. 누군가가 "잘하고 있어"라고 말해줬으면 좋겠다는 생각이요.
    AI: 그 감정을 표현하기 위해 어떤 단어나 이미지가 떠오르시나요?
    Human: 젖은 운동화, 가로등 밑 그림자, 빗소리에 가려진 숨소리 같은 게 떠올라요.
    AI: 좋네요. "가로등 밑에 선 그림자"는 고요하면서도 쓸쓸한 느낌을 줄 수 있어요. 혹시 짧은 문장이나 떠오르는 구절이 있다면 적어주실 수 있나요?
    Human: “말없이 나를 감싸던 소나기처럼”이라는 문장이 떠올라요.
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
            response, data, flag = making_lyrics("", data, llm, slot)
        
        checked_slot=slot.model_dump()
        if None not in checked_slot.values():
            response, data, flag = making_lyrics("", data, llm, slot)
        
    if data['turn'] >= quit_threshold['turn_max']:
        response, data, flag = making_lyrics("", data, llm, slot)

    return response, data, flag

    


def making_lyrics(user_input, data, llm, slot):
    flag=1

    memory=data['memory']
    memory_vars = memory.load_memory_variables({})
    history = memory_vars.get("history", "")
    
    making_lyrics_prompt = """
    [Lyrics Generative Task]
    - {slot},{history}을 바탕으로 가사를 생성합니다.
    - 아래와 같은 format으로 output을 제시해야합니다.
    Output Format:
    [Verse]
    깊은 밤에 홀로 앉아
    차가운 달빛 아래 머물러
    고요한 바람 속에 숨결을 찾아
    흐릿한 기억 속을 헤매네

    [Verse 2]
    별빛도 나를 외면하네
    그리움은 마음을 감싸네
    텅 빈 거리에 내 발소리만
    끝없는 길로 나를 데려가네

    [Chorus]
    깊은 밤 외로움이
    내 가슴을 또 울리네
    눈물에 젖은 이 마음
    아무도 몰라줄 사랑이네

    [Bridge]
    달에게 속삭여본다
    이 아픔을 누가 알까
    눈 감으면 사라질까
    끝나지 않는 이 노래

    [Verse 3]
    새벽이 와도 잠들지 못해
    꿈속에서도 너를 찾아
    바람결에 실려온 목소리
    다시 나를 흔들어 깨우네

    [Chorus]
    깊은 밤 외로움이
    내 가슴을 또 울리네
    눈물에 젖은 이 마음
    아무도 몰라줄 사랑이네
    """

    question_prompt = PromptTemplate(input_variables=["slot", "history"], template=making_lyrics_prompt + "\n" + "Chat history: {history}\n" + "slot: {slot}")

    question_chain = question_prompt | llm | StrOutputParser()
    response = question_chain.invoke({"history": history, "slot":slot})

    memory.save_context({"input": user_input}, {"output": response})

    data['gen_lyrics']=response
    data['option']='option4'
    return response, data, flag

    
def high_lyrics_change(user_input, data, llm) -> str:
    flag=0
    change_lyrics_prompt="""
    [Lyrics Change Task]
    사용자가 요구하는 지시에 따라 가사를 변경하세요.

    {user_input}

    [유의사항]
    - 되도록 변경할 가사 부분과 글자수를 맞추어 변경하세요.
    - 오로지 변경한 가사 부분의 텍스트만 output으로 출력합니다.
    - 전체 가사는 다음과 같습니다. 어색하지 않게 생성하세요. 

    전체가사:
    {total_lyrics}
    """

    change_prompt = PromptTemplate(
    template=change_lyrics_prompt,
    input_variables=["user_input","total_lyrics"]
    )

    change_lyrics_chain = change_prompt | llm | StrOutputParser()

    response = change_lyrics_chain.invoke({"user_input": user_input, "total_lyrics":data['gen_lyrics']})

    return response, data, flag