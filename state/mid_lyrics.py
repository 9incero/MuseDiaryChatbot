from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


#user_input을 바꿀가사부분 -> 어떻게 바꿀지에 대한 지시에 대한 내용으로
def mid_lyrics_change(user_input, data, llm) -> str:
    #이거 바꾸기 0으로
    flag=0
    option=None
    if data['turn']==0:
        response=''
        option='option4'
        return response, data, flag
    
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


    response = change_lyrics_chain.invoke({"user_input": user_input, "total_lyrics":data['lyrics']})

    return response, data, flag, option