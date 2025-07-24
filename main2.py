from .execute_chatbot import execute_state
from langchain.memory import ConversationSummaryMemory 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import json


from dotenv import load_dotenv

load_dotenv()
def main():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    #energy level: low
    # data={
    #     'state':None,
    #     'memory': None,
    #     'music_tag':None,
    #     'listen_lyrics':'',
    #     'turn':0,
    #     'quit_response':0,
    #     'energy':'low',
    #     'gen_lyrics':None,
    #     'want_lyrics':False,
    #     'option':None,
    #     'title':'예뻤어',
    #     'artist':'데이식스',
    #     'impressive_lyrics':None,
        # 'lyrics':None,

    # }

    # #energy level: mid / want lyrics: x
    # data={
    #     'state':None,
    #     'memory': None,
    #     'music_tag':None,
    #     'listen_lyrics':"지금 이 말이 우리가 다시 시작하자는 건 아냐 그저 너의 남아있던 기억들이 떠올랐을 뿐야 정말 하루도 빠짐없이 (너, 너는) 사랑한다 말해줬었지 (ah) 잠들기 전에, 또 눈 뜨자마자 말해주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 너도 이제는 (이젠) 나와의 기억이 추억이 되었을 거야 너에게는 어떤 말을 해도, 다 지나간 일일 거야 (ah) 정말 한 번도 빠짐없이 (너, 너는) 나를 먼저 생각해 줬어 (ah) 아무 일 아니어도, 미안해, 고마워해 주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 아직도 가끔 네 생각이 나, 어렵게 전화를 걸어볼까? 생각이 들 때도 많지만, baby, I know it's already over 아무리 원해도, 너는 이제 이미 끝나버린 지난날의 한 편의 영화였었단 걸 난 알아 마지막, 날 바라봐주던 그 눈빛 잘 지내라던 그 목소리 다-아-아-아, 다-아-아-아 그마저도 내겐 예뻤어, 내게 보여준 눈물까지 너와 가졌던 순간들은 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어",
    #     'turn':0,
    #     'quit_response':0,
    #     'energy':'mid',
    #     'gen_lyrics':None,
    #     'want_lyrics':False,
    #     'option':None,
    #     'title':'예뻤어',
    #     'artist':'데이식스',
    #     'impressive_lyrics':None,
    # }
    #energy level: mid / want lyrics: o
    # data={
    #     'state':None,
    #     'memory': None,
    #     'music_tag':None,
    #     'listen_lyrics':"지금 이 말이 우리가 다시 시작하자는 건 아냐 그저 너의 남아있던 기억들이 떠올랐을 뿐야 정말 하루도 빠짐없이 (너, 너는) 사랑한다 말해줬었지 (ah) 잠들기 전에, 또 눈 뜨자마자 말해주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 너도 이제는 (이젠) 나와의 기억이 추억이 되었을 거야 너에게는 어떤 말을 해도, 다 지나간 일일 거야 (ah) 정말 한 번도 빠짐없이 (너, 너는) 나를 먼저 생각해 줬어 (ah) 아무 일 아니어도, 미안해, 고마워해 주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 아직도 가끔 네 생각이 나, 어렵게 전화를 걸어볼까? 생각이 들 때도 많지만, baby, I know it's already over 아무리 원해도, 너는 이제 이미 끝나버린 지난날의 한 편의 영화였었단 걸 난 알아 마지막, 날 바라봐주던 그 눈빛 잘 지내라던 그 목소리 다-아-아-아, 다-아-아-아 그마저도 내겐 예뻤어, 내게 보여준 눈물까지 너와 가졌던 순간들은 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어",
    #     'turn':0,
    #     'quit_response':0,
    #     'energy':'mid',
    #     'gen_lyrics':None,
    #     'want_lyrics':True,
    #     'option':None,
    #     'title':'예뻤어',
    #     'artist':'데이식스',
    #     'impressive_lyrics':None,
    # }



    #energy level: high / want lyrics: x
    data={
        'state':None,
        'memory': None,
        'music_tag':None,
        'listen_lyrics':"지금 이 말이 우리가 다시 시작하자는 건 아냐 그저 너의 남아있던 기억들이 떠올랐을 뿐야 정말 하루도 빠짐없이 (너, 너는) 사랑한다 말해줬었지 (ah) 잠들기 전에, 또 눈 뜨자마자 말해주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 너도 이제는 (이젠) 나와의 기억이 추억이 되었을 거야 너에게는 어떤 말을 해도, 다 지나간 일일 거야 (ah) 정말 한 번도 빠짐없이 (너, 너는) 나를 먼저 생각해 줬어 (ah) 아무 일 아니어도, 미안해, 고마워해 주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 아직도 가끔 네 생각이 나, 어렵게 전화를 걸어볼까? 생각이 들 때도 많지만, baby, I know it's already over 아무리 원해도, 너는 이제 이미 끝나버린 지난날의 한 편의 영화였었단 걸 난 알아 마지막, 날 바라봐주던 그 눈빛 잘 지내라던 그 목소리 다-아-아-아, 다-아-아-아 그마저도 내겐 예뻤어, 내게 보여준 눈물까지 너와 가졌던 순간들은 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어",
        'turn':0,
        'quit_response':0,
        'energy':'high',
        'gen_lyrics':None,
        'want_lyrics':False,
        'option':None,
        'title':'예뻤어',
        'artist':'데이식스',
        'impressive_lyrics':None,
    }
    # 대화 로그를 저장할 리스트
    transcript = []

    data['memory']=ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=False)
    response, data = execute_state('', data)
    print("AI: ",response)
    print("data: ", data)
    serializable_data = {k: v for k, v in data.items() if k != 'memory'}

    transcript.append({
        'user': '',
        'bot': response,
        'state': data['state'],
        'data': serializable_data,
        # 'history':history
    })

    while 1: 

        memory_vars = data['memory'].load_memory_variables({})
        history = memory_vars.get("history", "")
        print('history\n',history)

        change_prompt = PromptTemplate(
            template='당신은 경미한 우울을 가지고 있는 사람입니다. 오늘 들은 곡 {title} {artist} 에 대해 이야기를 나눕니다. 챗봇의 응답{response}에 대답하세요.',
            input_variables=["response","title","artist","history"]
            )

        change_lyrics_chain = change_prompt | llm | StrOutputParser()

        user_input = change_lyrics_chain.invoke({"title":data['title'],"artist":data['artist'],"response":response,"history":history})

        if data['state']=='mid_lyrics':
            user_input="예뻤어, 날 바라봐주던 그 눈빛을 조금 더 슬프게 바꿔줘"

        
        
        response, data = execute_state(user_input, data)


        print("============")
        print("user: ", user_input)
        print("\nAI: ",response)
        print("\ndata: ", data)
        print("============")
        serializable_data = {k: v for k, v in data.items() if k != 'memory'}

        # 로그에 추가
        transcript.append({
            'user': user_input,
            'bot': response,
            'state': data['state'],
            'data': serializable_data,
            'history':history
        })

        if data['state'] is None:
            with open('high_want_X_conversation_log.json', 'w', encoding='utf-8') as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)

                print("대화가 conversation_log.json에 저장되었습니다.")
            break


        if data['option']=='option6':
            data['impressive_lyrics']="너도 이제는 (이젠) 나와의 기억이 추억이 되었을 거야"

        if data['option']=='option5':
            data['music_tag']='피아노, 드럼, 락'

        if data['option']=='option4':
            data['lyrics']="수정수정 그저 너의 남아있던 기억들이 떠올랐을 뿐야 정말 하루도 빠짐없이 (너, 너는) 사랑한다 말해줬었지 (ah) 잠들기 전에, 또 눈 뜨자마자 말해주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 너도 이제는 (이젠) 나와의 기억이 추억이 되었을 거야 너에게는 어떤 말을 해도, 다 지나간 일일 거야 (ah) 정말 한 번도 빠짐없이 (너, 너는) 나를 먼저 생각해 줬어 (ah) 아무 일 아니어도, 미안해, 고마워해 주던 너 생각이 나 말해보는 거야 예뻤어, 날 바라봐주던 그 눈빛 날 불러주던 그 목소리 다-아-아-아, 다-아-아-아 그 모든 게 내겐 예뻤어, 더 바랄 게 없는 듯한 느낌 오직 너만이 주던 순간들 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어 아직도 가끔 네 생각이 나, 어렵게 전화를 걸어볼까? 생각이 들 때도 많지만, baby, I know it's already over 아무리 원해도, 너는 이제 이미 끝나버린 지난날의 한 편의 영화였었단 걸 난 알아 마지막, 날 바라봐주던 그 눈빛 잘 지내라던 그 목소리 다-아-아-아, 다-아-아-아 그마저도 내겐 예뻤어, 내게 보여준 눈물까지 너와 가졌던 순간들은 다-아-아-아, 다-아-아-아 지났지만, 넌 너무 예뻤어"
            
if __name__ == "__main__":
    main()
