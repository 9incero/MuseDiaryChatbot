from .state.default import default
from .state.mid_lyrics import mid_lyrics_change
from .state.mid_music import mid_music_selection
from .state.high_lyrics import high_lyrics_concept
from .state.high_music import high_music_selection
from .state.discussion import discussion
from typing import Tuple
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# PATH[energy][want_lyrics]
PATH={
    'low':{ 
        True:['default'], 
        False:['default']},
    'mid':{
        True:['default','mid_lyrics','mid_music','discussion'],
        False:['default','mid_music', 'discussion']},
    'high':{
        True: ['default','high_lyrics','high_music','discussion'],
        False: ['default','high_music','discussion']},
    }

STATE_FUNCTION={
    'default': default,
    'mid_lyrics':mid_lyrics_change,
    'mid_music': mid_music_selection,
    'high_lyrics': high_lyrics_concept,
    'high_music': high_music_selection,
    'discussion': discussion
}

def execute_state(
    user_input: str,
    data: dict
) -> Tuple[str, dict]:
    
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    if data['state'] is None:
        data['state']='default'

    state=data['state']
    state_func=STATE_FUNCTION[state]
    path_list=PATH[data['energy']][data['want_lyrics']]

    response, data, flag = state_func(user_input, data, llm)

    data['turn']+=1


    # state change 감지
    if flag==1:
        idx=path_list.index(data['state'])
        if idx + 1 < len(path_list):
            next_state = path_list[idx + 1]
        else:
            next_state = None 
        data['state']=next_state
        data['turn']=0


    return response, data

