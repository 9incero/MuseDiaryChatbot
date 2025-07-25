
import os
import requests
import time
import json
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate



# .env 파일에서 환경 변수 로드
load_dotenv()

# Mureka API 엔드포인트 및 API 키 설정
mureka_api_endpoint = "https://api.mureka.ai"
mureka_api_key = os.getenv("MUREKA_API_KEY")


def query_mureka_task(id: str):
    """지정된 ID의 작업 상태를 Mureka API에 조회합니다."""
    headers = {
        "Authorization": f"Bearer {mureka_api_key}",
    }
    response = requests.get(mureka_api_endpoint + f"/v1/song/query/{id}", headers=headers)
    response.raise_for_status()
    return response.json()


def generate_mureka_song_and_wait(title: str, lyrics: str, music_component: str) -> str:
    """
    Mureka API에 노래 생성을 요청하고, 작업이 완료될 때까지 대기한 후
    오디오 URL을 반환합니다.
    """
    print(f"제목: {title}")
    print(f"음악 스타일: {music_component}")

    # 1. 노래 생성 요청 (POST)
    headers = {"Authorization": f"Bearer {mureka_api_key}", "Content-Type": "application/json"}
    payload = {
        "lyrics": lyrics,
        "model": "auto",
        "prompt": music_component,
    }

    try:
        response = requests.post(mureka_api_endpoint + "/v1/song/generate", headers=headers, json=payload, timeout=(5, 60))
        response.raise_for_status()

        res_data = response.json()
        task_id = res_data.get("id")
        print(f"노래 생성 작업 시작. 작업 ID: {task_id}")

        # 2. 작업 완료까지 대기 (while 루프)
        retry_delay = 5  # 5초마다 상태 확인
        max_retries = 100  # 최대 100번 시도 (약 8분)
        retry_count = 0

        while retry_count < max_retries:
            task_status_response = query_mureka_task(task_id)
            status = task_status_response.get("status")

            if status == "succeeded":
                audio_url = task_status_response["choices"][0]["url"]
                print(f"노래 생성 성공! 오디오 URL: {audio_url}")
                return audio_url
            elif status == "failed":
                print(f"작업 실패: {task_status_response}")
                return "Task failed"
            else:
                # 상태가 'processing' 이거나 다른 상태일 경우
                print(f"작업 진행 중... (상태: {status}). {retry_delay}초 후 다시 시도합니다.")
                time.sleep(retry_delay)
                retry_count += 1

        print("최대 시도 횟수를 초과했습니다. 작업 시간 초과.")
        return "Task timed out"

    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return f"API Error: {e}"


def music_creation(user_input, data, llm, slot):
    option=None
    flag=0
    """
    CombinedSlot(dict) 타입의 user_input에서 가사와 음악 스타일 정보를 추출하여
    Mureka API로 음악을 생성하고, 오디오 URL을 반환합니다.
    """

    # 1. 가사 추출
    # mid + want_lyrics x 일때 기성 가사 그대로 넣고 노래 만듦
    if (data['want_lyrics']==False) and (data['state']=='mid_music'):
        lyrics = data['listen_lyrics']
    else:
        lyrics = data['gen_lyrics']
    if not lyrics:
        response = "가사가 입력되지 않았습니다."
        #여기 가사 없으면 instrumental만 나오는 걸로 바꿔야함
        flag=1
        return response, data, flag, option

    # 2. 음악 스타일 프롬프트 생성
    music_prompt = PromptTemplate(
        input_variables=["music_tag", "add_selected_music_component"],
        template="""Combine the following keywords into a single music generation prompt for an AI. 
    Keep the output as a comma-separated list of concise musical descriptors. 
    Do not write full sentences. 

    Just return a list like:
    keyword1, keyword2, keyword3, ...

    Input:
    {music_tag}, {add_selected_music_component}
    """
    )

    music_chain = music_prompt | llm | StrOutputParser()
    music_component = music_chain.invoke({"music_tag":data['music_tag'],"add_selected_music_component":slot.add_selected_music_component})

    # 3. 제목 추출 (없으면 'Untitled Song')
    title = "Untitled Song"

    # 4. Mureka API 호출 및 결과 반환
    audio_url = generate_mureka_song_and_wait(title, lyrics, music_component)

    if audio_url.startswith("http"):
        response = f"노래가 성공적으로 생성되었습니다!\n오디오 파일: {audio_url}"
        flag=1
    else:
        response = f"노래 생성에 실패했습니다: {audio_url}"

    return response, data, flag, option