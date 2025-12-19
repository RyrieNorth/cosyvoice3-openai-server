from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://10.189.3.18:50000/v1", api_key="123")

speech_file_path = Path(__file__).parent / "speech.wav"

with client.audio.speech.with_streaming_response.create(
    model="CosyVoice3",
    voice="Theresa_zh",
    input="我还记得这间会议室，这是专门为特雷西娅空着的位置么？",
    instructions="",
    response_format="wav",
    speed=1.0,
) as response:
    response.stream_to_file(speech_file_path)
