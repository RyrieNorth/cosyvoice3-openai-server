import requests
import wave

url = "http://10.189.3.18:50000/v1/audio/speech"

payload = {
    "model": "CosyVoice3",
    "voice": "Theresa_zh",
    "input": "这是一个真正的PCM流式测试。",
    "stream": True,
}

headers = {"Content-Type": "application/json", "Authorization": "Bearer 123"}

pcm_data = bytearray()

with requests.post(url, json=payload, headers=headers, stream=True) as r:
    r.raise_for_status()

    sample_rate = int(r.headers.get("X-Sample-Rate", 24000))
    channels = int(r.headers.get("X-Channels", 1))
    bit_depth = int(r.headers.get("X-Bit-Depth", 16))

    for chunk in r.iter_content(chunk_size=4096):
        if chunk:
            pcm_data.extend(chunk)

# 手动封装 WAV
with wave.open("test_pcm_stream.wav", "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(bit_depth // 8)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)

print("PCM 流保存完成：test_pcm_stream.wav")
