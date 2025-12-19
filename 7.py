import requests
import pyaudio
import wave

url = "http://10.189.3.18:50000/v1/audio/speech"

payload = {
    "model": "CosyVoice3",
    "voice": "Theresa_zh",
    "input": "这是一个真正的PCM流式测试。我虽不知归途何处，但心中的明镜，早已映出前行的方向。我愿感受你晨晖洒落，双手紧拥，风儿轻吻苍穹时的暖和。",
    "stream": True,
}

headers = {"Content-Type": "application/json", "Authorization": "Bearer 123"}

pcm_data = bytearray()

# 初始化 PyAudio
p = pyaudio.PyAudio()

try:
    with requests.post(url, json=payload, headers=headers, stream=True) as r:
        r.raise_for_status()

        # 从响应头获取音频参数，如果没有则使用默认值
        sample_rate = int(r.headers.get("X-Sample-Rate", 24000))
        channels = int(r.headers.get("X-Channels", 1))
        bit_depth = int(r.headers.get("X-Bit-Depth", 16))

        # 打开音频输出流
        output_stream = p.open(
            format=p.get_format_from_width(bit_depth // 8),
            channels=channels,
            rate=sample_rate,
            output=True,
        )

        print("正在同步接收并播放...")

        # 边接收边播放
        for chunk in r.iter_content(
            chunk_size=1024
        ):  # 较小的 chunk 可以降低首字响应延迟
            if chunk:
                output_stream.write(chunk)
                pcm_data.extend(chunk)

        print("播放完成。")

finally:
    # 清理资源
    if "output_stream" in locals():
        output_stream.stop_stream()
        output_stream.close()
    p.terminate()


# 手动封装 WAV
with wave.open("test_pcm_stream.wav", "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(bit_depth // 8)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)

print("PCM 流保存完成：test_pcm_stream.wav")
