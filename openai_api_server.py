import os
import io
import sys
import wave
import argparse
import random
import uvicorn
import lameenc
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/third_party/Matcha-TTS".format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed

from pydantic import BaseModel
from typing import List, Optional


# 全局配置
class Config:
    model_dir = "pretrained_models/Fun-CosyVoice3-0.5B"
    fp16 = False
    enable_trt = True
    enable_vllm = True

config = Config()

cosyvoice = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cosyvoice = AutoModel(
        model_dir=config.model_dir,
        fp16=config.fp16,
        load_trt=config.enable_trt,
        load_vllm=config.enable_vllm
    )
    yield
    app.state.cosyvoice = None

app = FastAPI(lifespan=lifespan)


# 跨域解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# 模型结构
class ModelInfo(BaseModel):
  id: str
  object: str = "model"
  owned_by: str
  
class ModelList(BaseModel):
  object: str = "list"
  data: List[ModelInfo]
  

# 请求结构
class TTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    instructions: str = ""
    stream: bool = False
    seed: Optional[int] = None
    speed: float = 1.0
    response_format: str = "pcm"

class SpeakerInfo(BaseModel):
    id: str
    object: str = "speaker"

class SpeakerList(BaseModel):
    object: str = "list"
    data: List[SpeakerInfo]


model_infos = ModelList(
  object="list",
  data = [
    ModelInfo(
      id="CosyVoice3",
      owned_by="Theresa TTS by NorthSky"
    )
  ]
)
AVAILABLE_MODELS = {m.id for m in model_infos.data}

# 将模型输出从numpy数组转成字节
def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


# 接口：获取模型列表、列出所有预训练的说话人
@app.get("/v1/models", response_model=ModelList)
async def list_models():
  return model_infos


# 接口：列出所有预训练的说话人
@app.get("/v1/speakers", response_model=SpeakerList)
async def list_speakers():
    spks = app.state.cosyvoice.list_available_spks() or [""]
    return {
        "object": "list",
        "data": [{"id": s} for s in spks]
    }


# 接口：生成随机数
@app.get("/v1/seed")
async def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "seed": seed
    }


# 接口：TTS生成
@app.post("/v1/audio/speech")
async def inference(req: TTSRequest):
    if not req.input:
        raise HTTPException(400, "文本不能为空")

    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(400, "未知模型")

    if req.voice not in app.state.cosyvoice.list_available_spks():
        raise HTTPException(400, "未知说话人")

    if req.seed is not None:
        set_all_random_seed(req.seed)

    prompt = req.input
    if req.instructions:
        prompt = f"You are a helpful assistant.{req.instructions}<|endofprompt|>{req.input}"
    else:
        prompt = f"You are a helpful assistant.<|endofprompt|>{req.input}"

    if req.stream:
        def generate_stream():
            # 1. 初始化 MP3 编码器
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(24000)  # CosyVoice 24KHz 采样率
            encoder.set_channels(1)
            encoder.set_quality(2)

            try:
                for chunk in app.state.cosyvoice.stream(
                    tts_text=prompt, 
                    spk_id=req.voice, 
                    stream=True,
                    speed=req.speed
                ):
                    audio_tensor = chunk.get('tts_speech')
                    if audio_tensor is None:
                        continue
                    
                    # 转换格式：Tensor -> float32 -> int16
                    float_data = audio_tensor.numpy().flatten()
                    int16_data = (float_data * 32767).clip(-32768, 32767).astype(np.int16)
                    
                    # 编码为 MP3 字节
                    mp3_chunk = encoder.encode(int16_data.tobytes())
                    if mp3_chunk:
                        yield bytes(mp3_chunk)
                
                # 冲刷编码器剩余数据
                last_chunk = encoder.flush()
                if last_chunk:
                    yield bytes(last_chunk)
                    
            except Exception as e:
                print(f"Streaming error: {e}")

        return StreamingResponse(generate_stream(), media_type="audio/mpeg")

    else:
        # 非流式处理逻辑
        model_output = app.state.cosyvoice.inference_sft(
            tts_text=prompt,
            spk_id=req.voice,
            stream=False,
            speed=req.speed
        )
        
        # 提取完整音频 Tensor 并转为 bytes
        all_audio = []
        for chunk in model_output:
            all_audio.append(chunk['tts_speech'].numpy().flatten())
        
        full_audio = np.concatenate(all_audio)
        int16_audio = (full_audio * 32767).clip(-32768, 32767).astype(np.int16)
        
        byte_io = io.BytesIO()
        
        # 根据请求的格式返回
        if req.response_format == "wav":
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(int16_audio.tobytes())
            media_type = "audio/wav"
        else:
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(24000)
            encoder.set_channels(1)
            mp3_data = encoder.encode(int16_audio.tobytes()) + encoder.flush()
            byte_io.write(mp3_data)
            media_type = "audio/mpeg"

        byte_io.seek(0)
        return StreamingResponse(byte_io, media_type=media_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--port",
                        type=int,
                        default=50000,
                        help="API服务启动端口")

    parser.add_argument("--model_dir",
                        type=str,
                        default="pretrained_models/Fun-CosyVoice3-0.5B",
                        help="本地或 HF / ModelScope 模型路径")

    parser.add_argument("--fp16",
                        action="store_true",
                        help="以 fp16 加载模型")

    parser.add_argument("--no-trt",
                        action="store_false",
                        dest="enable_trt",
                        help="禁用 TensorRT")

    parser.add_argument("--no-vllm",
                        action="store_false",
                        dest="enable_vllm",
                        help="禁用 vLLM")

    parser.set_defaults(enable_trt=True, enable_vllm=True)

    args = parser.parse_args()
    
    config.model_dir = args.model_dir
    config.fp16 = args.fp16
    config.enable_trt = args.enable_trt
    config.enable_vllm = args.enable_vllm
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)