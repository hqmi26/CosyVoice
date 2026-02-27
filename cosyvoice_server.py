"""
CosyVoice API Server for AI Waifu
Runs on port 9880, accepts TTS requests with voice cloning.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'third_party/Matcha-TTS')

import io
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()

# Load model globally
print("Loading CosyVoice2-0.5B model...")
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
)
print("Model loaded!")


class TTSRequest(BaseModel):
    text: str
    ref_audio_path: str = ""
    ref_text: str = ""
    speed: float = 1.0


@app.post("/tts")
async def tts(req: TTSRequest):
    """Generate speech with voice cloning from reference audio."""
    try:
        # Load reference audio
        ref_audio = load_wav(req.ref_audio_path, 16000)

        # Use cross-lingual mode (JP ref -> EN output)
        output_list = []
        for result in cosyvoice.inference_cross_lingual(
            req.text,
            ref_audio,
            source_language='<|en|>',
            speed=req.speed,
        ):
            output_list.append(result['tts_speech'])

        if output_list:
            audio = torch.concat(output_list, dim=1)

            # Convert to WAV bytes
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio, 22050, format="wav")
            buffer.seek(0)

            return Response(
                content=buffer.read(),
                media_type="audio/wav"
            )
        else:
            return Response(content="No audio generated", status_code=500)

    except Exception as e:
        print(f"TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=str(e), status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "CosyVoice2-0.5B"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9880)
