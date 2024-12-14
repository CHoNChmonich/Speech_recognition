from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import os
import wave
import json
import random

app = FastAPI()

# Загрузка модели VOSK (скачайте модель заранее: https://alphacephei.com/vosk/models)
MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Модель VOSK не найдена. Скачайте её по ссылке https://alphacephei.com/vosk/models")

model = Model(MODEL_PATH)


# Функция для анализа поднятого голоса (примитивная, на основе громкости)
def analyze_raised_voice(segment: AudioSegment) -> bool:
    loudness = segment.dBFS  # Уровень громкости в дБ
    return loudness > -20  # Условно: если громкость выше -20 дБ, считаем, что голос повышенный


# Определение пола человека (псевдослучайное, т.к. VOSK этого не поддерживает напрямую)
def detect_gender(text: str) -> str:
    return random.choice(["male", "female"])


# Разделение на стороны (receiver/transmitter)
def assign_side(index: int) -> str:
    return "receiver" if index % 2 == 0 else "transmitter"


@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    # Сохраняем загруженный файл временно
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Конвертация MP3 в WAV
    audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    # Открытие WAV-файла
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        raise ValueError("Аудиофайл должен быть моно, 16 бит, 8к или 16к Hz")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    # Преобразование аудио в текст
    dialog = []
    total_durations = {"receiver": 0, "transmitter": 0}
    index = 0
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result and result["text"]:
                duration = result.get("result")[-1]["end"] if result.get("result") else 0
                segment = audio[index * 1000: (index + 1) * 1000]
                raised_voice = analyze_raised_voice(segment)
                gender = detect_gender(result["text"])
                side = assign_side(index)
                dialog.append({
                    "source": side,
                    "text": result["text"],
                    "duration": duration,
                    "raised_voice": raised_voice,
                    "gender": gender
                })
                total_durations[side] += duration
                index += 1

    # Удаление временных файлов
    os.remove(audio_path)
    os.remove(wav_path)

    # Формирование ответа
    response = {
        "dialog": dialog,
        "result_duration": total_durations
    }
    return JSONResponse(content=response)
