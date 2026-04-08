"""Jarvis — Main async orchestrator."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from jarvis.audio_capture import AudioCapture
from jarvis.audio_player import AudioPlayer
from jarvis.config import Config, load_config
from jarvis.conversation import Conversation
from jarvis.hotkey import HotkeyListener
from jarvis.llm import stream_chat
from jarvis.stt import SpeechToText
from jarvis.token_logger import TokenLogger
from jarvis.tts import PiperTTS, SentenceSplitter
from jarvis.wake_word import WakeWordDetector

SYSTEM_PROMPT = (
    "Voce e Jarvis, assistente de voz pessoal. "
    "Regras obrigatorias: "
    "1) Responda APENAS em portugues brasileiro. "
    "2) Maximo 2-3 frases curtas. Respostas longas sao proibidas. "
    "3) Sem listas, markdown, URLs, emojis ou formatacao. "
    "4) Tom educado e direto, como uma conversa natural. "
    "5) Se nao souber, diga brevemente. "
    "Interesses do usuario: ciencia, astronomia, fatos curiosos."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/jarvis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("jarvis")


class Jarvis:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.conversation = Conversation(SYSTEM_PROMPT, max_turns=config.max_history)
        self.token_logger = TokenLogger("logs/token_usage.jsonl") if config.log_tokens else None

        log.info("Loading STT model (%s on %s)...", config.whisper_model, config.whisper_device)
        self.stt = SpeechToText(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

        piper_model_path = str(Path(config.piper_model_dir) / f"{config.piper_voice}.onnx")
        log.info("Loading TTS model (%s)...", config.piper_voice)
        self.tts = PiperTTS(piper_model_path, piper_dir=config.piper_model_dir)

        log.info("Loading audio capture with VAD...")
        self.audio_capture = AudioCapture(
            vad_threshold=config.vad_threshold,
            silence_duration_ms=config.silence_duration_ms,
        )

        self.player = AudioPlayer(sample_rate=self.tts.sample_rate)

        log.info("Loading wake word detector (%s)...", config.wake_word)
        self.wake_detector = WakeWordDetector(
            model_name=config.wake_word,
            threshold=config.wake_threshold,
        )

        self.hotkey: HotkeyListener | None = None

    async def run(self) -> None:
        """Main loop: listen, transcribe, chat, speak, repeat."""
        loop = asyncio.get_event_loop()

        self.hotkey = HotkeyListener(self.cfg.hotkey, loop)
        self.hotkey.start()
        log.info("Hotkey listener started: %s", self.cfg.hotkey)

        log.info("Jarvis pronto! Diga '%s' ou pressione %s.", self.cfg.wake_word, self.cfg.hotkey)

        await self._speak("Jarvis online. Como posso ajudar?")

        try:
            while True:
                activated = await self._wait_for_activation(loop)
                if not activated:
                    continue

                log.info("Ativado! Ouvindo...")

                audio = await loop.run_in_executor(None, self.audio_capture.record_until_silence)
                if audio is None:
                    log.info("Nenhuma fala detectada, voltando ao standby.")
                    continue

                log.info("Transcrevendo...")
                text = await loop.run_in_executor(None, self.stt.transcribe, audio)
                if not text:
                    log.info("Transcricao vazia, voltando ao standby.")
                    continue

                log.info("Usuario: %s", text)

                await self._chat_and_speak(text)

        except KeyboardInterrupt:
            log.info("Encerrando Jarvis...")
        finally:
            if self.hotkey:
                self.hotkey.stop()

    async def _wait_for_activation(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Wait for wake word or hotkey. Returns True when activated."""
        wake_future = loop.run_in_executor(None, self.wake_detector.listen_blocking)
        hotkey_task = asyncio.create_task(self.hotkey.wait())

        done, pending = await asyncio.wait(
            [wake_future, hotkey_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self.wake_detector.reset()
        return True

    async def _chat_and_speak(self, user_text: str) -> None:
        """Send text to Minimax API (streaming), synthesize and play response."""
        messages = self.conversation.get_messages_for_api(user_text)

        splitter = SentenceSplitter()
        self.player.start()
        full_response = []
        usage = None

        try:
            async for content, chunk_usage in stream_chat(
                api_key=self.cfg.minimax_api_key,
                messages=messages,
                model=self.cfg.minimax_model,
            ):
                if chunk_usage:
                    usage = chunk_usage

                if content:
                    full_response.append(content)
                    for char in content:
                        sentence = splitter.feed(char)
                        if sentence:
                            audio = self.tts.synthesize(sentence)
                            self.player.enqueue(audio)

            for sentence in splitter.flush():
                if sentence:
                    audio = self.tts.synthesize(sentence)
                    self.player.enqueue(audio)

        except Exception as e:
            log.error("Erro na API Minimax: %s", e)
            error_audio = self.tts.synthesize(
                "Nao consegui me conectar. Tente novamente em instantes."
            )
            self.player.enqueue(error_audio)

        self.player.finish()

        response_text = "".join(full_response).strip()
        if response_text:
            self.conversation.add_turn(user_text, response_text)
            log.info("Jarvis: %s", response_text)

        if usage and self.token_logger:
            self.token_logger.log(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
            log.info("Tokens: %s (acumulado: %d)", usage, self.token_logger.cumulative)

    async def _speak(self, text: str) -> None:
        """Synthesize and play a simple text message."""
        self.player.start()
        audio = self.tts.synthesize(text)
        self.player.enqueue(audio)
        self.player.finish()


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    config = load_config()
    jarvis = Jarvis(config)
    asyncio.run(jarvis.run())


if __name__ == "__main__":
    main()
