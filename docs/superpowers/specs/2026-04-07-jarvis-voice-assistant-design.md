# Jarvis — Assistente de Voz Hibrido (Local + Minimax API)

## Resumo

Assistente de voz pessoal que escuta passivamente via wake word ("Jarvis") ou hotkey, transcreve fala localmente com faster-whisper (GPU), envia o texto para a Minimax API (streaming) e sintetiza a resposta em voz com Piper TTS (CPU). Foco em ciencia, astronomia e fatos do mundo. Respostas em portugues brasileiro, frases curtas e naturais.

## Hardware e Ambiente

- GPU: NVIDIA RTX 4070 12GB VRAM
- OS: Windows 11 Pro
- Python: 3.12.4
- CUDA: disponivel (LM Studio e Ollama ja funcionam com GPU)
- Budget: $20 credito Minimax Token Plan, zero gasto adicional

## Arquitetura — Pipeline com Streaming

```
STANDBY (wake word + hotkey)
    |
    v
LISTENING (silero-vad detecta fala, grava ate silencio)
    |
    v
STT (faster-whisper large-v3, GPU, language="pt")
    |
    v
MINIMAX API (streaming SSE, token a token)
    |
    v
TTS + PLAYBACK (Piper sintetiza frase a frase, reproduz incrementalmente)
    |
    v
VOLTA AO STANDBY
```

A chave da baixa latencia esta no streaming: enquanto a Minimax ainda gera tokens, o Piper ja sintetiza e reproduz as primeiras frases. Latencia estimada ate primeira palavra falada: ~1-2s.

## Estrutura de Arquivos

```
projeto-jarvis/
├── jarvis/
│   ├── __init__.py
│   ├── main.py              # Loop principal async, orquestra tudo
│   ├── audio_capture.py     # Captura do microfone + VAD (silero-vad)
│   ├── wake_word.py         # Deteccao de "Jarvis" (openwakeword)
│   ├── stt.py               # Speech-to-Text (faster-whisper, GPU)
│   ├── llm.py               # Cliente Minimax API (streaming)
│   ├── tts.py               # Text-to-Speech (Piper TTS, CPU)
│   ├── audio_player.py      # Reproducao de audio (sounddevice)
│   ├── hotkey.py            # Listener de tecla de atalho (pynput)
│   ├── conversation.py      # Historico de conversa (ultimas 5 trocas)
│   └── config.py            # Configuracoes centralizadas via .env
├── logs/                    # token_usage.jsonl, errors.log
├── .env                     # Chaves e parametros (NUNCA commitado)
├── .gitignore
├── requirements.txt
└── start_jarvis.bat         # Script de inicializacao Windows
```

## Stack de Tecnologias

| Componente | Biblioteca | Recurso | VRAM |
|---|---|---|---|
| Wake word | openwakeword | CPU | 0 |
| VAD | silero-vad | CPU | 0 |
| Hotkey | pynput | CPU | 0 |
| STT | faster-whisper (large-v3, int8) | GPU | ~1.5GB |
| LLM | Minimax API (Token Plan) | Cloud | 0 |
| TTS | Piper TTS (pt_BR-faber-medium) | CPU | 0 |
| Audio I/O | sounddevice + numpy | CPU | 0 |
| **Total VRAM** | | | **~1.5GB** |

## Detalhamento dos Componentes

### 1. Wake Word — openwakeword

- Modelo pre-treinado para "Jarvis" (disponivel na comunidade openwakeword)
- Roda em CPU com ~1% de uso constante
- Escuta o microfone em loop, quando detecta a palavra ativa o modo LISTENING
- Threshold configuravel para evitar falsos positivos

### 2. Hotkey — pynput

- Combinacao padrao: `Ctrl+Alt+J` (configuravel no .env)
- Funciona com Jarvis em background
- Ativa o mesmo fluxo que o wake word
- Alternativa para ambientes ruidosos onde wake word pode falhar

### 3. Audio Capture + VAD — silero-vad + sounddevice

- sounddevice captura audio do microfone em chunks de 30ms
- silero-vad classifica cada chunk como fala ou silencio
- Inicia gravacao quando detecta fala
- Encerra gravacao apos 1.5s de silencio continuo
- Retorna o audio completo como numpy array (16kHz, mono)

### 4. STT — faster-whisper

- Modelo: `large-v3` com quantizacao int8 (CTranslate2)
- Device: CUDA (RTX 4070)
- Parametros fixos: `language="pt"`, `beam_size=3`
- Modelo carregado uma vez na inicializacao, fica residente na GPU
- Performance: ~10s de audio transcrito em <1s
- VRAM: ~1.5GB

### 5. LLM — Minimax API (Token Plan)

- Endpoint: `https://api.minimax.chat/v1/text/chatcompletion_v2`
- Autenticacao: `Authorization: Bearer <MINIMAX_API_KEY>`
- Streaming via SSE (Server-Sent Events)
- Modelo: a ser confirmado na documentacao do Token Plan na implementacao (candidato: abab6.5s-chat ou equivalente disponivel)
- Historico enviado a cada request (system + ultimas 5 trocas)
- Estimativa de consumo: ~700-1000 tokens por interacao
- Budget estimado: ~2000-4000 interacoes com $20

### 6. TTS — Piper

- Voz: `pt_BR-faber-medium` (masculina, qualidade media-alta)
- Roda como processo CLI, chamado por frase
- Latencia: ~150ms por frase
- Estrategia de streaming:
  - Acumula texto recebido da API ate detectar fim de frase (`.` `!` `?`)
  - Sintetiza a frase completa em audio WAV
  - Reproduz imediatamente via sounddevice
  - Proxima frase sintetiza em paralelo enquanto a anterior toca

### 7. Conversacao — Historico com Janela Deslizante

- System prompt fixo (~80 tokens)
- Janela de 5 trocas (user + assistant)
- Quando excede 5, remove a troca mais antiga (FIFO)
- Sem compressao/resumo por simplicidade

## System Prompt

```
Voce e Jarvis, um assistente pessoal inteligente e prestativo.
Responda sempre em portugues brasileiro.
Seja conciso: frases curtas e diretas, adequadas para fala.
Evite listas, markdown, URLs ou formatacao visual.
Quando nao souber algo, diga honestamente.
Seu tom e educado, levemente formal e com um toque de humor sutil.
Areas de interesse do usuario: ciencia, astronomia, fatos curiosos.
```

## Configuracao (.env)

```env
MINIMAX_API_KEY=<sua_chave_aqui>
MINIMAX_MODEL=abab6.5s-chat
HOTKEY=ctrl+alt+j
WAKE_WORD=jarvis
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
PIPER_VOICE=pt_BR-faber-medium
MAX_HISTORY=5
LOG_TOKENS=true
```

## Tratamento de Erros

| Situacao | Comportamento |
|---|---|
| Minimax API offline/timeout (>10s) | Jarvis fala: "Nao consegui me conectar. Tente novamente em instantes." |
| Rate limit (429) | Espera 5s, tenta uma vez. Se falhar: avisa o usuario |
| Credito esgotado (402/403) | Jarvis fala: "Meu credito de API acabou. Estou offline por enquanto." |
| Microfone nao detectado | Log de erro + mensagem no console na inicializacao |
| STT transcreve vazio/ruido | Ignora silenciosamente, volta ao standby |
| TTS falha ao sintetizar | Loga o erro, pula a frase, tenta a proxima |
| Excecao nao tratada | Log em logs/errors.log, Jarvis continua rodando |

## Monitoramento de Tokens

Cada interacao salva em `logs/token_usage.jsonl`:

```json
{"timestamp": "2026-04-07T14:32:00", "input_tokens": 450, "output_tokens": 120, "total": 570, "cumulative": 12340}
```

Na inicializacao, Jarvis calcula credito restante estimado. Abaixo de 10% do budget: avisa "Atencao, meu credito esta ficando baixo."

## Inicializacao Automatica (Windows)

- `start_jarvis.bat` ativa o venv e executa `python -m jarvis.main`
- Atalho colocado em `shell:startup` para iniciar com o Windows
- Inicia minimizado no console
- Futuro: pode migrar para system tray com pystray

## Decisoes Explicitas

- **Sem fallback LLM local por agora**: se a API falhar, Jarvis avisa e para. Fallback via Ollama pode ser adicionado futuramente.
- **Idioma fixo em PT-BR**: sem deteccao automatica de idioma, simplifica STT e prompts.
- **Piper em vez de XTTS**: troca qualidade de voz (nao soa como Jarvis do filme) por zero VRAM e latencia muito baixa.
- **Whisper large-v3 int8**: melhor precisao em portugues com VRAM razoavel (~1.5GB). Alternativa: medium (~0.8GB) se precisar liberar VRAM.
- **Console em vez de system tray**: simplicidade agora, upgrade futuro facil.

## Escopo Futuro (Nao Implementar Agora)

- Comandos de OS (abrir Spotify, criar pasta, etc.)
- Fallback LLM local via Ollama
- System tray com icone e controles visuais
- Clonagem de voz estilo Jarvis com XTTS
- Interface web para configuracao
