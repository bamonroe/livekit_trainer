# Wake-word scoring service. Bases on the trainer image so the mel/embedding
# frontends and onnxruntime match training exactly. Adds only the web layer.
FROM livekit-wakeword-trainer:latest

RUN pip install --no-cache-dir flask==3.0.*

WORKDIR /srv
COPY scorer.py app.py ./

ENV MODELS_DIR=/output PORT=8770
EXPOSE 8770
CMD ["python", "app.py"]
