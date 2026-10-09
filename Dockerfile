FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    LYRISENT_PUBLIC=1

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt gunicorn

COPY --chown=user . .

USER user
EXPOSE 7860

CMD ["gunicorn", "--preload", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "web_app:app"]
