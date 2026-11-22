FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /bot
COPY mechfight.py world.py brain.py bot.py ./
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
