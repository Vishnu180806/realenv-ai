FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "src/app.py"]
