FROM python:3.11-slim


WORKDIR /app


COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt


EXPOSE 6789

# CMD ["python3", "app.py", "--host", "0.0.0.0", "--port", "6789"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]