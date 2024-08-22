FROM python:3.11-slim

WORKDIR /app

RUN apt-get update 
RUN apt-get install -y libmagic-dev 
RUN apt-get install -y poppler-utils 
RUN apt-get install -y tesseract-ocr 
RUN apt-get install -y libgl1
RUN apt-get install -y curl

COPY requirements.txt ./
COPY app.py ./
COPY setup_libs.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python setup_libs.py
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]