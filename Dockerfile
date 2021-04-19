FROM python:3.7.3-slim-stretch
RUN mkdir /app
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000
CMD ["uvicorn", "server:app", "--port", "5000", "--host","0.0.0.0"]



