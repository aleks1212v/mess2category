FROM python:3.6-stretch
MAINTAINER vitomskov <vitomskov@mail.ru>

# устанавливаем параметры сборки
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# проверяем окружение python 
RUN python3 --version
RUN pip3 --version

# set a directory for the app
WORKDIR /usr/src/shell_m2c/src

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all the files to the container
COPY /src/ .

# tell the port number the container should expose
EXPOSE 8501

# run the command
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]
