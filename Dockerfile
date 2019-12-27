FROM python:3.7.6-buster
RUN apt-get update && apt-get install -y zsh\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt