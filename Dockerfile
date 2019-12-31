FROM python:3.7.6-buster
RUN apt-get update && apt-get install -y zsh\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
COPY ./kaggle.json root/.kaggle/kaggle.json
RUN chmod 600 ~/.kaggle/kaggle.json
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
ENV PYTHONPATH='/code'