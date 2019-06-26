FROM talalaev_vnc1

USER root

RUN apt-get update && apt-get -y install tzdata
RUN echo "Europe/Moscow" > /etc/timezone && rm -f /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

WORKDIR /root/juul_video_stream

COPY . .

RUN crontab crontab-jobs.txt

RUN chmod 777 video_stream.py

ENV stage docker

CMD ["cron", "-f"]
