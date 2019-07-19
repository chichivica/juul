FROM nvidia/cuda:10.0-base

USER root

RUN apt-get update && apt-get -y install cron wget git 
RUN export DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && apt-get -y install tzdata
#RUN echo "Europe/Moscow" > /etc/timezone && rm -f /etc/localtime && dpkg-reconfigure -f noninteractive tzdata
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python3.7
RUN apt update && apt -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev curl \
                    && cd /usr/src \
                    && curl -O https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tar.xz \
                    && tar -xf Python-3.7.3.tar.xz \
                    && cd Python-3.7.3 && ./configure --enable-optimizations \
                   && make -j 8 && make altinstall

RUN apt-get install -y python3-pip libglib2.0-0 
RUN apt-get install -y libpq-dev

WORKDIR /root/people_count

COPY requirements.txt .

RUN python3.7 -m pip install -r requirements.txt

# instal dlib
RUN apt-get install -y cmake
RUN python3.7 -m pip install dlib
#RUN apt-get update && apt-get install -y cmake libblas-dev liblapack-dev libatlas-base-dev
#RUN git clone https://github.com/davisking/dlib.git
#WORKDIR dlib
#RUN echo "os.environ['CC']='gcc-6'" >> setup.py
#RUN mkdir build
#WORKDIR build
#RUN cmake .. -DCUDA_HOST_COMPILER=/usr/bin/gcc-6 -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DUSE_F16C=1
#RUN cmake --build . --config Release
#RUN ldconfig
#WORKDIR ..
#RUN python3.7 setup.py install --record files.txt --compiler-flags "-DCUDA_HOST_COMPILER=/usr/bin/gcc-6"
#RUN wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2 && bunzip2 shape_predictor_5_face_landmarks.dat.bz2
#RUN wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 && bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2

# install pytorch
RUN python3.7 -m pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
RUN python3.7 -m pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl

WORKDIR /root/people_count
COPY models ./models
COPY src ./src
COPY run.sh . 
COPY crontab-jobs.txt . 

# install pytorch-mtcnn
WORKDIR src
WORKDIR FaceDetector
RUN export CUDA_HOME=/usr/local/cuda && PATH=${CUDA_HOME}/bin:${PATH}
RUN python3.7 setup.py build_ext --inplace
RUN python3.7 setup.py install

WORKDIR ../..

RUN crontab crontab-jobs.txt

RUN chmod 777 src/face_detector.py
RUN chmod 777 src/facenet_embeddings.py
RUN chmod 777 src/cluster_faces.py
RUN chmod 777 src/write_results.py

CMD ["cron", "-f"]

