# Обработка видео

### Этапы

0. ``src/env.py`` - задать вручную глобальные переменные.
1. ``python3 src/face_detector.py`` - чтение кадров из видео и детекция лиц. 
2. ``python3 src/facenet_embeddings.py`` - расчет эмбеддингов на основе facenet.
3. ``python3 src/cluster_faces.py`` - пред-обработка лиц, кластеризация и 
пост-обработка кластеров
4. ``python3 src/write_results.py`` - запись результатов кластеризации в бд.
5. ``python3 src/make_video.py`` - создание демо-видео.

### Docker

- собирается образ ``docker build -t people-count:project .``
- запускается по крону (crontab-jobs.txt) основной скрипт ``run.sh``.
- команда запуска ``docker run -d --name project_crontime --runtime nvidia
    -v /mnt:/mnt --restart always people-count:project``

### Алгоритмы

1. Детекция:
    - [pytorc_mtcnn](https://github.com/Seanlinx/mtcnn), на уровне 20-25 FPS,
    наивысшая точность. Предпочтительный алгоритм

2. Внутреннее представление лиц:
    - **[facenet](https://github.com/davidsandberg/facenet)**, хорошая точность
    
3. Пред-обработка лица:
    - выбрать лица выше порога по ширине и высоте
    - выбрать фронтальные лица (5 точек mtcnn)
    - удалить лица с слишком большой или маленькой дисперсией по лапласиану

4. Кластеризация:
    - наилучший вариант **dlib.chinese_whispers** с порогом 1.0 facenet и 0.45 dlib.
    
5. Пост-обработка:
    - выбрать кластеры с высоким **silhouette score** (шум)
    - отсечь кластеры с кол-вом членов менее порога (шум)
    - отсечь кластеры с длительностью  меньше порога (прохожие)
    - отсечь кластера с частотой выше порога (работники)
    
    
### Возможные улучшения

1. Быстрая детекция лиц:
    - [fast](https://www.groundai.com/project/a-fast-face-detection-method-via-convolutional-neural-network/1)
    - [newest](https://arxiv.org/pdf/1904.12094.pdf)
    - [russian](https://arxiv.org/ftp/arxiv/papers/1508/1508.01292.pdf)
    
2. Кластеризация:
    - [DDC](https://arxiv.org/pdf/1706.05067.pdf)
    - [debacl](https://blog.dominodatalab.com/topology-and-density-based-clustering/)
    - [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
    
3. Другие улучшения:
    - [Фронтализация лиц](https://github.com/dougsouza/face-frontalization)
    - [детекция обоих глаз для выделения фронтальных лиц](https://www.ecse.rpi.edu/~qji/Papers/frgc_eye.pdf)
    - другой алгоритм детекции блюра. Для людей в очках слишком высокие значения. Ссылки внутри рабочей тетрадки.
    - убрать прикрытые лица (обычно чужой головой)
    - dlib.chinese_clustering ->  chinese clustering with cosine distance + networkX
    
### Что уже опробовано

- **dlib cnn detecotor**, на уровне 30 FPS, точность ниже, чем у mtcnn, 
    в особенности по ключевым точкам.
- [mobilenet_ssd](https://github.com/yeephycho/tensorflow-face-detection)
    на уровне 60-80 FPS, неточная детекция, слишком большие кропы
- **dlib resnet face recognition (+ shape predictor)**, точность достаточно
    хорошая, показывает себя хуже при неполном лице, размытом изображении
    или боковых лиц (высокий FAR)
- **иерархическая кластеризация** с предложенным порогом 0.6 от авторов dlib. 
    Работает неплохо, но из-за высокого FAR на плохих изображениях, много неточностей.
    Подходит как запасной вариант
- **sklearn dbscan** отсекает множество изображений как шум.