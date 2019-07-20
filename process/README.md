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
    - **dlib cnn detecotor**, на уровне 30 FPS, точность ниже.
    - [mobilenet_ssd](https://github.com/yeephycho/tensorflow-face-detection)
    на уровне 60-80 FPS, неточная детекция, слишком большие кропы

2. Внутреннее представление лиц:
    - **dlib resnet face recognition (+ shape predictor)**, точность достаточно
    хорошая, показывает себя хуже при неполном лице, размытом изображении
    - **[facenet](https://github.com/davidsandberg/facenet)**, хорошая точность
    
3. Пред-обработка лица:
    - выбрать лица выше порога по ширине и высоте
    - выбрать фронтальные лица (5 точек mtcnn)
    - удалить лица с слишком большой или маленькой дисперсией по лапласиану

4. Кластеризация:
    - наилучший вариант **dlib.chinese_whispers** с порогом 1.0 facenet и 0.45 dlib.
    - запасной вариант **иерархическая кластеризация** с предложенным порогом
    0.6 от авторов dlib.
    
5. Пост-обработка:
    - выбрать кластеры с высоким **silhouette score** (шум)
    - отсечь кластеры с кол-вом членов менее порога (шум)
    - отсечь кластеры с длительностью  меньше порога (прохожие)
    - отсечь кластера с частотой выше порога (работники)