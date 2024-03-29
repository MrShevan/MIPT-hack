# MIPT hackathon

Решение, занявшее первое место в Хакатоне от Ситимобила в задаче Коррекции расчетного времени поездки.

https://vc.ru/citymobil/130020-vy-pribyli-kak-uchastniki-onlayn-hakatona-razrabatyvali-dlya-sitimobil-modeli-prognoza-ozhidaemogo-vremeni-poezdki

### Docker:

Чтобы запустить контейнер необходимо указать в `docker-compose` пути на хосте, где лежит репозиторий (в папку app) и данные (в папку data). И выполнить следующую команду:

```
docker-compose up --build
```

### Структура кода:

Для обучения и теста выполни скрипт:

```angular2
route_features.py

Аргументы:
  train_file [str]: путь до обучающего файла
  test_file [str]: путь до тестового файла
  val_file [str]: путь до валидационного файла
  test_add_file [str]: путь до тестового файла с добавленным route
```

```angular2
baseline.py

Аргументы:
  train_file [str]: путь до обучающего файла
  test_file [str]: путь до тестового файла
  val_file [str]: путь до валидационного файла
  train_val_merge [int]: если параметр 1 то обучается на объединении train и val
                         если 0 то обучается только на train и проверяет метрики на val
```

### Порядок прогона: 

Генерируем гео фичи 
```
python route_features.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --test_add_file /data/test_additional.csv
```

Обучаем на трейне, проверяем на валидации и предиктим на тесте
```
python baseline.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --train_val_merge 0
```

Обучаем на трейне и валидации, предиктим на тесте
```
python baseline.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --train_val_merge 1
```

### Актуальный скор на валидации

```
Validation MAPE:  0.1401547477305674
```
