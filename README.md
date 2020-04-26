# MIPT hackathon

Для обучения и теста выполни скрипт:

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
python route_features.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv
```

Обучаем на трейне, проверяем на валидации и предиктим на тесте
```
python baseline.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --train_val_merge 0
```

Обучаем на трейне и валидации, предиктим на тесте
```
python baseline.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --train_val_merge 1
```