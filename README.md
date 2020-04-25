# MIPT hackathon

Для обучения и теста выполни скрипт:

```angular2
baseline.py

Аргументы:
  train_file [str]:
  test_file [str]: 
  val_file [str]:
  train_val_merge [int]: если параметр 1 то обучается на объединении train и val
                         если 0 то обучается только на train и проверяет метрики на val
```

Запуск
```
python baseline.py --train_file /data/train.csv --val_file /data/validation.csv --test_file /data/test.csv --train_val_merge 0
```