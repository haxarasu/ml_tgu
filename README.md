### Структура

ml_tgu/
├── datasets/                 # данные
│   ├── gray_surface.csv      
│   ├── green_surface.csv    
│   └── table_surface.csv   
│
├── src/                      
│   ├── preprocess.py         # загрузка данных, очистка, выбор признаков
│   ├── correlation.py        # корреляционный анализ и тепловая карта
│   ├── pipelines.py          # модели (CatBoost/LightGBM/XGBoost, деревья, logreg, NN, k-means)
│   └── evaluation.py         # оценка моделей (accuracy, precision, f1, ROC-AUC)
│
├── main.ipynb                # основной ноутбук с запуском всех шагов
├── requirements.txt          # зависимости проекта
├── README.md                 # описание проекта и инструкция по запуску
└── .gitignore                # файлы и папки, не попадающие в git

## 1. Создание venv

### В папке с проектом:
на venv с python 3.12  работает, если на компе стоит версия новее, то при создании среды нужно указывать версию старее

```bash
py -m venv venv
```

- `venv` — имя папки с окружением.
---

## 2. Активация окружения

### Через **PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Установка зависимостей

```bash
pip install -r requirements.txt
```
## 4. Запуск

в VS code при просмотре файла .ipynb в правом верхнем углу появляется кнопка выбора виртуальной среды, выбираете свою и запускаете ячейки
