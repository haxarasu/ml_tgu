# импорты 
from __future__ import annotations # чтобы можно было писать что возвращает метод класса
from typing import Any # штука для типизации кода (используется чтобы сказать: "здесь может быть любой тип данных")
import pandas as pd # для табличек


class SurfaceDataLoader:
    """
    класс для создания датафреймов с табличными данными по трём поверхностям
    (зелёная, серая и стол) и присвоения им меток (label).

    лейблы:
        1 — зелёные (green_surface.csv)
        2 — серая   (gray_surface.csv)
        3 — стол    (table_surface.csv)

    как используется (пример):
        loader = SurfaceDataLoader(base_path="../datasets")
        green_data, gray_data, table_data = loader.load_labeled_frames()
        full_data = loader.load_full_dataset()
    """

    def __init__(self, base_path: str = "../datasets") -> None:
        """
        путь к папке с файлами .csv.
        значение "../datasets" означает:
        - ..  -> подняться на одну папку выше относительно текущего файла
        - затем зайти в папку datasets
        """
        self.base_path = base_path

        # имена файлов с данными по каждой поверхности
        self.gray_filename = "gray_surface.csv"
        self.green_filename = "green_surface.csv"
        self.table_filename = "table_surface.csv"


    def _read_csv(self, filename: str) -> pd.DataFrame:
        """
        внутренний метод для чтения одного CSV-файла.

        параметры
        filename : str
            имя файла (например, "gray_surface.csv").

        возвращает
        pd.DataFrame
            табличка с данными, считанными из CSV.
        """
       
        # собирает полный путь к файлу как строку: base_path + "/" + имя файла
        filepath = f"{self.base_path.rstrip('/')}/{filename}"

        """
        создание датафреймов с табличными данными
        первый аргумент функции - путь к файлу
        второй - разделитель (в нашем случае данные разделены ;)
        третий - задает header = None (говорим что осознанно не используем первую строку как заголовок)
        четвертый - отключение умного чтения по кускам чтобы не выкидывало Warning по поводу разных типов данных в одной колонке
        """

        df = pd.read_csv(
            filepath,
            sep=";",         
            header=None,      
            low_memory=False 
        )

        return df


    def _slice_numeric_part(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        оставить только числовую часть таблицы (начиная с 4-й строки).

        в исходных файлах первые три строки — это "шапка":
        числовые значения, на которых будут обучаться алгоритмы, начинаются с 4-й строки.

        .iloc[3:] — берём все строки, начиная с 3-й (четвёртая по счёту)
        .reset_index(drop=True) — пересоздаём индексы (0,1,2,...) после обрезки.
        """

        numeric_df = df.iloc[3:].reset_index(drop=True)

        return numeric_df


    def load_labeled_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        загружает три таблицы (зелёная, серая, стол), оставить только числовые
        части и добавляет столбец label с нужными значениями.

        возвращает
        (green_data, gray_data, table_data) : tuple из трёх df
            каждый df содержит:
            - только числовые строки
            - дополнительный столбец 'label'
        """
        # читает сырые данные из CSV
        gray_df = self._read_csv(self.gray_filename)
        green_df = self._read_csv(self.green_filename)
        table_df = self._read_csv(self.table_filename)

        # оставляет только числовую часть
        green_data = self._slice_numeric_part(green_df)
        gray_data = self._slice_numeric_part(gray_df)
        table_data = self._slice_numeric_part(table_df)

        # создает новый столбец в каждом df и присваивает ему константу
        green_data["label"] = 1  # все строки из green — класс 1 (зелёная поверхность)
        gray_data["label"] = 2   # все строки из gray  — класс 2 (серая поверхность)
        table_data["label"] = 3  # все строки из table — класс 3 (стол)

        return green_data, gray_data, table_data


    def load_full_dataset(self) -> pd.DataFrame:
        """
        объединяет три размеченные таблицы (зелёная, серая, стол) в один общий датафрейм.

        возвращает

        pd.DataFrame
            один общий датафрейм, содержащий:
            - все строки с трёх поверхностей
            - столбец 'label' с метками классов (1, 2, 3)
        """
        green_data, gray_data, table_data = self.load_labeled_frames()

        # объединяет три датафрейма в один
        full_data = pd.concat(
            [green_data, gray_data, table_data],
            ignore_index=True  # пересобирает индексы
        )

        return full_data


class DataCleaner:
    """
    missing_strategy : str, по умолчанию "drop"
        способ обработки пропусков:
        - "drop" — удалить строки с пропусками (DataFrame.dropna);
        - "fill" — заполнить пропуски фиксированным значением.
    fill_value : Any, по умолчанию 0
        значение, которым будут заполняться пропуски, если выбрана стратегия
        "fill". может быть числом, строкой и т.п.

    как используется: (пример)
    cleaner = DataCleaner(missing_strategy="fill", fill_value=0)
    df_clean = cleaner.clean(df)
    """

    def __init__(self, missing_strategy: str = "drop", fill_value: Any = None) -> None:
        # сохраняет выбранную стратегию в атрибуте объекта
        self.missing_strategy = missing_strategy
        # значение для заполнения пропусков (используется только при стратегии "fill")
        if self.missing_strategy == 'fill':
            self.fill_value = fill_value
            if fill_value == None:
                raise ValueError("Задайте значение для заполнения при использовании fill")

        # проверка, что передано допустимое значение стратегии
        allowed_strategies = ("drop", "fill")
        if self.missing_strategy not in allowed_strategies:
            raise ValueError(
                "Некорректная стратегия обработки пропусков: "
                f"{self.missing_strategy!r}. Допустимые варианты: {allowed_strategies}."
            )

    
    # по факту не нужен но можете перед преподом выебнуться что еще и это сделали
    def describe_missing(self, df: pd.DataFrame) -> pd.Series:
        """
        считает количество пропусков в каждом столбце.

        параметры
        df : pd.DataFrame
            таблица с данными, в которой нужно проверить пропуски.

        возвращает
        pd.Series
            Series, где индекс — названия столбцов, а значения — количество
            пропусков в каждом столбце.
        """

        # isna() возвращает таблицу True/False, где True — это пропуск
        # sum() по столбцам считает количество True, т.е. количество пропусков
        missing_counts = df.isna().sum()

        return missing_counts


    def has_missing(self, df: pd.DataFrame) -> bool:
        """
        проверяет, есть ли в таблице хотя бы один пропуск.

        возвращает True, если в df есть хотя бы одно значение NaN, иначе False.
        """

        # isna().any().any():
        #   - первый any() проверяет по каждому столбцу, есть ли в нём пропуск;
        #   - второй any() проверяет, есть ли хотя бы один столбец с пропусками.

        return df.isna().any().any()


    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        очищает таблицу от пропусков согласно выбранной стратегии.

        метод не изменяет исходный DataFrame, а возвращает его копию с применёнными преобразованиями.

        параметры
        df : pd.DataFrame
            исходная таблица с данными.

        возвращает
        pd.DataFrame
            новая таблица с обработанными пропусками.
        """

        # работает с копией, чтобы не трогать исходные данные "по ссылке".
        result = df.copy()

        if self.missing_strategy == "drop":
            # удаляет все строки, в которых есть хотя бы один пропуск.
            # inplace=False (по умолчанию) — возвращает новый DataFrame.
            result = result.dropna()
        else: 
            # заполняет все пропуски фиксированным значением.
            result = result.fillna(self.fill_value)

        return result


    def select_training_columns(
        self,
        df: pd.DataFrame,
        target_column: str = "label",
        drop_constant: bool = True,
    ) -> pd.DataFrame:
        """
        выбирает столбцы для обучения модели.

        параметры
        df : pd.DataFrame
            таблица с данными (признаки + целевой столбец).
        target_column : str, по умолчанию "label"
            имя столбца с целевой переменной, который исключает из признаков.
        drop_constant : bool, по умолчанию True
            если True, выкидывает столбцы, у которых 0 или 1 уникальное значение.

        возвращает
        pd.DataFrame
            таблица, содержащая только данные для обучения.
        """
        # работает с копией, чтобы не трогать исходные данные
        df_copy = df.copy()

        # если указан столбец с целевой переменной и он присутствует в данных — исключает его
        if target_column in df_copy.columns:
            df_copy = df_copy.drop(columns=[target_column])

        # приводит все оставшиеся признаки к типу float64
        features = df_copy.astype("float64")

        if drop_constant:
            # оставляет только признаки, у которых больше одного уникального значения
            features = features.loc[:, features.nunique() > 1]

        return features