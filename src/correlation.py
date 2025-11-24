# импорты
from __future__ import annotations  # как в preprocess.py
from typing import Optional  # как Any в preprocess.py, нужен чтобы ссылаться на опциональные аргументы
import pandas as pd  # ну вы видели уже
import matplotlib.pyplot as plt  # для построения графиков (тепловой карты)


class CorrelationAnalyzer:
    """
    класс для анализа взаимосвязей между параметрами

    - считает матрицу корреляций между признаками;
    - находит пары признаков с сильной (по модулю) корреляцией;
    - выводит тепловую карту (аналог imagesc из matlab) для матрицы корреляций.
    """

    def __init__(self, df: pd.DataFrame, label_column: Optional[str] = "label") -> None:
        """
        сохраняет внутри объекта таблицу с данными и имя столбца с меткой (если он есть).

        параметры
        df : pd.DataFrame
            датафрейм с данными, по которым будет считаться корреляция.
        label_column : Optional[str]
            имя столбца с метками класса ("label").
            если указано, этот столбец будет исключён из корреляционного анализа, так как обычно связи строят только между числовыми признаками.
        """
        # сохраняет исходный датафрейм
        self.df = df

        # запоминает имя столбца с меткой (может быть None)
        self.label_column = label_column


    def _get_numeric_features(self) -> pd.DataFrame:
        """
        возвращает только числовые столбцы датафрейма (без столбца label).

        метод corr у pandas умеет корректно работать только с числами.
        """
        df = self.df # запись в локальную переменную чтобы не изменять self.df

        # если указан столбец с метками и он присутствует в данных — исключает его
        if self.label_column is not None and self.label_column in df.columns:
            df = df.drop(columns=[self.label_column])

        # приводит все оставшиеся признаки к типу float64
        numeric_df = df.astype("float64")
        
        # оставляет только признаки, у которых болльше 1 уникального значения чтобы в корреляционной матрице не было пропусков
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

        return numeric_df


    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        считает матрицу корреляций между числовыми признаками.

        возвращает
        pd.DataFrame
            квадратную матрицу корреляций: строки и столбцы — названия признаков.
        """
        # берёт только числовые признаки (без label)
        numeric_df = self._get_numeric_features()

        # считает матрицу корреляции средствами pandas с расчетом корреляции методом пирсона
        corr_matrix = numeric_df.corr(method="pearson")

        return corr_matrix


    def find_strong_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        ищет пары признаков с сильной взаимосвязью по модулю корреляции.

        параметры
        corr_matrix : pd.DataFrame]
            заранее посчитанная матрица корреляций.
        threshold : float
            порог по модулю корреляции. пары с |corr| >= threshold считаются "сильно" связанными.

        возвращает
        pd.DataFrame
            таблицу с колонками:
            - feature_1 — имя первого признака;
            - feature_2 — имя второго признака;
            - correlation — значение коэффициента корреляции между ними.
        """
        pairs = []

        # получает список всех признаков
        columns = list(corr_matrix.columns)

        # как вот эта ебань работает лучше у нейронки спросите, тут или рисовать или паста текста по которому все равно непонятно
        # совсем коротко - пробег по верхнему треугольнику матрицы, не включая диагональ
        for i, col_i in enumerate(columns):
            for j in range(i + 1, len(columns)):
                col_j = columns[j]                         
                corr_value = corr_matrix.loc[col_i, col_j]

                # фильтрует только достаточно сильные связи по модулю
                if abs(corr_value) >= threshold:
                    pairs.append(
                        {
                            "feature_1": col_i,
                            "feature_2": col_j,
                            "correlation": corr_value,
                        }
                    )

        # превращает список словарей в датафрейм
        strong_pairs_df = pd.DataFrame(pairs)

        # сортирует по убыванию |corr|, чтобы сверху были самые сильные связи
        if not strong_pairs_df.empty:
            strong_pairs_df = strong_pairs_df.reindex(
                strong_pairs_df["correlation"].abs().sort_values(ascending=False).index
            )

        return strong_pairs_df


    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        figsize: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        """
        рисует тепловую карту (heatmap) матрицы корреляций — аналог imagesc из matlab.

        параметры
        corr_matrix : pd.DataFrame
            заранее посчитанная матрица корреляций.
        figsize: размер рисунка в дюймах (ширина, высота).
        """

        # создаёт фигуру указанного размера
        plt.figure(figsize=figsize)

        # imshow рисует "картинку" из числовой матрицы (аналог imagesc)
        # vmin=-1, vmax=1 фиксирует шкалу по цвету, т.к. коэффициент корреляции лежит в [-1, 1]
        img = plt.imshow(corr_matrix.values, vmin=-1, vmax=1)

        # добавляет панель с цветовой шкалой
        plt.colorbar(img, fraction=0.046, pad=0.04)

        # ставит подписи по осям — названия признаков
        feature_names = list(corr_matrix.columns)
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(feature_names)), feature_names)

        # чуть плотнее размещает элементы, чтобы подписи не обрезались
        plt.tight_layout()

        # отображает рисунок
        plt.show()