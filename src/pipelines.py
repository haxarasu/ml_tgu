# импорты
from typing import Any, Optional, Tuple
import matplotlib.pyplot as plt  # библиотека для визуализации дерева
from sklearn.model_selection import GridSearchCV, train_test_split # функция для разбиения данных на train, test и подбор параметров
from sklearn.preprocessing import StandardScaler # скалирование признаков для нормального обучения логистической регрессии
from sklearn.decomposition import PCA # метод главных компонент для ужатия пространства
from sklearn.neural_network import MLPClassifier # полносвязная нейронная сеть
from sklearn.cluster import KMeans # алгоритм k-means для кластеризации
from sklearn.tree import DecisionTreeClassifier, plot_tree # алгоритм решающего дерева и функция для его отрисовки
from sklearn.ensemble import RandomForestClassifier # отсюда и ниже алгоритмы обучения моделей
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier 
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def scale_features(X: Any) -> Tuple[Any, StandardScaler]:
    """
    скалирует признаки с помощью standardscaler и возвращает скалированные данные и объект скалера.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def train_with_grid_search(
    estimator,
    X_train,
    y_train,
    param_grid: Optional[dict] = None,
    cv: int = 3,
    scoring: Optional[str] = "accuracy",
    n_jobs: int = -1,
):
    """
    обучает модель с помощью обычного fit или gridsearchcv, в зависимости от того, передана ли сетка параметров.

    estimator  — объект модели (catboost, lightgbm, xgboost, randomforest и т.д.)
    X_train    — обучающие признаки
    y_train    — обучающий таргет
    param_grid — словарь с сеткой гиперпараметров; если None или пустой, используется обычный fit
    cv         — число фолдов кросс-валидации
    scoring    — метрика для подбора гиперпараметров
    n_jobs     — число потоков (-1 — использовать все ядра)
    """

    # если сетка параметров не задана — обучаем модель обычным fit
    if not param_grid:
        estimator.fit(X_train, y_train)

        return estimator

    # если сетка задана — используем gridsearchcv
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    return best_model


class Splitter:
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        # сохраняет параметры разбиения и random_state (для детерменированности)
        self.test_ratio = test_ratio
        self.random_state = random_state

        # делит исходные данные на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )
        

# классы выглядят почти одинаково, логику можно инкапуслировать
"""
class BaseModel(Splitter):

    # базовый класс для моделей без скалирования.
  
    def __init__(self, X, y, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        super().__init__(X, y, test_ratio=test_ratio, random_state=random_state)
        self.model = None  # сюда будут класться конкретные модели

    def fit(self, model_cls, **params):
        # создаёт и обучает переданную модель на обучающей выборке
        model = model_cls(**params)
        model.fit(self.X_train, self.y_train)
        self.model = model

        return model

    def predict(self, X):
        # проверяет, что модель уже обучена
        if self.model is None:
            raise ValueError("модель не обучена, сначала вызовите fit()")

        return self.model.predict(X)
"""
# и аналогично для моделей со скалированием через вызов scale_features
# но хз насколько вам и преподу это надо, мб на защите только больше запутаетесь

class Boosters(Splitter): # наследует Splitter
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2) -> None:
        # вызывает базовый класс, который делает разбиение на train и test
        super().__init__(X, y, test_ratio=test_ratio)

        # атрибуты для хранения обученных моделей
        self.catboost_model = None
        self.lightgbm_model = None
        self.xgboost_model = None


    def train_catboost(self, param_grid: Optional[dict] = None) -> CatBoostClassifier:
        # создаёт и обучает модель catboost на обучающей выборке
        base_model = CatBoostClassifier(verbose=False)  # False чтобы не выводил лишнюю инфу

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.catboost_model = model

        return model


    def train_lightgbm(self, param_grid: Optional[dict] = None) -> LGBMClassifier:
        # создаёт и обучает модель lightgbm на обучающей выборке
        base_model = LGBMClassifier(verbose=-1)  # то же что и False в CatBoost 

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.lightgbm_model = model

        return model


    def train_xgboost(self, param_grid: Optional[dict] = None) -> XGBClassifier:
        # создаёт и обучает модель xgboost на обучающей выборке
        # xgboost ожидает на вход label как [0, 1, 2], а получает [1, 2, 3], нужно поменять
        y_train_zero_based = self.y_train - self.y_train.min()

        base_model = XGBClassifier(
            verbosity=1, # выводит только предупреждения
            objective="multi:softprob",
            num_class=3,
        )  # учится на 3 меткаъ

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=y_train_zero_based,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.xgboost_model = model

        return model


class DTree(Splitter): # наследует Splitter
    """
    класс для обучения и визуализации модели решающего дерева.
    """
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        # вызывает базовый класс, который делает разбиение на train и test
        super().__init__(X, y, test_ratio=test_ratio, random_state=random_state)

        # сохраняет имена признаков
        self.feature_names = list(X.columns)

        # атрибут для хранения обученной модели дерева
        self.tree_model = None


    def train_tree(self, param_grid: Optional[dict] = None, **kwargs: Any) -> DecisionTreeClassifier:
        # создаёт и обучает модель решающего дерева на обучающей выборке
        params = {
            "max_depth": 3,
            "random_state": self.random_state,
        }
        # позволяет переопределить аргументы при запуске 
        params.update(**kwargs)

        base_model = DecisionTreeClassifier(**params)

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.tree_model = model

        return model


    def plot_tree(self, figsize: tuple[int, int] = (12, 8)) -> None:
        # визуализирует обученное решающее дерево с помощью matplotlib
        if self.tree_model is None:
            raise ValueError("модель дерева не обучена, сначала вызовите train_tree()")

        plt.figure(figsize=figsize)
        plot_tree(
            self.tree_model,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=8,
        )
        plt.tight_layout()
        plt.show()


class RandomForest(Splitter):
    """
    класс для обучения модели случайного леса.
    """
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        # вызывает базовый класс, который делает разбиение на train и test
        super().__init__(X, y, test_ratio=test_ratio, random_state=random_state)

        # атрибут для хранения обученной модели случайного леса
        self.random_forest_model = None


    def train_random_forest(self, param_grid: Optional[dict] = None, **kwargs: Any) -> RandomForestClassifier:
        # создаёт и обучает модель случайного леса на обучающей выборке
        params = {
            "n_estimators": 100,
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        # позволяет переопределить гиперпараметры при запуске
        params.update(**kwargs)

        base_model = RandomForestClassifier(**params)

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.random_forest_model = model

        return model


class LogisticBinary(Splitter):
    """
    класс для решения задачи бинарной классификации с помощью логистической регрессии.
    """
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        # скалирует признаки и сохраняет объект скалера
        X_scaled, self.scaler = scale_features(X)
        # вызывает базовый класс, который делает разбиение на train и test
        super().__init__(X_scaled, y, test_ratio=test_ratio, random_state=random_state)

        # атрибут для хранения обученной модели логистической регрессии
        self.logreg_model = None


    def fit(self, param_grid: Optional[dict] = None, **kwargs: Any) -> LogisticRegression:
        # создаёт и обучает модель логистической регрессии на обучающей выборке
        params = {
            "random_state": self.random_state,
            "n_jobs": -1,
            "max_iter": 5000,
        }

        # позволяет докинуть/изменить гиперпараметры при запуске
        params.update(**kwargs)

        base_model = LogisticRegression(**params)

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.logreg_model = model

        return model


    def predict(self, X: Any) -> Any:
        # проверяет, что модель логистической регрессии уже обучена
        if self.logreg_model is None:
            raise ValueError("модель логистической регрессии не обучена, сначала вызовите fit()")

        # масштабирует входные данные так же, как обучающую выборку
        X_scaled = self.scaler.transform(X)
        preds = self.logreg_model.predict(X_scaled)

        return preds


class NeuralNet(Splitter):
    """
    класс для решения задачи классификации с помощью полносвязной нейронной сети.
    """
    def __init__(self, X: Any, y: Any, test_ratio: float = 0.2, random_state: Optional[int] = None) -> None:
        # сохраняет random_state для воспроизводимости
        self.random_state = random_state

        # скалирует признаки и сохраняет объект скалера
        X_scaled, self.scaler = scale_features(X)

        # вызывает базовый класс, который делает разбиение на train и test
        super().__init__(X_scaled, y, test_ratio=test_ratio, random_state=random_state)

        # атрибут для хранения обученной модели нейронной сети
        self.nn_model = None


    def fit(self, param_grid: Optional[dict] = None, **kwargs: Any) -> MLPClassifier:
        # создаёт и обучает модель полносвязной нейронной сети на обучающей выборке
        params = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "random_state": self.random_state,
            "max_iter": 200,
        }

        # позволяет докинуть или изменить гиперпараметры при запуске
        params.update(**kwargs)

        base_model = MLPClassifier(**params)

        model = train_with_grid_search(
            estimator=base_model,
            X_train=self.X_train,
            y_train=self.y_train,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        self.nn_model = model

        return model


    def predict(self, X: Any) -> Any:
        # проверяет, что модель нейронной сети уже обучена
        if self.nn_model is None:
            raise ValueError("модель нейронной сети не обучена, сначала вызовите fit()")

        # масштабирует входные данные так же, как обучающую выборку
        X_scaled = self.scaler.transform(X)
        preds = self.nn_model.predict(X_scaled)

        return preds


class KMeansClustering:
    """
    класс для кластеризации с помощью k-means, с ужатием пространства и визуализацией в 2d и 3d.
    """
    def __init__(self, X: Any, n_clusters: int = 3, random_state: Optional[int] = None, scale: bool = True) -> None:
        # сохраняет параметры кластеризации
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scale = scale

        # по желанию скалирует признаки перед обучением k-means
        if self.scale:
            X_scaled, self.scaler = scale_features(X)
            self.X = X_scaled
        else:
            self.X = X
            self.scaler = None

        # атрибуты для хранения обученной модели и проекций пространства
        self.kmeans_model = None
        self.pca_2d = None
        self.pca_3d = None
        self.X_2d = None
        self.X_3d = None


    def fit(self, **kwargs: Any) -> KMeans:
        # создаёт и обучает модель k-means на всём наборе данных
        params = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": "auto",
        }

        # позволяет переопределить гиперпараметры при запуске
        params.update(**kwargs)

        model = KMeans(**params)
        model.fit(self.X)
        self.kmeans_model = model

        # строит pca-проекции в 2d и 3d для визуализации кластеров
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        self.X_2d = self.pca_2d.fit_transform(self.X)

        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        self.X_3d = self.pca_3d.fit_transform(self.X)

        return model


    def plot_2d(self, figsize: tuple[int, int] = (8, 6)) -> None:
        # визуализирует результат кластеризации в 2d после pca-ужатия
        if self.kmeans_model is None or self.X_2d is None:
            raise ValueError("сначала нужно обучить модель, вызвав fit()")

        labels = self.kmeans_model.labels_

        plt.figure(figsize=figsize)
        plt.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=labels, s=20)
        plt.xlabel("pc1")
        plt.ylabel("pc2")
        plt.title("k-means кластеры в 2d pca-пространстве")
        plt.tight_layout()
        plt.show()


    def plot_3d(self, figsize: tuple[int, int] = (8, 6)) -> None:
        # визуализирует результат кластеризации в 3d после pca-ужатия
        if self.kmeans_model is None or self.X_3d is None:
            raise ValueError("сначала нужно обучить модель, вызвав fit()")

        labels = self.kmeans_model.labels_

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.X_3d[:, 0], self.X_3d[:, 1], self.X_3d[:, 2], c=labels, s=20)
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
        ax.set_zlabel("pc3")
        ax.set_title("k-means кластеры в 3d pca-пространстве")
        plt.tight_layout()
        plt.show()
