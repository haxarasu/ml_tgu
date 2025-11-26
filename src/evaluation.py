from typing import Any, Dict, Sequence # типизация
import numpy as np # для поиска индекса при расчете roc auc
import matplotlib.pyplot as plt  # библиотека для визуализации
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, roc_curve  # метрики качества моделей


class ModelEvaluator:
    """
    класс для оценки качества нескольких моделей по accuracy, precision, f1 и auc/roc.
    принимает на вход кортеж уже обученных моделей и считает метрики для каждой.
    """
    def __init__(
        self,
        models: Sequence[Any],
        X_test: Any,
        y_test: Any,
        pos_label: int = 1,
    ) -> None:
        # сохраняет модели и разбиение на обучающую и тестовую выборки
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

        # какое значение таргета считать положительным классом (для roc/auc)
        self.pos_label = pos_label

        # здесь будут храниться результаты для каждой модели
        # ключ: имя модели, значение: словарь с метриками и предсказаниями
        self.results: dict[str, dict] = {}


    def evaluate(self, average: str = "binary") -> Dict[str, Dict[str, Any]]:
        """
        оценивает все модели из кортежа и считает для каждой accuracy, precision, f1 и auc (если есть вероятности).
        для бинарной классификации average="binary",
        для мультиклассовой можно использовать "macro" или "weighted".
        """
        for model in self.models:
            # имя модели берём из её класса
            model_name = model.__class__.__name__

            X_test = self.X_test
            y_test = self.y_test

            # получает предсказания на тестовой выборке
            # для xgboost учитываем, что модель обучалась на сдвинутых метках (0, 1, 2),
            # поэтому после predict добавляем минимум из y_test, чтобы вернуться к исходным меткам (1, 2, 3)
            lower_name = model_name.lower()
            if "xgb" in lower_name:
                y_pred_zero_based = model.predict(X_test)
                y_pred = y_pred_zero_based + y_test.min()
            else:
                y_pred = model.predict(X_test)

            # по умолчанию auc не считаем (если нет вероятностей)
            y_proba = None
            auc_value = None

            # если у модели есть метод predict_proba, пробуем получить вероятности положительного класса
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                # для бинарной классификации берём вероятность класса pos_label
                if proba.ndim == 2 and proba.shape[1] > 1:
                    # ищем столбец по фактическому индексу класса
                    class_indices = np.where(model.classes_ == self.pos_label)[0]
                    if class_indices.size == 0:
                        # если положительный класс отсутствует в разбиении, просто пропускаем auc для этой модели
                        y_proba = None
                    else:
                        y_proba = proba[:, class_indices[0]]
                else:
                    y_proba = proba.ravel()

            # считает accuracy
            acc = accuracy_score(y_test, y_pred)

            # считает precision и f1
            prec = precision_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)


            # считает auc только для бинарной классификации
            # в многоклассовом случае auc оставляем None, чтобы не вызывать ошибку
            if y_proba is not None and len(set(y_test)) == 2:
                auc_value = roc_auc_score(y_test, y_proba)

            # сохраняет результаты для этой модели
            self.results[model_name] = {
                "model": model,
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "accuracy": acc,
                "precision": prec,
                "f1": f1,
                "auc": auc_value,
            }

        return self.results


    def plot_roc_curve(self, model_name: str, figsize: tuple[int, int] = (8, 6)) -> None:
        """
        строит roc-кривую для указанной модели по имени (класс модели).
        требует, чтобы перед этим был вызван evaluate(),
        а у модели были посчитаны вероятности положительного класса.
        """
        # проверяет, что для модели уже посчитаны результаты
        if model_name not in self.results:
            raise ValueError(f"для модели {model_name} ещё не посчитаны метрики, сначала вызовите evaluate()")

        y_true = self.results[model_name]["y_true"]
        y_proba = self.results[model_name]["y_proba"]

        # проверяет, что есть вероятности положительного класса
        if y_proba is None:
            raise ValueError(f"для модели {model_name} не удалось получить вероятности (y_proba), roc-кривая недоступна")

        # считает точки roc-кривой
        fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=self.pos_label)

        # считает площадь под кривой
        auc_value = roc_auc_score(y_true, y_proba)

        # рисует roc-кривую
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f"roc-кривая {model_name} (auc = {auc_value:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайный классификатор")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title(f"roc-кривая для модели {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()