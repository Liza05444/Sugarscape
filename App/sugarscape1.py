"""Mesa агенты для модели Sugarscape (sequential/asynchronous).

Реализует правило движения агентов (движение по основным осям в пределах видимости, 
выбор ячейки с наибольшим количеством сахара с учетом расстояния и координат). Поедание, 
голодание, восстановление регулируются моделью с сохранением порядка «движение -> 
поедание -> восстановление -> сбор данных», соответствующему плану событий.
"""
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
from time import perf_counter
from typing import Tuple
import numpy as np
import polars as pl

import mesa
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid

from plotting import plot_model_metrics

###
class AntAgent(mesa.Agent):
    """агент в Sugarscape - муравей с показателями сахар/метаболизм/обзор и движение."""
    def __init__(
        self,
        model: mesa.Model,
        *,
        sugar: int,
        metabolism: int,
        vision: int,
    ) -> None:
        super().__init__(model)
        self.sugar = int(sugar)
        self.metabolism = int(metabolism)
        self.vision = int(vision)

    # --- функции передвижения (sequential/asynchronous) ---
    def _visible_cells(self, origin: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = origin
        width, height = self.model.width, self.model.height
        cells: list[tuple[int, int]] = [origin]
        for step in range(1, self.vision + 1):
            if x0 + step < width:
                cells.append((x0 + step, y0))
            if x0 - step >= 0:
                cells.append((x0 - step, y0))
            if y0 + step < height:
                cells.append((x0, y0 + step))
            if y0 - step >= 0:
                cells.append((x0, y0 - step))
        return cells

    def _choose_best_cell(self, origin: tuple[int, int]) -> tuple[int, int]:
        # макс.сахар; расчет расстояния; координаты.
        best_cell = origin
        best_sugar = int(self.model.sugar_current[origin[0], origin[1]])
        best_distance = 0
        ox, oy = origin
        for cx, cy in self._visible_cells(origin):
            # блокировка занятых ячеек кроме цели (в свою ячейку может быть откат)
            if (cx, cy) != origin and not self.model.grid.is_cell_empty((cx, cy)):
                continue
            sugar_here = int(self.model.sugar_current[cx, cy])
            distance = abs(cx - ox) + abs(cy - oy)
            better = False
            if sugar_here > best_sugar:
                better = True
            elif sugar_here == best_sugar:
                if distance < best_distance:
                    better = True
                elif distance == best_distance and (cx, cy) < best_cell:
                    better = True
            if better:
                best_cell = (cx, cy)
                best_sugar = sugar_here
                best_distance = distance
        return best_cell

    def move(self) -> None:
        best = self._choose_best_cell(self.pos)
        if best != self.pos:
            self.model.grid.move_agent(self, best)

__all__ = ["AntAgent"]

"""Mesa реализация
Имеет структуру Mesa, метод `run` сохраняет результаты 
в формате CSV и строит траекторию движения. 
Модель обновляется в следующем порядке:
 движение -> питание -> восстановление -> сбор данных
"""

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """ Безопасное вычисление корреляции Пирсона между двумя одномерными массивами.
    Отражает принцип работы вспомогательной функции фрейма:
      возвращает nan для вырожденных входных данных.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def corr_sugar_metabolism(model: mesa.Model) -> float:
    sugars = np.fromiter((a.sugar for a in model.agent_list), dtype=float)
    mets = np.fromiter((a.metabolism for a in model.agent_list), dtype=float)
    return _safe_corr(sugars, mets)

def corr_sugar_vision(model: mesa.Model) -> float:
    sugars = np.fromiter((a.sugar for a in model.agent_list), dtype=float)
    vision = np.fromiter((a.vision for a in model.agent_list), dtype=float)
    return _safe_corr(sugars, vision)

def gini(values: Iterable[float]) -> float:
    array = np.fromiter(values, dtype=float)
    if array.size == 0:
        return float("nan")
    if np.allclose(array, 0.0):
        return 0.0
    if np.allclose(array, array[0]):
        return 0.0
    sorted_vals = np.sort(array)
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)

##
class Sugarscape(mesa.Model):
    def __init__(
        self,
        agents: int,
        *,
        width: int,
        height: int,
        max_sugar: int = 4,
        seed: int | None = None,
    ) -> None:
        if agents > width * height:
           raise ValueError(
           "Cannot place more agents than grid cells when capacity is 1.")

        super().__init__()
        if seed is None:
            seed = self.random.randint(0, np.iinfo(np.int32).max)
        self.reset_randomizer(seed)
        self.width = int(width)
        self.height = int(height)

        # Поле сахар (текущее значение и макс) как 2D массивы в форме (width, height)
        numpy_rng = np.random.default_rng(seed)
        self.sugar_max = numpy_rng.integers(
            0, max_sugar + 1, size=(width, height), dtype=np.int64
        )
        self.sugar_current = self.sugar_max.copy()

        # Сетка (Grid) емкостью 1 (агент)
        self.grid = SingleGrid(width, height, torus=False)

        # агенты (Python list, вручную перемешиваем и перебираем)
        self.agent_list: list[AntAgent] = []
        # расставим агентов в пустые ячейки; инициализируем с RNG
        placed = 0
        while placed < agents:
            x = int(self.random.randrange(0, width))
            y = int(self.random.randrange(0, height))
            if self.grid.is_cell_empty((x, y)):
                a = AntAgent(
                    self,
                    sugar=int(self.random.randint(6, 25)),
                    metabolism=int(self.random.randint(2, 5)),
                    vision=int(self.random.randint(1, 6)),
                )
                self.grid.place_agent(a, (x, y))
                self.agent_list.append(a)
                placed += 1

        # Функции сбора данных уровня модели,
        # и Frames реализация сохранения в CSV
        self.datacollector = DataCollector(
            model_reporters={
                "mean_sugar": lambda m: (
                    float(np.mean([a.sugar for a in m.agent_list]))
                    if m.agent_list
                    else 0.0
                ),
                "total_sugar": lambda m: (
                    float(sum(a.sugar for a in m.agent_list)) if m.agent_list else 0.0
                ),
                "agents_alive": lambda m: float(len(m.agent_list)),
                "gini": lambda m: gini(a.sugar for a in m.agent_list),
                "corr_sugar_metabolism": lambda m: corr_sugar_metabolism(m),
                "corr_sugar_vision": lambda m: corr_sugar_vision(m),
                "seed": lambda m: seed,
            },
            agent_reporters={
                "traits": lambda a: {
                    "sugar": a.sugar,
                    "metabolism": a.metabolism,
                    "vision": a.vision,
                }
            },
        )
        self.datacollector.collect(self)

    # --- планирование ---
    def _harvest_and_survive(self) -> None:
        survivors: list[AntAgent] = []
        for a in self.agent_list:
            x, y = a.pos
            a.sugar += int(self.sugar_current[x, y])
            a.sugar -= a.metabolism
            # сахар в ячейках с муравьями установим в 0
            self.sugar_current[x, y] = 0
            if a.sugar > 0:
                survivors.append(a)
            else:
                # удаление мертвых агентов из ячеек
                self.grid.remove_agent(a)
                a.remove()
        self.agent_list = survivors

    def _regrow(self) -> None:
        # пустые ячейки восстанвливаем в max;
        # занятые ячейки установим в 0
        for x in range(self.width):
            for y in range(self.height):
                if self.grid.is_cell_empty((x, y)):
                    self.sugar_current[x, y] = self.sugar_max[x, y]
                else:
                    self.sugar_current[x, y] = 0

    def step(self) -> None:
        # случайный порядок агентов 
        # логика= move-eat/starve-regrow-collect
        self.random.shuffle(self.agent_list)
        for a in self.agent_list:
            a.move()
        self._harvest_and_survive()
        self._regrow()
        self.datacollector.collect(self)
        if not self.agent_list:
            self.running = False

    def run(self, steps: int) -> None:
        for _ in range(steps):
            if not getattr(self, "running", True):
                break
            self.step()

###
def run(config=None):
    agents = 10000
    width = 200
    height = 200
    steps = 100
    max_sugar = 4
    seed = None
    plot = False
    save_results = True
    results_dir = None

    if config:
        agents = int(config.get("agents", agents))
        width = int(config.get("width", width))
        height = int(config.get("height", height))
        steps = int(config.get("steps", steps))
        max_sugar = int(config.get("max_sugar", max_sugar))
        if config.get("seed") is not None:
            seed = int(config["seed"])
        plot = config.get("plot", plot)
        save_results = config.get("save_results", save_results)
        results_dir = config.get("results_dir")

    print(f"Запуск модели Sugarscape(mesa) с {agents} агентами на поле {width}x{height} за {steps} шагов")
    start_time = perf_counter()
    model = Sugarscape(agents, width=width, height=height, max_sugar=max_sugar, seed=seed)
    model.run(steps)
    sim_time = perf_counter() - start_time
    model_pd = model.datacollector.get_model_vars_dataframe().reset_index().rename(columns={"index": "step"})
    print(f"Моделирование завершилось за {sim_time:.3f} сек")
    model_pd.attrs["runtime_seconds"] = sim_time
    model_pd["seed"] = model_pd["seed"].astype(int)
    display_pd = model_pd.drop(columns=["seed"])
    print(f"Метрики на 5 последних шагах:\n{display_pd.tail(5).to_string(index=False)}")

    str_plot_dir = "."
    if save_results or plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        results_dir = Path(results_dir or (Path(__file__).resolve().parent.parent / "results" / timestamp))
        results_dir.mkdir(parents=True, exist_ok=True)
    if save_results:
        model_pd.to_csv(results_dir / "model.csv", index=False)
        print(f"Сохранили результаты в CSV в {results_dir}")
    if plot:
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for col in display_pd.columns.drop("step"):
            single = pl.from_pandas(display_pd[["step", col]])
            plot_model_metrics(
                single, plots_dir, col, title=f"Модель Sugarscape — {col.capitalize()}",
                subtitle=f"mesa; seed={model_pd['seed'].iat[0]}", agents=agents, steps=steps,
            )
        print(f"Сохранили графики в {plots_dir}")
        str_plot_dir = str(plots_dir)
    return display_pd, str_plot_dir
