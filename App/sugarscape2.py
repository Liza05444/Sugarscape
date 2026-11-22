"""Mesa-frames модель Sugarscape

Реализует правило движения агентов (движение по основным осям в пределах видимости, 
выбор ячейки с наибольшим количеством сахара с учетом расстояния и координат). Поедание, 
голодание, восстановление регулируются моделью с сохранением порядка «движение -> 
поедание -> восстановление -> сбор данных», соответствующему плану событий.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Annotated
from time import perf_counter
from dataclasses import dataclass
import numpy as np
import polars as pl
import mesa_frames
from mesa_frames.concrete.polars.agentset import AgentSetPolars as AgentSet
from mesa_frames.concrete.model import ModelDF as Model
from mesa_frames.concrete.polars.space import GridPolars as Grid

from plotting import plot_model_metrics

class AntsBase(AgentSet):
    """Базовый агент для примера Sugarscape.
    Класс реализует общее поведение агентов с перемещениями
     (sequential, numba-accelerated, parallel).

    Примечания
    ----------
    Агенты имеют целочисленные атрибуты: sugar, metabolism, vision.
     Инциализация  через `__init__`.
    Под-класс должен реализовать move для правильного положения агента
      на сетке мира (`mesa_frames.Grid`).
    """

    def __init__(self, model: Model, agent_frame: pl.DataFrame) -> None:
        """Инициализация агентов в agentset и валидация колонок характеристик.

        Параметры
        ----------
        model : Model
            Модель-родитель с RNG и пространством(space).
        agent_frame : pl.DataFrame
            Polars DataFrame с колонками ``sugar``,
            ``metabolism``, ``vision`` для каждого агента.

        Сигналы
        ------
        ValueError
            Если есть ошибка в колонках ``agent_frame``.
        """
        super().__init__(model)
        required = {"sugar", "metabolism", "vision"}
        missing = required.difference(agent_frame.columns)
        if missing:
            raise ValueError(
                f"Initial agent frame must include columns {sorted(required)}; missing {sorted(missing)}."
            )
        self.add(agent_frame.clone())

    @property
    def df(self) -> pl.DataFrame:
        """базовый Polars DataFrame с атрибутами agentset.

        mesa_frames 0.1 поддерживает внутренний фрейм ``.agents``;
        как псевдоним/alias ``.df``в API 
        """
        return self.agents

    def step(self) -> None:
        """Выполняет один шаг над agentset.

        Здесь важен порядок вычислений: агенты рандомно перемешиваются,
         (это важно для последовательных вариантов), затем двигаются,
        затем собирают сахар в своей ячейке, затем агенты, у которых
        сахар упал до 0 и меньше удаляются.
        """
        # Randomise ordering for movement decisions when required by the
        # implementation (e.g. sequential update uses this shuffle).
        self.agents = self.agents.sample(
            fraction=1, shuffle=True, seed=int(self.random.integers(2**32))
        )
        # Movement policy implemented by subclasses.
        self.move()
        # Agents harvest sugar on their occupied cells.
        self.eat()
        # Remove agents that starved after eating.
        self._remove_starved()

    def move(self) -> None:  # pragma: no cover
        """Абстрактный метод для движения.

        Под-классы должны переопределить этот метод для смещения по сетке.
        Реализации должны использовать метод `mesa_frames.Grid.move_agents`
        """
        raise NotImplementedError

    def eat(self) -> None:
        """Агенты собирают сахар из ячеек, которые агент занял.

        поведение:
        - осмотреть массив занятых ячеек (cell со ссылкой на agent_id).
        - для каждой такой ячейки добавить сахар из ячейки в сазар агента
          и вычесть затраты на метаболизм (metabolism).
        - установить значение сахара в ячейке =0.
        """
        # Map of currently occupied agent ids on the grid.
        occupied_ids = self.index
        # `occupied_ids` is a Polars Series; calling `is_in` with a Series
        # of the same datatype is ambiguous in newer Polars. Use `implode`
        # to collapse the Series into a list-like value for membership checks.
        occupied_cells = self.space.cells.filter(
            pl.col("agent_id").is_in(occupied_ids.implode())
        )
        if occupied_cells.is_empty():
            return
        # The agent ordering here uses the agent_id values stored in the
        # occupied cells frame; indexing the agent set with that vector updates
        # the matching agents' sugar values in one vectorised write.
        agent_ids = occupied_cells["agent_id"]
        self[agent_ids, "sugar"] = (
            self[agent_ids, "sugar"]
            + occupied_cells["sugar"]
            - self[agent_ids, "metabolism"]
        )
        # After harvesting, occupied cells have zero sugar.
        self.space.set_cells(
            occupied_cells.select(["dim_0", "dim_1"]),
            {"sugar": pl.Series(np.zeros(len(occupied_cells), dtype=np.int64))},
        )

    def _remove_starved(self) -> None:
        """
        Выбрасывание агентов, у которых запас сахара упал до 0 или ниже.
        Этот метод выполняет векторизованный фильтр для фрейма агентов и
        удаляет из набора все совпадающие строки.
        """
        starved = self.df.filter(pl.col("sugar") <= 0)
        if not starved.is_empty():
            # ``discard`` accepts a DataFrame of agents to remove.
            self.discard(starved)

#
class AntsParallel(AntsBase):
    def move(self) -> None:
        """
        Параллельное перемещение агентов путем ранжирования видимых ячеек и разрешения конфликтов.

        Декларативная ментальная модель: выразите, *что* хочет каждый агент (ранжирование кандидатов),
        затем используйте операции с фреймами данных для *распределения* (объединения, группировки с помощью лотереи).
        За производительность отвечает Polars/LazyFrames, избегая преждевременных микрооптимизаций.

        Возвращает
        -------
        None
            Изменение смещения в пространстве не имеет результата.
        """
        # Early exit if there are no agents.
        if len(self.df) == 0:
            return

        # current_pos columns:
        # ┌──────────┬────────────────┬────────────────┐
        # │ agent_id ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---      ┆ ---            ┆ ---            │
        # │ u64      ┆ i64            ┆ i64            │
        # ╞══════════╪════════════════╪════════════════╡
        current_pos = self.pos.select(
            [
                pl.col("unique_id").alias("agent_id"),
                pl.col("dim_0").alias("dim_0_center"),
                pl.col("dim_1").alias("dim_1_center"),
            ]
        )

        neighborhood = self._build_neighborhood_frame(current_pos)
        choices, origins, max_rank = self._rank_candidates(neighborhood, current_pos)
        if choices.is_empty():
            return

        assigned = self._resolve_conflicts_in_rounds(choices, origins, max_rank)
        if assigned.is_empty():
            return

        # move_df columns:
        # ┌────────────┬────────────┬────────────┐
        # │ unique_id  ┆ dim_0      ┆ dim_1      │
        # │ ---        ┆ ---        ┆ ---        │
        # │ u64        ┆ i64        ┆ i64        │
        # ╞════════════╪════════════╪════════════╡
        move_df = pl.DataFrame(
            {
                "unique_id": assigned["agent_id"],
                "dim_0": assigned["dim_0_candidate"],
                "dim_1": assigned["dim_1_candidate"],
            }
        )
        # `move_agents` accepts IdsLike and SpaceCoordinates (Polars Series/DataFrame),
        # so pass Series/DataFrame directly rather than converting to Python lists.
        self.space.move_agents(move_df["unique_id"], move_df.select(["dim_0", "dim_1"]))

    def _build_neighborhood_frame(self, current_pos: pl.DataFrame) -> pl.DataFrame:
        """Собрать соседство с атрибуцией сахара для каждого агента.

        Параметры
        ----------
        current_pos : pl.DataFrame
            Фрейм с колонками agent_id, dim_0_center, dim_1_center
             содержащий текущее положение кадлого агента.

        Возвращает
        -------
        pl.DataFrame
            фрейм с колонками agent_id, radius, dim_0_candidate,
            dim_1_candidate, sugar - содержащий видимые агенту ячейки
        """
        # Build a neighbourhood frame: for each agent and visible cell we
        # attach the cell sugar. The raw offsets contain the candidate
        # cell coordinates and the center coordinates for the sensing agent.
        # Raw neighborhood columns:
        # ┌────────────┬────────────┬────────┬────────────────┬────────────────┐
        # │ dim_0      ┆ dim_1      ┆ radius ┆ dim_0_center   ┆ dim_1_center   │
        # │ ---        ┆ ---        ┆ ---    ┆ ---            ┆ ---            │
        # │ i64        ┆ i64        ┆ i64    ┆ i64            ┆ i64            │
        # ╞════════════╪════════════╪════════╪════════════════╪════════════════╡
        neighborhood_cells = self.space.get_neighborhood(
            radius=self["vision"], pos=current_pos.select(
                pl.col("dim_0_center").alias("dim_0"),
                pl.col("dim_1_center").alias("dim_1"),
            ), include_center=True
        )

        # sugar_cells columns:
        # ┌────────────┬────────────┬────────┐
        # │ dim_0      ┆ dim_1      ┆ sugar  │
        # │ ---        ┆ ---        ┆ ---    │
        # │ i64        ┆ i64        ┆ i64    │
        # ╞════════════╪════════════╪════════╡

        sugar_cells = self.space.cells.select(["dim_0", "dim_1", "sugar"])

        neighborhood_cells = (
            neighborhood_cells.join(sugar_cells, on=["dim_0", "dim_1"], how="left")
            .with_columns(pl.col("sugar").fill_null(0))
            .rename({"dim_0": "dim_0_candidate", "dim_1": "dim_1_candidate"})
        )

        neighborhood_cells = neighborhood_cells.join(
            current_pos,
            left_on=["dim_0_center", "dim_1_center"],
            right_on=["dim_0_center", "dim_1_center"],
            how="left",
        )

        # Final neighborhood columns:
        # ┌──────────┬────────┬──────────────────┬──────────────────┬────────┐
        # │ agent_id ┆ radius ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  │
        # │ ---      ┆ ---    ┆ ---              ┆ ---              ┆ ---    │
        # │ u64      ┆ i64    ┆ i64              ┆ i64              ┆ i64    │
        # ╞══════════╪════════╪══════════════════╪══════════════════╪════════╡
        neighborhood_cells = neighborhood_cells.drop(
            ["dim_0_center", "dim_1_center"]
        ).select(["agent_id", "radius", "dim_0_candidate", "dim_1_candidate", "sugar"])

        return neighborhood_cells

    def _rank_candidates(
        self,
        neighborhood: pl.DataFrame,
        current_pos: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Ранжированные ячейки-кандидаты положения агентов.

        Параметры
        ----------
        neighborhood : pl.DataFrame
            результат из метода _build_neighborhood_frame с колонками
            agent_id, radius, dim_0_candidate, dim_1_candidate, sugar.
        current_pos : pl.DataFrame
            фрейм с колонками agent_id, dim_0_center, dim_1_center
             содержащий текущее положение агентов.

        Возвращает
        -------
        choices : pl.DataFrame
            ранжированные кандидаты для агентов с agent_id,
            dim_0_candidate, dim_1_candidate, sugar, radius, rank.
        origins : pl.DataFrame
            первоначальные координаты дляя агентов с agent_id,
            dim_0 и dim_1.
        max_rank : pl.DataFrame
            макс.ранг для агентов с agent_id и max_rank.
        """
        # Create ranked choices per agent: sort by sugar (desc), radius
        # (asc), then coordinates. Keep the first unique entry per cell.
        # choices columns (after select):
        # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius │
        # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    │
        # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    │
        # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╡
        choices = (
            neighborhood.select(
                [
                    "agent_id",
                    "dim_0_candidate",
                    "dim_1_candidate",
                    "sugar",
                    "radius",
                ]
            )
            .sort(
                ["agent_id", "sugar", "radius", "dim_0_candidate", "dim_1_candidate"],
                descending=[False, True, False, False, False],
            )
            .unique(
                subset=["agent_id", "dim_0_candidate", "dim_1_candidate"],
                keep="first",
                maintain_order=True,
            )
            .with_columns(pl.col("agent_id").cum_count().over("agent_id").alias("rank"))
        )

        # Precompute per‑agent candidate rank once so conflict resolution can
        # promote losers by incrementing a cheap `current_rank` counter,
        # without re-sorting after each round. Alternative: drop taken cells
        # and re-rank by sugar every round; simpler conceptually but requires
        # repeated sorts and deduplication, which is heavier than filtering by
        # `rank >= current_rank`.

        # Origins for fallback (if an agent exhausts candidates it stays put).
        # origins columns:
        # ┌──────────┬────────────┬────────────┐
        # │ agent_id ┆ dim_0      ┆ dim_1      │
        # │ ---      ┆ ---        ┆ ---        │
        # │ u64      ┆ i64        ┆ i64        │
        # ╞══════════╪════════════╪════════════╡
        origins = current_pos.select(
            [
                "agent_id",
                pl.col("dim_0_center").alias("dim_0"),
                pl.col("dim_1_center").alias("dim_1"),
            ]
        )

        # Track the maximum available rank per agent to clamp promotions.
        # This bounds `current_rank`; once an agent reaches `max_rank` and
        # cannot secure a cell, they fall back to origin cleanly instead of
        # chasing nonexistent ranks.
        # max_rank columns:
        # ┌──────────┬───────────┐
        # │ agent_id ┆ max_rank │
        # │ ---      ┆ ---       │
        # │ u64      ┆ u32       │
        # ╞══════════╪═══════════╡
        max_rank = choices.group_by("agent_id").agg(
            pl.col("rank").max().alias("max_rank")
        )
        return choices, origins, max_rank

    def _resolve_conflicts_in_rounds(
        self,
        choices: pl.DataFrame,
        origins: pl.DataFrame,
        max_rank: pl.DataFrame,
    ) -> pl.DataFrame:
        """Определяет и разбирает конфликты.

        Параметры
        ----------
        choices : pl.DataFrame
            ранжированные ячейки-кандидаты для агента с заголовками для
            choices фрейма, возвращенного из _rank_candidates.
        origins : pl.DataFrame
            Первоначальные координаты с agent_id, dim_0 и dim_1.
        max_rank : pl.DataFrame
            макс.сдвиг ранга для агента с agent_id и max_rank.

        Возвращает
        -------
        pl.DataFrame
            Применяемое движение с agent_id, dim_0_candidate
            и dim_1_candidate; каждый ряд содержит назначенную ячейку агентам
        """
        # Prepare unresolved agents and working tables.
        agent_ids = choices["agent_id"].unique(maintain_order=True)

        # unresolved columns:
        # ┌──────────┬────────────────┐
        # │ agent_id ┆ current_rank   │
        # │ ---      ┆ ---            │
        # │ u64      ┆ i64            │
        # ╞══════════╪════════════════╡
        unresolved = pl.DataFrame(
            {
                "agent_id": agent_ids,
                "current_rank": pl.Series(np.zeros(len(agent_ids), dtype=np.int64)),
            }
        )

        # assigned columns:
        # ┌──────────┬──────────────────┬──────────────────┐
        # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---      ┆ ---              ┆ ---              │
        # │ u64      ┆ i64              ┆ i64              │
        # ╞══════════╪══════════════════╪══════════════════╡
        assigned = pl.DataFrame(
            {
                "agent_id": pl.Series(
                    name="agent_id", values=[], dtype=agent_ids.dtype
                ),
                "dim_0_candidate": pl.Series(
                    name="dim_0_candidate", values=[], dtype=pl.Int64
                ),
                "dim_1_candidate": pl.Series(
                    name="dim_1_candidate", values=[], dtype=pl.Int64
                ),
            }
        )

        # taken columns:
        # ┌──────────────────┬──────────────────┐
        # │ dim_0_candidate  ┆ dim_1_candidate  │
        # │ ---              ┆ ---              │
        # │ i64              ┆ i64              │
        # ╞══════════════════╪══════════════════╡
        # Treat all currently occupied cells (origins) as taken from the start.
        # Each agent may still target its own origin; we handle that exception
        # when filtering candidate pools.
        taken = origins.select(
            [
                pl.col("dim_0").alias("dim_0_candidate"),
                pl.col("dim_1").alias("dim_1_candidate"),
            ]
        )
        origins_for_filter = origins.rename(
            {"dim_0": "dim_0_origin", "dim_1": "dim_1_origin"}
        )

        # Resolve in rounds: each unresolved agent proposes its current-ranked
        # candidate; winners per-cell are selected at random and losers are
        # promoted to their next choice.
        while unresolved.height > 0:
            # Using precomputed `rank` lets us select candidates with
            # `rank >= current_rank` and avoid re-ranking after each round.
            # Alternative: remove taken cells and re-sort remaining candidates
            # by sugar/distance per round (heavier due to repeated sort/dedupe).
            # candidate_pool columns (after join with unresolved):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            candidate_pool = choices.join(unresolved, on="agent_id")
            candidate_pool = candidate_pool.filter(
                pl.col("rank") >= pl.col("current_rank")
            )
            candidate_pool = (
                candidate_pool.join(origins_for_filter, on="agent_id", how="left")
                .join(
                    taken.with_columns(pl.lit(True).alias("is_taken")),
                    on=["dim_0_candidate", "dim_1_candidate"],
                    how="left",
                )
                .filter(
                    pl.col("is_taken").is_null()
                    | (
                        (pl.col("dim_0_candidate") == pl.col("dim_0_origin"))
                        & (pl.col("dim_1_candidate") == pl.col("dim_1_origin"))
                    )
                )
                .drop(["dim_0_origin", "dim_1_origin", "is_taken"])
            )

            if candidate_pool.is_empty():
                # No available candidates — everyone falls back to origin.
                # Note: this covers both agents with no visible cells left and
                # the case where all remaining candidates are already taken.
                # fallback columns:
                # ┌──────────┬────────────┬────────────┬──────────────┐
                # │ agent_id ┆ dim_0      ┆ dim_1      ┆ current_rank │
                # │ ---      ┆ ---        ┆ ---        ┆ ---          │
                # │ u64      ┆ i64        ┆ i64        ┆ i64          │
                # ╞══════════╪════════════╪════════════╪══════════════╡
                fallback = unresolved.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                break

            # best_candidates columns (per agent first choice):
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          │
            # │ u64      ┆ i64              ┆ i64              ┆ i64    ┆ i64    ┆ u32  ┆ i64          │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╡
            best_candidates = (
                candidate_pool.sort(["agent_id", "rank"])
                .group_by("agent_id", maintain_order=True)
                .first()
            )

            # Agents that had no candidate this round fall back to origin.
            # missing columns:
            # ┌──────────┬──────────────┐
            # │ agent_id ┆ current_rank │
            # │ ---      ┆ ---          │
            # │ u64      ┆ i64          │
            # ╞══════════╪══════════════╡
            missing = unresolved.join(
                best_candidates.select("agent_id"), on="agent_id", how="anti"
            )
            if not missing.is_empty():
                # fallback (missing) columns match fallback table above.
                fallback = missing.join(origins, on="agent_id", how="left")
                assigned = pl.concat(
                    [
                        assigned,
                        fallback.select(
                            [
                                "agent_id",
                                pl.col("dim_0").alias("dim_0_candidate"),
                                pl.col("dim_1").alias("dim_1_candidate"),
                            ]
                        ),
                    ],
                    how="vertical",
                )
                unresolved = unresolved.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
                )
                best_candidates = best_candidates.join(
                    missing.select("agent_id"), on="agent_id", how="anti"
                )
                if unresolved.is_empty() or best_candidates.is_empty():
                    continue

            # Add a small random lottery to break ties deterministically for
            # each candidate set.
            lottery = pl.Series("lottery", self.random.random(best_candidates.height))
            best_candidates = best_candidates.with_columns(lottery)

            # winners columns:
            # ┌──────────┬──────────────────┬──────────────────┬────────┬────────┬──────┬──────────────┬─────────┐
            # │ agent_id ┆ dim_0_candidate  ┆ dim_1_candidate  ┆ sugar  ┆ radius ┆ rank ┆ current_rank │ lottery │
            # │ ---      ┆ ---              ┆ ---              ┆ ---    ┆ ---    ┆ ---  ┆ ---          ┆ f64     │
            # ╞══════════╪══════════════════╪══════════════════╪════════╪════════╪══════╪══════════════╪═════════╡
            winners = (
                best_candidates.sort(
                    ["dim_0_candidate", "dim_1_candidate", "lottery"],
                )
                .group_by(["dim_0_candidate", "dim_1_candidate"], maintain_order=True)
                .first()
            )

            assigned = pl.concat(
                [
                    assigned,
                    winners.select(
                        [
                            "agent_id",
                            pl.col("dim_0_candidate"),
                            pl.col("dim_1_candidate"),
                        ]
                    ),
                ],
                how="vertical",
            )
            taken = pl.concat(
                [
                    taken,
                    winners.select(["dim_0_candidate", "dim_1_candidate"]),
                ],
                how="vertical",
            )

            # Origins of agents that move away become available to others in
            # subsequent rounds. Keep origins for agents that stayed put.
            vacated = (
                winners.join(origins_for_filter, on="agent_id", how="left")
                .filter(
                    (pl.col("dim_0_candidate") != pl.col("dim_0_origin"))
                    | (pl.col("dim_1_candidate") != pl.col("dim_1_origin"))
                )
                .select(
                    pl.col("dim_0_origin").alias("dim_0_candidate"),
                    pl.col("dim_1_origin").alias("dim_1_candidate"),
                )
            )
            if not vacated.is_empty():
                taken = taken.join(
                    vacated,
                    on=["dim_0_candidate", "dim_1_candidate"],
                    how="anti",
                )

            winner_ids = winners.select("agent_id")
            unresolved = unresolved.join(winner_ids, on="agent_id", how="anti")
            if unresolved.is_empty():
                break

            # loser candidates columns mirror best_candidates (minus winners).
            losers = best_candidates.join(winner_ids, on="agent_id", how="anti")
            if losers.is_empty():
                continue

            # loser_updates columns (after select):
            # ┌──────────┬───────────┐
            # │ agent_id ┆ next_rank │
            # │ ---      ┆ ---       │
            # │ u64      ┆ i64       │
            # ╞══════════╪═══════════╡
            loser_updates = (
                losers.select(
                    "agent_id",
                    (pl.col("rank") + 1).cast(pl.Int64).alias("next_rank"),
                )
                .join(max_rank, on="agent_id", how="left")
                .with_columns(
                    pl.min_horizontal(pl.col("next_rank"), pl.col("max_rank")).alias(
                        "next_rank"
                    )
                )
                .select(["agent_id", "next_rank"])
            )

            # Promote losers' current_rank (if any) and continue.
            # unresolved (updated) retains columns agent_id/current_rank.
            unresolved = (
                unresolved.join(loser_updates, on="agent_id", how="left")
                .with_columns(
                    pl.when(pl.col("next_rank").is_not_null())
                    .then(pl.col("next_rank"))
                    .otherwise(pl.col("current_rank"))
                    .alias("current_rank")
                )
                .drop("next_rank")
            )
        return assigned

__all__ = [
    "AntsBase",
    "AntsParallel",
]

# Model-level reporters
class DataCollector:
    """Минимальная реализация DataCollector в стиле mesa-frames (dev-версия).

    Установленная версия mesa_frames (0.1.0a) не содержит DataCollector,
    поэтому определим совместимый минимум API, используемый в этом примере:
    collect(), data["model"], flush(), seed.
    """
    def __init__(
        self,
        model,
        model_reporters=None,
        agent_reporters=None,
        storage="memory",
        storage_uri=None,
    ):
        self._model = model
        self._model_reporters = model_reporters or {}
        self._agent_reporters = agent_reporters or {}
        self._storage = storage
        self._storage_uri = storage_uri
        self.seed = model._seed
        self._step = 0
        self._frames = []

    def collect(self):
        row = {"step": self._step, "seed": str(self.seed), "batch": 0}
        for name, reporter in self._model_reporters.items():
            row[name] = (
                reporter(self._model) if callable(reporter)
                else getattr(self._model, reporter)
            )
        self._frames.append(("model", self._step, 0, pl.DataFrame([row]).lazy()))
        self._step += 1

    @property
    def data(self):
        model_frames = [
            lf.collect() for kind, _, _, lf in self._frames if kind == "model"
        ]
        return {"model": pl.concat(model_frames) if model_frames else pl.DataFrame()}

    def flush(self):
        if self._storage == "csv" and self._storage_uri:
            for kind, step, batch, lf in self._frames:
                lf.collect().write_csv(
                    f"{self._storage_uri}/{kind}_step{step}_batch{batch}.csv"
                )

@dataclass
class FramesSimulationResult:
    """
    Контейнер для вывода результатов моделирования.
    Класс данных универсален: некоторые бэкенды предоставляют только
    `metrics`, в то время как другие также возвращают `agent_metrics`.
    """
    datacollector: DataCollector

def gini(model: Model) -> float:
    """
    Вычисляет коэффициент Джини, отражающий распределение сахара между агентами.

    Функция считывает основной набор агентов из model.sets[0] и вычисляет 
    коэффициент Джини по столбцу sugar. Реализация устойчива к пустым наборам и отсутствию сахара.
    
    Параметры
    ----------
    model : модель
        Модель содержащая agentset. Первичный agentset
        будет с индексом model.sets[0], достуаен через Polars DataFrame
        с именем .df 

    Возвращает
    -------
    float
        коэффициент Джини в диапазоне [0, 1] если есть расчет,
        0.0 если сахар нет,
        NaN если агент не существует.
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    sugar = primary_set.df["sugar"].to_numpy().astype(np.float64)
    sorted_vals = np.sort(sugar)
    n = sorted_vals.size
    if n == 0:
        return float("nan")
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(index, sorted_vals) / (n * total)) - (n + 1) / n)

def corr_sugar_metabolism(model: Model) -> float:
    """Корреляция Пирсона между сахаром агента и метаболизмом.

    здесь выбираются колонки sugar и metabolism из первичного
    agentset и возвращается коэф.корр.Пирсона.
    Если agentset пуст или есть ошибки данных возвращает nan.

    Параметры
    ----------
    model : Model
        The simulation model that contains agent sets. The primary agent set
        is expected to be at ``model.sets[0]`` and provide a Polars DataFrame
        with ``sugar`` and ``metabolism`` columns.

    Возвращает
    -------
    float
        коэф.корр.Пирсона между sugar и metabolism, или nan
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    agent_df = primary_set.df
    sugar = agent_df["sugar"].to_numpy().astype(np.float64)
    metabolism = agent_df["metabolism"].to_numpy().astype(np.float64)
    return _safe_corr(sugar, metabolism)

def corr_sugar_vision(model: Model) -> float:
    """Корреляция Пирсона между сахаром агента и обзорностью.

    выбирает колонки sugar и vision из первичного agentset
    и возвращает их коэф.корр.Пирсона. Если функция не может
    вычислить корреляцию (например, если agentset пуст)
    возвращает ``nan``.

    Параметры
    ----------
    model : Model
        The simulation model that contains agent sets. The primary agent set
        is expected to be at ``model.sets[0]`` and provide a Polars DataFrame
        with ``sugar`` and ``vision`` columns.

    Возвращает
    -------
    float
        коэф.корр.Пирсона между sugar и vision, или ``nan``
    """
    if len(model.sets) == 0:
        return float("nan")

    primary_set = model.sets[0]
    if len(primary_set) == 0:
        return float("nan")

    agent_df = primary_set.df
    sugar = agent_df["sugar"].to_numpy().astype(np.float64)
    vision = agent_df["vision"].to_numpy().astype(np.float64)
    return _safe_corr(sugar, vision)

def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Безопасное вычисление корреляции Пирсона между двумя одномерными массивами.
    Этот вспомогательный метод защищает от некорректных входных данных 
    (слишком малое количество наблюдений или массивы с константами), из-за которых 
    корреляция Пирсона может быть неопределенной или численно нестабильной. 
    Если корреляцию можно вычислить корректно, функция возвращает число float.

    Параметры
    ----------
    x : np.ndarray
        одномерный числовый массив с первой переменной для расчета корреляции.
    y : np.ndarray
        одномерный числовый массив со второй переменной для расчета корреляции.

    Возвращает
    -------
    float
        коэф.корр.Пирсона в формате Python float, или ``nan``
    """
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

#
class Sugarscape(Model):
    """
    Минимальная модель Sugarscape

    Этот класс объединяет сетку, в ячейках которой хранится «сахар», реализацию 
    набора агентов (передается как agent_type) и сборщик данных, который 
    записывает статистику на уровне модели и агентов.

    В задачи модели входит:
    - создание «сахарного ландшафта» (ячеек с текущим и максимальным количеством сахара)
    - создание и размещение агентов на сетке
    - применение правила восстановления сахара на каждом шаге
    - запуск модели на фиксированное количество шагов и сбор данных

    Параметры
    ----------
    agent_type : type[AntsBase]
        Класс `AgentSet` как под-класс, реализующий правила движения
        (sequential, numba-accelerated, parallel).
    n_agents : int
        число агентов для размещения в пространстве.
    width : int
        ширина мира-сетки (Grid width, число колонок).
    height : int
        высота мира-сетки (Grid height, число рядов).
    max_sugar : int, optional
        Макс.значение для рандомного значения сахара в ячейке, по умолчанию 4.
    seed : int | None, optional
        RNG seed для управления повторными прогонами, по умолчанию None.
    results_dir : Path | None, optional
        Папка для сохранения результатов CSV/графики. Если ``None``
        то модель гн будет сохранять CSV на диск (in-memory storage).

    Примечание
    -----
    В сетке используется окрестность фон Неймана и емкость 1 (максимум один агент 
    на ячейку). И сахарный ландшафт, и начальные характеристики агентов генерируются 
    с помощью self.random, поэтому при одинаковых начальных условиях можно создать 
    разные варианты движения, передав одно и то же начальное значение.
    """

    def __init__(
        self,
        agent_type: type[AntsBase],
        n_agents: int,
        *,
        width: int,
        height: int,
        max_sugar: int = 4,
        seed: int | None = None,
        results_dir: Path | None = None,
    ) -> None:
        
        if n_agents > width * height:
            raise ValueError(
            "Cannot place more agents than grid cells when capacity is 1.")
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        super().__init__(seed)
        
        # 1. Создадим пространственную сетку с сахаром

        sugar_grid_df = self._generate_sugar_grid(width, height, max_sugar)
        self.space = Grid(
            self, [width, height], neighborhood_type="von_neumann", capacity=1)
        self.space.set_cells(sugar_grid_df)
        self._max_sugar = sugar_grid_df.select(["dim_0", "dim_1", "max_sugar"])

        # 2. Создаем агентов и размещаем их на сетке

        agent_frame = self._generate_agent_frame(n_agents)
        main_set = agent_type(self, agent_frame)
        self.sets = [main_set]
        # Регистрируем agentset в модели для пространственных вычислений
        # (например place_to_empty) в структуре модельной среды
        self.agents.add(main_set)
        self.space.place_to_empty(self.sets)

        # 3. Добавляем сборщик данных (data collector)
        # Для тестирования возможен прогон модели без results_dir; не нужна выгрузка
        # результатов в CSV (и путь storage_uri), а все результаты только в memory.
        if results_dir is None:
            storage = "memory"
            storage_uri = None
        else:
            storage = "csv"
            storage_uri = str(results_dir)
        self.datacollector = DataCollector(
            model=self,
            model_reporters={
                "mean_sugar": lambda m: (
                    0.0 if len(m.sets[0]) == 0 else float(m.sets[0].df["sugar"].mean())
                ),
                "total_sugar": lambda m: (
                    float(m.sets[0].df["sugar"].sum()) if len(m.sets[0]) else 0.0
                ),
                "agents_alive": lambda m: float(len(m.sets[0])) if len(m.sets) else 0.0,
                "gini": gini,
                "corr_sugar_metabolism": corr_sugar_metabolism,
                "corr_sugar_vision": corr_sugar_vision,
            },
            agent_reporters={
                "sugar": "sugar",
                "metabolism": "metabolism",
                "vision": "vision",
            },
            storage=storage,
            storage_uri=storage_uri,
        )
        self.datacollector.collect()

    def _generate_sugar_grid(
        self, width: int, height: int, max_sugar: int
    ) -> pl.DataFrame:
        """Генерация сетки с сахаром рандомизировано

        Параметры
        ----------
        width : int    ширина (число колонок)
        height : int    высота (число рядов)
        max_sugar : int  макс.запас сахара для каждой ячейки

        Возвращает
        -------
        pl.DataFrame - датафрейм Polars
            DataFrame с колонками dim_0, dim_1, sugar (актуальное
            значение) и max_sugar (целевой максимум)
        """
        sugar_vals = self.random.integers(
            0, max_sugar + 1, size=(width, height), dtype=np.int64
        )
        dim_0 = pl.Series("dim_0", pl.arange(width, eager=True)).to_frame()
        dim_1 = pl.Series("dim_1", pl.arange(height, eager=True)).to_frame()
        return dim_0.join(dim_1, how="cross").with_columns(
            sugar=sugar_vals.flatten(), max_sugar=sugar_vals.flatten())

    def _generate_agent_frame(self, n_agents: int) -> pl.DataFrame:
        """Create the initial agent frame populated with agent traits.

        Parameters
        ----------
        n_agents : int
            Number of agents to create.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns ``sugar``, ``metabolism`` and ``vision``
            (integer values) for each agent.
        """
        rng = self.random
        dfpl = pl.DataFrame(
            {
                # верхняя граница integers() не включается: диапазоны совпадают
                # с random.randint() в sugarscape1 (6-25, 2-5, 1-6)
                "sugar": rng.integers(6, 26, size=n_agents, dtype=np.int64),
                "metabolism": rng.integers(2, 6, size=n_agents, dtype=np.int64),
                "vision": rng.integers(1, 7, size=n_agents, dtype=np.int64),
            })
        #dfpl = dfpl.with_columns(pl.int_range(pl.len()).alias("unique_id"))
        return dfpl.with_columns(pl.int_range(pl.len()).alias("unique_id"))

    def step(self) -> None:
        """Advance the model by one step.

        Notes
        -----
        The per-step ordering is important and this tutorial implements the
        classic Sugarscape "instant growback": agents move and eat first,
        and then empty cells are refilled immediately (move -> eat -> regrow
        -> collect).
        """
        if len(self.sets[0]) == 0:
            self.running = False
            return
        self.sets[0].step()
        self._advance_sugar_field()
        self.datacollector.collect()
        if len(self.sets[0]) == 0:
            self.running = False

    def run(self, steps: int) -> None:
        """Выполняем модель на заданное число шагов

        Параметры
        ----------
        steps : int
            макс.число шагов. Модель может закончится раньше
            если self.running установить в False (агентов не осталось).
        """
        for _ in range(steps):
            if not self.running:
                break
            self.step()

    def _advance_sugar_field(self) -> None:
        """Apply the instant-growback sugar regrowth rule.
        Empty cells (no agent present) are refilled to their max_sugar
        value. Cells that are occupied are set to zero because agents harvest
        the sugar when they eat. The method uses vectorised DataFrame joins
        and writes to keep the operation efficient.
        """
        empty_cells = self.space.empty_cells
        if not empty_cells.is_empty():
            # находим макс.сахар для каждой ячейки и восстанавливаем ее
            refresh = empty_cells.join(
                self._max_sugar, on=["dim_0", "dim_1"], how="left" )
            self.space.set_cells(refresh.select("dim_0", "dim_1", pl.col("max_sugar").alias("sugar")))
        full_cells = self.space.full_cells
        if not full_cells.is_empty():
            # занятые ячейки считаются опустошенными; sugar = 0
            zeros = pl.Series(np.zeros(len(full_cells), dtype=np.int64))
            self.space.set_cells(full_cells, {"sugar": zeros})

def simulate(
    *,
    agents: int,
    steps: int,
    width: int,
    height: int,
    max_sugar: int = 4,
    seed: int | None = None,
    results_dir: Path | None = None,
) -> FramesSimulationResult:

    model = Sugarscape(
        agent_type=AntsParallel,
        n_agents=agents,
        width=width,
        height=height,
        max_sugar=max_sugar,
        seed=seed,
        results_dir=results_dir,
    )
    model.run(steps)
    return FramesSimulationResult(datacollector=model.datacollector)

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

    print(f"Запуск модели Sugarscape(mesa-frames) с {agents} агентами на поле {width}x{height} за {steps} шагов")
    start_time = perf_counter()
    result = simulate(agents=agents, steps=steps, width=width, height=height, max_sugar=max_sugar, seed=seed)
    sim_time = perf_counter() - start_time
    model_pd = result.datacollector.data["model"].drop("batch").to_pandas()
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
                subtitle=f"mesa-frames; seed={model_pd['seed'].iat[0]}", agents=agents, steps=steps,
            )
        print(f"Сохранили графики в {plots_dir}")
        str_plot_dir = str(plots_dir)
    return display_pd, str_plot_dir
