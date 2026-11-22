# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os, math
from PIL import Image
# Импорт модели
import sugarscape1 as mdl

df1, df2 = 0, 0

# Функция для запуска имитации и получения данных в реальном времени
def run_simulation (params):
    """Запускает симуляцию с использованием параметров"""
    # Запуск моделирования
    results = mdl.run(config=params)
    return results

# Настройка интерфейса Streamlit
st.set_page_config(
   page_title="модель_1",
   page_icon=":shark:",
   layout="centered",
   initial_sidebar_state="expanded"
)
folder = "."
st.title("Муравьи в сахарном Мире")
ts1 = st.container()

with st.sidebar:
  st.page_link("start.py", label="***Стартовая страница***")
  with st.form(key="input_form"):
    st.write("*Параметры прогона модели*")
    # Параметры модели
    sel_width = st.slider("Размер в ширину", 100, 1000, 100)
    sel_height = st.slider("Размер в высоту", 100, 1000, 100)
    sel_num_agents = st.selectbox("Количество агентов", (1000,10000,50000,100000,150000,
200000,250000))
    sel_steps = st.slider("Длительность моделирования (шаг)", 10, 100, 10)
    sel_seed_select = st.radio('seed', [642, 791, 3085])
    sel_plot = st.radio('Графики', ['Без графиков', 'С графиками'])
    st.session_state.pr_click = st.form_submit_button(label="👍 Запуск модели")

# Подготовка параметров
params = {
    "agents": sel_num_agents,
    "width": sel_width,
    "height": sel_height,
    "plot": True if sel_plot =='С графиками' else False,
    "seed": sel_seed_select,
    "steps": sel_steps
}

# Контейнеры для графиков и метрик
tc1 = st.container()
tc1.graph1_placeholder = st.empty()

tc2 = st.container()
tc2.data1_placeholder = st.empty()

tc3 = st.container()
tc3.graph3_placeholder = st.empty()

# Кнопка для запуска симуляции
if st.session_state.pr_click:       # Запуск симуляции
    if sel_num_agents > sel_width * sel_height:
        st.error("Число агентов превышает число клеток.")
        st.stop()
    with st.spinner("♻ Запуск модели..." ):
       metrics, folder = run_simulation(params)
       ts1.success("Моделирование завершено!", icon="🗺")
    df1 = pd.DataFrame(metrics)

# Обновление графика
if isinstance(df1, pd.DataFrame):
   tc1.graph1_placeholder = st.line_chart(df1[["agents_alive"]])
   tc1.write ("🗺 Количество агентов в модели")

# Итоговые метрики
if isinstance(df1, pd.DataFrame):
   st.write("Итоговый журнал моделирования")
   tc2.data1_placeholder = st.dataframe(df1)

st.divider()

if params["plot"] == True:
  if folder != ".":
    images = []; cols=2
    for file in os.listdir(folder):
        if file.endswith(".png"):
            images.append(os.path.join(folder, file))
    if len(images) > 0:
        # Создаем сетку
        fig, axes =plt.subplots(nrows=math.ceil(len(images)/cols), ncols=cols, figsize=(6,6))
        for ax, img_path in zip(axes.flatten(), images):
            ax.imshow(Image.open(img_path))
            ax.axis('off')
        tc3.graph3_placeholder = st.pyplot(fig)
        plt.close(fig)

st.divider()
st.text("(^_^)")
