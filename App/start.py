# -*- coding: utf-8 -*-
import streamlit as st
import graphviz

# Настройка интерфейса Streamlit
st.set_page_config(
   page_title="Сахарный Мир",
   page_icon="🎲",
   layout="wide",
   initial_sidebar_state="expanded"
)

st.title(":clubs: Модель жизни муравьев в сахарном мире")
st.write("Моделирование процессов в мультиагентном формате")
st.sidebar.page_link("pages/model1.py", label="модель 1")
st.sidebar.page_link("pages/model2.py", label="модель 2")
st.divider();
st.write("Схема процесса")
dot_string = """
 digraph MG {
 rankdir="LR"
 K [label="агент", shape=ellipse, style=filled, fillcolor=cyan];
 A [label="Перемещение", shape=rectangle, style=filled, fillcolor=lightblue];
 B [label="Питание", shape=rectangle, style=filled, fillcolor=lightblue];
 C [label="Восстановление", shape=rectangle, style=filled, fillcolor=lightblue];
 D [label="Сбор", shape=circle, style=filled, fillcolor=white];
 E [label="Утиль", shape=rectangle, style=filled, fillcolor=white];

 A -> B [color=blue, style=bold];
 B -> C [color=blue, style=bold];
 C -> D [color=blue, style=bold];
 D -> A [color=blue, style=bold];
 C -> E [color=blue, style=bold];
 A -> K [color=green, style=dotted];
 B -> K [color=green, style=dotted];
 C -> K [color=green, style=dotted];
 D -> K [color=green, style=dotted];
 E -> K [color=green, style=dotted];
 }
"""
st.graphviz_chart(dot_string)
