import json
import os
import sqlite3
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import base64
import subprocess
import threading
import queue
import sys
import io
import winsound
from matplotlib import pyplot as plt

import linearRegression

DB_PATH = 'fatigue_metrics.db'
class AppState:
    process = None

def start_main_py(output_queue):
    AppState.process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        bufsize=1
    )
    for line in AppState.process.stdout:
        output_queue.put(line.strip())

def process_output(output_queue):
    try:
        while True:
            line = output_queue.get_nowait()
            if line.startswith("[FRAME]"):
                frame_data = line.replace("[FRAME]", "")
                img_bytes = base64.b64decode(frame_data)
                image = Image.open(io.BytesIO(img_bytes))
                image = image.resize((400, 300))
                photo = ImageTk.PhotoImage(image)
                camera_label.configure(image=photo)
                camera_label.image = photo
            elif "Всё в порядке. Сохраняйте концентрацию и продолжайте движение." in line:
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
                # play sound
                os.system('start "" sound0.mp3')
                update_fatigue_prediction(DB_PATH)
            elif "Слегка снижена концентрация. Освежитесь: откройте окно, включите музыку или выпейте кофе." in line:
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
                # play sound
                os.system('start "" sound1.mp3')
                update_fatigue_prediction(DB_PATH)
            elif "Усталость нарастает. Остановитесь в ближайшем разрешённом месте и отдохните не менее 20 минут." in line:
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
                # play sound
                os.system('sound2.mp3')
                update_fatigue_prediction(DB_PATH)
            elif "Серьёзный риск! Срочно сбавьте скорость и найдите любое безопасное место для паузы." in line:
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
                # play sound
                os.system('sound3.mp3')
                update_fatigue_prediction(DB_PATH)
            elif "Критическая усталость! Немедленно прекратите движение и съезжайте на обочину!" in line:
                text_area.insert(tk.END, line + "\n")
                text_area.see(tk.END)
                # play sound
                os.system('sound4.mp3')
                update_fatigue_prediction(DB_PATH)


    except queue.Empty:
        pass
    root.after(100, process_output, output_queue)


def update_fatigue_prediction(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    try:
        # Проверяем количество записей
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM metrics")
        count = cur.fetchone()[0]

        if count > 5:
            # Получаем данные для прогноза
            df = pd.read_sql("""
                SELECT timestamp, fatigue_level 
                FROM metrics 
                ORDER BY timestamp DESC 
                LIMIT 30
            """, conn)

            # Вычисляем прогноз
            prediction = linearRegression.predict_time_to_threshold(df)

            if prediction['error'] is None:
                # Форматируем сообщение
                msg = (
                    f"Прогноз: критический уровень через {prediction['predicted_time']:.1f} мин\n"
                    f"Уверенность: {prediction['confidence'] * 100:.1f}% | Тренд: {prediction['current_trend']}\n"
                    "----------------------------------------\n"
                )

                # Обновляем текстовую область
                text_area.insert(tk.END, msg)
                text_area.see(tk.END)


    except Exception as e:
        print(f"Ошибка прогнозирования: {str(e)}")


def on_start():

    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    thread = threading.Thread(target=start_main_py, args=(output_queue,), daemon=True)
    thread.start()


def on_stop():
    global process
    if AppState.process:
        try:
            # Посылаем сигнал прерывания
            AppState.process.terminate()
            # Даем время на завершение
            AppState.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            # Принудительное завершение
            AppState.process.kill()
        finally:
            AppState.process = None

    show_summary()
    clear_db()
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    text_area.insert(tk.END, "Поездка завершена\n")



def clear_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM metrics')
    conn.commit()
    conn.close()




# Статистика поездки и построение графиков
def show_summary():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # Данные по усталости и эмоциям
    cur.execute('SELECT timestamp, fatigue_level, emotions FROM metrics')
    rows = cur.fetchall()
    conn.close()

    if not rows:
        summary_label.config(text='Нет данных о поездке')
        return

    # Длительность поездки
    t0 = datetime.fromisoformat(rows[0][0])
    t1 = datetime.fromisoformat(rows[-1][0])
    mins = (t1 - t0).seconds // 60
    summary_label.config(text=f'Ваша поездка продлилась {mins+1} минут(ы)')

    # График fatigue_level
    times = [(datetime.fromisoformat(r[0]) - t0).seconds/60 for r in rows]
    values = [r[1] for r in rows]
    plt.figure()
    plt.plot(times, values)
    plt.title('Уровень усталости')
    plt.xlabel('Время (мин)')
    plt.ylabel('Усталость')
    plt.grid(True)
    plt.savefig('fatigue_plot.png')
    plt.close()
    fatigue_img = Image.open('fatigue_plot.png')
    fatigue_photo = ImageTk.PhotoImage(fatigue_img)
    fatigue_graph_label.config(image=fatigue_photo)
    fatigue_graph_label.image = fatigue_photo

    emotion_translations = {
        "happy": "Радость",
        "sad": "Грусть",
        "angry": "Злость",
        "surprised": "Удивление",
        "neutral": "Нейтрально",
        "fear": "Страх",
        "disgust": "Отвращение"
    }

    # Обработка статистики эмоций
    emo_totals = {}
    for _, _, emo_json in rows:
        try:
            data = json.loads(emo_json)
            for emo, cnt in data.items():
                emo_ru = emotion_translations.get(emo, emo)  # Перевод или оставить как есть
                emo_totals[emo_ru] = emo_totals.get(emo_ru, 0) + cnt
        except Exception:
            continue

    # Круговая диаграмма эмоций
    labels = list(emo_totals.keys())
    sizes = list(emo_totals.values())
    if labels and sizes:
        plt.figure()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Статистика эмоций')
        plt.savefig('emotion_pie.png')
        plt.close()
        emo_img = Image.open('emotion_pie.png')
        emo_photo = ImageTk.PhotoImage(emo_img)
        emotion_graph_label.config(image=emo_photo)
        emotion_graph_label.image = emo_photo





def make_scrollable_frame(container):
    canvas = tk.Canvas(container, borderwidth=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    return scroll_frame



# Интерфейс
root = tk.Tk()
root.title("Система усталости")
root.geometry("800x600")

output_queue = queue.Queue()

camera_label = tk.Label(root)
camera_label.pack(pady=10)

ctrl_frame = tk.Frame(root)
ctrl_frame.pack(pady=5)
start_button = tk.Button(ctrl_frame, text="Старт", width=12, command=on_start)
start_button.pack(side=tk.LEFT, padx=5)
stop_button = tk.Button(ctrl_frame, text="Стоп", width=12, state=tk.DISABLED, command=on_stop)
stop_button.pack(side=tk.LEFT, padx=5)


text_area = tk.Text(root, wrap=tk.WORD, height=10, width=90)
text_area.pack(pady=10)

scroll_container = tk.Frame(root)
scroll_container.pack(fill="both", expand=True)
content_frame = make_scrollable_frame(scroll_container)

summary_label = tk.Label(content_frame, text='', justify=tk.LEFT)
summary_label.pack(fill=tk.X, padx=10, pady=5)

fatigue_graph_label = tk.Label(content_frame)
fatigue_graph_label.pack(pady=5)

emotion_graph_label = tk.Label(content_frame)
emotion_graph_label.pack(pady=5)



root.after(100, process_output, output_queue)
root.mainloop()
