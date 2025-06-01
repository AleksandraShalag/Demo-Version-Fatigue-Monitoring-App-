import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from collections import deque, Counter
from fer import FER
import time
import sqlite3
import json
from datetime import datetime

from FatigueExpertSystem import FatigueFuzzySystem
from FatigueExpertSystem2 import RecommendationFuzzySystem

# Настройки
EAR_THRESH = 0.2
WINDOW_SEC = 60
FPS = 30
N = WINDOW_SEC * FPS

# Инициализация видеопотока
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть веб-камеру")

# Инициализация Dlib
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Очереди для метрик
closed_queue = deque(maxlen=N)
ear_log = []
perclos_log = []
mar_log = []
pose_log = []
emotion_log = []

# Инициализация FER
emotion_detector = FER(mtcnn=True)

# 3D модельные точки
model_points = np.array([
    (0.0, 0.0, 0.0),  # нос
    (0.0, -330.0, -65.0),  # подбородок
    (-225.0, 170.0, -135.0),  # левый глаз левый уголок
    (225.0, 170.0, -135.0),  # правый глаз правый уголок
    (-150.0, -150.0, -125.0),  # левый край рта
    (150.0, -150.0, -125.0)  # правый край рта
], dtype="double")

# Инициализация базы данных
conn = sqlite3.connect('fatigue_metrics.db')
cursor = conn.cursor()

cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp DATETIME,
            avr_ear REAL,
            avr_perclos REAL,
            avr_mar REAL,
            avg_pitch REAL,
            avg_yaw REAL,
            avg_roll REAL,
            emotions TEXT,
            fatigue_level REAL,
            risk_category TEXT,
            recommendations TEXT
        )
    ''')

cursor.execute('DELETE FROM metrics')
conn.commit()


def compute_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def compute_MAR(mouth):
    A = distance.euclidean(mouth[2], mouth[6])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)


def get_head_pose(shape, frame_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),
        (shape.part(8).x, shape.part(8).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y)
    ], dtype="double")
    h, w = frame_size
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_mat = np.hstack((rmat, translation_vector))
    angles = cv2.decomposeProjectionMatrix(proj_mat)[6]
    return angles.flatten()


def calculate_averages(ear_list, perclos_list, mar_list, pose_list, emotions):
    avg_ear = np.mean(ear_list) if ear_list else 0
    avg_perclos = perclos_list[-1] if perclos_list else 0
    avg_mar = np.mean(mar_list) if mar_list else 0

    if pose_list:
        pitchs, yaws, rolls = zip(*pose_list)
        avg_pitch = np.mean(np.abs(pitchs))
        avg_yaw = np.mean(np.abs(yaws))
        avg_roll = np.mean(np.abs(rolls))
    else:
        avg_pitch = avg_yaw = avg_roll = 0

    top_emotions = [max(e, key=e.get) for e in emotions if e]
    emotion_counts = Counter(top_emotions)

    return avg_ear, avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts


def expert_system( avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts):

    fatigue_fis = FatigueFuzzySystem()
    recommend_fis = RecommendationFuzzySystem()

    res1 = fatigue_fis.evaluate({
        'perclos': avg_perclos,
        'mar': avg_mar,
        'pitch': avg_pitch,
        'yaw': avg_yaw,
        'roll': avg_roll,
        'emotion_counts': emotion_counts
    })


    fat_val = res1['fatigue_level']
    risk_val = res1['risk_category']
    print(f"Усталость (0 - 100): {fat_val:.1f}, Риск(0 - 4): {risk_val:.1f}")

    # 2) Второй ФИС: получаем рекомендацию
    rec_val = recommend_fis.evaluate(fat_val, risk_val)
    labels = ["Всё в порядке. Сохраняйте концентрацию и продолжайте движение.",
        "Слегка снижена концентрация. Освежитесь: откройте окно, включите музыку или выпейте кофе.",
        "Усталость нарастает. Остановитесь в ближайшем разрешённом месте и отдохните не менее 20 минут.",
        "Серьёзный риск! Срочно сбавьте скорость и найдите любое безопасное место для паузы.",
        "Критическая усталость! Немедленно прекратите движение и съезжайте на обочину!" ]
    label = labels[int(round(rec_val))]
    print("Рекомендация:", label,flush=True)



    return fat_val, risk_val,label


def save_to_db(timestamp, avg_ear, avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts, fatigue_level,risk_category, recommendations):
    emotions_json = json.dumps(dict(emotion_counts))
    cursor.execute('''
        INSERT INTO metrics (
            timestamp, 
            avr_ear, 
            avr_perclos, 
            avr_mar, 
            avg_pitch, 
            avg_yaw, 
            avg_roll, 
            emotions,
            fatigue_level,
            risk_category,
            recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        avg_ear,
        avg_perclos,
        avg_mar,
        avg_pitch,
        avg_yaw,
        avg_roll,
        emotions_json,
        fatigue_level,
        risk_category,
        recommendations
    ))
    conn.commit()


# Таймер 60 секунд
cycle_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector_dlib(gray, 0)
    if rects:
        shape = predictor(gray, rects[0])
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        left_eye = coords[36:42]
        right_eye = coords[42:48]
        mouth_pts = coords[60:68]


        ear = (compute_EAR(left_eye) + compute_EAR(right_eye)) / 2.0
        mar = compute_MAR(mouth_pts)
        closed_queue.append(ear < EAR_THRESH)
        perclos = sum(closed_queue) / len(closed_queue)
        pitch, yaw, roll = get_head_pose(shape, frame.shape[:2])

        ear_log.append(ear)
        perclos_log.append(perclos)
        mar_log.append(mar)
        pose_log.append((pitch, yaw, roll))

        # Отображение
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Эмоции
    emotions = emotion_detector.detect_emotions(frame)
    detected = []
    for face in emotions:
        (x, y, w, h) = face["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        top_emotion = max(face["emotions"], key=face["emotions"].get)
        cv2.putText(frame, top_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        detected.append(face["emotions"])
    if detected:
        emotion_log.extend(detected)

    cv2.imshow("Monitoring", frame)

    # Проверка 60 секунд
    if time.time() - cycle_start >= WINDOW_SEC:
        print("\n=== Экспертная система: анализ прошедшей минуты ===")

        avg_ear, avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts = calculate_averages(
            ear_log, perclos_log, mar_log, pose_log, emotion_log
        )

        print("DEBUG INPUT:")
        print("avg_perclos:", avg_perclos)
        print("avg_mar:", avg_mar)
        print("avg_pitch:", avg_pitch)
        print("avg_yaw:", avg_yaw)
        print("avg_roll:", avg_roll)
        print("emotion_counts:", emotion_counts)

        fat_val, risk_val, res_val = expert_system( avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts)
        current_time = datetime.now()
        save_to_db(current_time, avg_ear, avg_perclos, avg_mar, avg_pitch, avg_yaw, avg_roll, emotion_counts,fat_val,risk_val,res_val)

        ear_log.clear()
        perclos_log.clear()
        mar_log.clear()
        pose_log.clear()
        emotion_log.clear()
        cycle_start = time.time()

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()
conn.commit()
conn.close()
