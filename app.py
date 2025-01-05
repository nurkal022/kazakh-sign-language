import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math
import os
from collections import deque
import json

# Параметры
model_path = os.path.join("Model", "kazakh_signs_resnet_final.resNet.keras")
labels_path = os.path.join("Model", "labels.txt")
config_path = os.path.join("Model", "model_config.json")
offset = 20
img_size = 224
confidence_threshold = 0.5  # Увеличен порог для более уверенных предсказаний
prediction_queue_size = 3

print("\nИнициализация приложения...")
print(f"Используется модель: {model_path}")
print(f"Файл меток: {labels_path}")

# Загрузка конфигурации модели
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    print("Конфигурация модели загружена успешно")
except Exception as e:
    print(f"Ошибка при загрузке конфигурации: {str(e)}")
    model_config = None

# Проверка существования файлов
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель не найдена по пути: {model_path}. Пожалуйста, сначала запустите train_model.py")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Файл с метками не найден по пути: {labels_path}")

# Загрузка меток
try:
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    raise Exception(f"Ошибка при чтении файла меток: {str(e)}")

num_classes = len(labels)
if num_classes == 0:
    raise ValueError("Файл с метками пуст.")

# Загрузка модели
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\nМодель успешно загружена.")
    print(f"Количество классов в модели: {num_classes}")
    print("Доступные метки:", labels)
except Exception as e:
    raise Exception(f"Ошибка при загрузке модели: {str(e)}")

def preprocess_hand_image(imgWhite):
    try:
        # Преобразование в float32
        imgWhite = tf.cast(imgWhite, tf.float32)
        
        # Нормализация значений в диапазон [0, 1]
        imgWhite = imgWhite / 255.0
        
        # Преобразование в оттенки серого и обратно для уменьшения шума
        imgWhite = tf.image.rgb_to_grayscale(imgWhite)
        imgWhite = tf.image.grayscale_to_rgb(imgWhite)
        
        # Улучшение контраста
        imgWhite = tf.image.per_image_standardization(imgWhite)
        
        # Преобразование обратно в numpy для отображения
        imgWhite = tf.cast(imgWhite * 255, tf.uint8)
        imgWhite = imgWhite.numpy()
        
        return imgWhite
    except Exception as e:
        print(f"Ошибка в предобработке изображения: {str(e)}")
        return None

def get_averaged_prediction(predictions):
    if not predictions:
        return None, 0, []
    
    # Усреднение вероятностей
    avg_predictions = np.mean(predictions, axis=0)
    
    # Применяем softmax для нормализации вероятностей
    exp_preds = np.exp(avg_predictions - np.max(avg_predictions))
    avg_predictions = exp_preds / exp_preds.sum()
    
    # Получаем топ 4 предсказания
    top_4_indices = np.argsort(avg_predictions)[-4:][::-1]
    top_4_predictions = [(idx, avg_predictions[idx]) for idx in top_4_indices]
    
    max_confidence = avg_predictions[top_4_indices[0]]
    predicted_class = top_4_indices[0]
    
    return predicted_class, max_confidence, top_4_predictions

# Очередь для хранения последних предсказаний
prediction_queue = deque(maxlen=prediction_queue_size)

# Инициализация захвата видео и детектора рук
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Создание окна с трекбаром для порога уверенности
cv2.namedWindow('Settings')
cv2.createTrackbar('Confidence Threshold', 'Settings', int(confidence_threshold * 100), 100, lambda x: None)

print("\nУправление:")
print("Q - выход")
print("Используйте ползунок 'Confidence Threshold' для настройки порога уверенности")

frame_counter = 0
while True:
    success, img = cap.read()
    if not success:
        break
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    # Получаем текущее значение порога уверенности
    confidence_threshold = cv2.getTrackbarPos('Confidence Threshold', 'Settings') / 100.0
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        if w < 50 or h < 50:
            cv2.putText(imgOutput, "Move hand closer", (x, y - 10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            continue
            
        # Обрезка изображения руки
        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                     max(0, x - offset):min(img.shape[1], x + w + offset)]
        
        if imgCrop.size == 0:
            continue

        # Изменение размера и добавление отступов
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
        
        # Сохраняем пропорции при изменении размера
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = img_size / h
            wCal = math.ceil(k * w)
            if wCal > img_size:
                wCal = img_size
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            wGap = (img_size - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = img_size / w
            hCal = math.ceil(k * h)
            if hCal > img_size:
                hCal = img_size
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hGap = (img_size - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize
        
        # Предобработка изображения
        imgProcessed = preprocess_hand_image(imgWhite)
        if imgProcessed is None:
            continue
            
        # Подготовка изображения для модели
        imgFinal = tf.cast(imgProcessed, tf.float32) / 255.0
        imgFinal = np.expand_dims(imgFinal, axis=0)

        # Получение предсказания
        prediction = model.predict(imgFinal, verbose=0)
        prediction_queue.append(prediction[0])
        
        # Получение усредненного предсказания
        predicted_class, confidence, top_4_predictions = get_averaged_prediction(prediction_queue)
        
        if confidence >= confidence_threshold:
            # Создаем прямоугольник для фона предсказаний
            rect_height = 180
            rect_width = 300
            rect_x = max(x - offset, 10)
            rect_y = max(y - offset - rect_height, 10)
            
            # Рисуем полупрозрачный фон
            overlay = imgOutput.copy()
            cv2.rectangle(overlay, (rect_x, rect_y), 
                         (rect_x + rect_width, rect_y + rect_height), 
                         (255, 0, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, imgOutput, 0.3, 0, imgOutput)
            
            # Отображаем топ 4 предсказания с русскими буквами
            for i, (class_idx, conf) in enumerate(top_4_predictions):
                predicted_label = labels[class_idx]
                # Конвертируем латинские метки обратно в кириллицу
                cyrillic_label = predicted_label
                if predicted_label == 'AE': cyrillic_label = 'Ә'
                elif predicted_label == 'GH': cyrillic_label = 'Ғ'
                elif predicted_label == 'NG': cyrillic_label = 'Ң'
                elif predicted_label == 'OE': cyrillic_label = 'Ө'
                elif predicted_label == 'U_': cyrillic_label = 'Ұ'
                elif predicted_label == 'UE': cyrillic_label = 'Ү'
                elif predicted_label == 'H_': cyrillic_label = 'Һ'
                elif predicted_label == 'I_': cyrillic_label = 'І'
                elif predicted_label == 'Y_': cyrillic_label = 'Ы'
                elif predicted_label == 'E_': cyrillic_label = 'Э'
                
                text = f"{cyrillic_label} ({predicted_label}): {conf * 100:.1f}%"
                
                # Рисуем текст с обводкой
                font = cv2.FONT_HERSHEY_COMPLEX
                scale = 0.7
                thickness = 2
                color = (255, 255, 255)
                
                y_pos = rect_y + 30 + (i * 40)
                x_pos = rect_x + 10
                
                # Обводка текста
                cv2.putText(imgOutput, text, (x_pos, y_pos), font, scale, (0, 0, 0), thickness + 1)
                # Основной текст
                cv2.putText(imgOutput, text, (x_pos, y_pos), font, scale, color, thickness)

        # Отображение прямоугольника вокруг руки
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                     (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Отображение обработанного изображения руки
        cv2.imshow("Processed Hand", imgProcessed)
        cv2.imshow("Original Hand", imgWhite)

    # Добавляем информацию о пороге уверенности
    cv2.putText(imgOutput, f"Порог уверенности: {confidence_threshold:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Kazakh Sign Language Recognition", imgOutput)
    cv2.imshow('Settings', np.zeros((100, 400, 3), np.uint8))  # Окно настроек
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()