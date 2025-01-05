import os
import tensorflow as tf
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Параметры
model_path = "Model/kazakh_signs_resnet.3.0.keras"
labels_path = "Model/labels.txt"
font_path = "font.ttf"  # Путь к шрифту
img_height, img_width = 224, 224
offset = 20
prediction_delay = 0.5  # Задержка между предсказаниями в секундах
prediction_queue_size = 5  # Размер очереди для усреднения предсказаний
initial_confidence = 0.1  # Начальный порог уверенности

# Словарь для конвертации латинских меток в кириллицу
LABEL_MAPPING = {
    'A': 'А', 'AE': 'Ә', 'B': 'Б', 'V': 'В', 'G': 'Г', 'GH': 'Ғ',
    'D': 'Д', 'E': 'Е', 'ZH': 'Ж', 'Z': 'З', 'I': 'И',
    'Y': 'Й', 'K': 'К', 'Q': 'Қ', 'L': 'Л', 'M': 'М', 'N': 'Н',
    'NG': 'Ң', 'O': 'О', 'OE': 'Ө', 'P': 'П', 'R': 'Р', 'S': 'С',
    'T': 'Т', 'U': 'У', 'U_': 'Ұ', 'UE': 'Ү', 'F': 'Ф', 'H': 'Х',
    'H_': 'Һ', 'TS': 'Ц', 'CH': 'Ч', 'SH': 'Ш', 'SCH': 'Щ', 'TB': 'Ъ',
    'Y_': 'Ы', 'I_': 'І', 'MS': 'Ь', 'E_': 'Э', 'YU': 'Ю', 'YA': 'Я'
}

def load_model_and_labels():
    # Загрузка меток
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Файл с метками не найден по пути: {labels_path}")

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    # Проверка соответствия меток
    if len(labels) == 0:
        raise ValueError("Файл с метками пуст.")

    # Загрузка модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print("Модель успешно загружена.")
    
    return model, labels

def preprocess_image(img):
    # Преобразование в RGB если нужно
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Изменение размера с сохранением пропорций
    h, w = img.shape[:2]
    aspect = h / w
    
    if aspect > 1:
        new_h = img_height
        new_w = int(img_height / aspect)
        if new_w > img_width:
            new_w = img_width
        resized = cv2.resize(img, (new_w, new_h))
        imgWhite = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        wGap = (img_width - new_w) // 2
        imgWhite[:, wGap:wGap + new_w] = resized
    else:
        new_w = img_width
        new_h = int(img_width * aspect)
        if new_h > img_height:
            new_h = img_height
        resized = cv2.resize(img, (new_w, new_h))
        imgWhite = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        hGap = (img_height - new_h) // 2
        imgWhite[hGap:hGap + new_h, :] = resized

    # Преобразование в тензор и нормализация
    img_array = tf.cast(imgWhite, tf.float32)
    img_array = img_array / 255.0
    
    # Преобразование в оттенки серого и обратно для уменьшения шума
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.image.grayscale_to_rgb(img_array)
    
    # Применяем предобработку ResNet
    img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array * 255.0)
    
    # Добавляем размерность батча
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array, imgWhite

def get_cyrillic_label(latin_label):
    return LABEL_MAPPING.get(latin_label, latin_label)

def put_text_with_custom_font(img, text, position, font_size, color):
    # Создаем изображение PIL из массива numpy
    img_pil = Image.fromarray(img)
    
    # Создаем объект для рисования
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Загружаем шрифт
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Ошибка загрузки шрифта: {str(e)}")
        # Используем стандартный шрифт если не удалось загрузить кастомный
        font = ImageFont.load_default()
    
    # Рисуем текст
    draw.text(position, text, font=font, fill=color)
    
    # Конвертируем обратно в numpy array
    return np.array(img_pil)

def get_averaged_prediction(predictions_queue):
    if not predictions_queue:
        return None
    
    # Усреднение вероятностей
    avg_predictions = np.mean(predictions_queue, axis=0)
    
    # Получаем топ-3 предсказания
    top_3_indices = np.argsort(avg_predictions)[-3:][::-1]
    top_3_predictions = [(idx, avg_predictions[idx]) for idx in top_3_indices]
    
    return top_3_predictions

def draw_predictions(img, predictions, labels, x, y):
    # Создаем полупрозрачный фон для текста
    overlay = img.copy()
    
    # Размеры прямоугольника для фона
    rect_height = 120  # Высота для 3 предсказаний
    rect_width = 300
    rect_x = max(x - offset, 10)
    rect_y = max(y - offset - rect_height - 10, 10)
    
    # Рисуем полупрозрачный фон
    cv2.rectangle(overlay, 
                 (rect_x, rect_y),
                 (rect_x + rect_width, rect_y + rect_height),
                 (255, 0, 255), 
                 cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    
    # Отображаем предсказания
    for i, (idx, conf) in enumerate(predictions):
        label = labels[idx]
        cyrillic = get_cyrillic_label(label)
        text = f"{cyrillic} ({label}): {conf * 100:.1f}%"
        
        # Позиция для текста
        y_pos = rect_y + 30 + (i * 35)
        x_pos = rect_x + 10
        
        # Рисуем текст с использованием кастомного шрифта
        img = put_text_with_custom_font(
            img, 
            text,
            (x_pos, y_pos),
            24,  # Размер шрифта
            (255, 255, 255)  # Цвет текста (белый)
        )
    
    return img

def main():
    try:
        # Проверяем наличие шрифта
        if not os.path.exists(font_path):
            print(f"Предупреждение: файл шрифта не найден по пути: {font_path}")
            print("Будет использован стандартный шрифт системы")
        
        # Загрузка модели и меток
        model, labels = load_model_and_labels()
        
        # Инициализация камеры и детектора рук
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1, detectionCon=0.8)
        
        # Очередь для хранения предсказаний
        predictions_queue = deque(maxlen=prediction_queue_size)
        last_prediction_time = 0
        
        print("\nУправление:")
        print("+ - увеличить порог уверенности")
        print("- - уменьшить порог уверенности")
        print("Q - выход")
        
        # Создаем окно и трекбар для порога уверенности
        cv2.namedWindow('Kazakh Sign Language Recognition')
        cv2.createTrackbar('Confidence Threshold', 'Kazakh Sign Language Recognition',
                          int(initial_confidence * 100), 100, lambda x: None)
        
        # Текущий порог уверенности
        confidence_threshold = initial_confidence
        
        while True:
            success, img = cap.read()
            if not success:
                print("Ошибка при получении кадра с камеры")
                break
            
            # Обновляем порог уверенности из трекбара
            confidence_threshold = cv2.getTrackbarPos('Confidence Threshold',
                                                    'Kazakh Sign Language Recognition') / 100.0
            
            # Находим руки на изображении
            hands, img = detector.findHands(img)
            
            current_time = time.time()
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Проверяем размер области руки
                if w < 50 or h < 50:
                    img = put_text_with_custom_font(
                        img,
                        "Move hand closer",
                        (x, y - 10),
                        24,
                        (0, 0, 255)
                    )
                else:
                    # Рисуем прямоугольник вокруг руки
                    cv2.rectangle(img, (x - offset, y - offset),
                                (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    
                    # Делаем предсказание с задержкой
                    if current_time - last_prediction_time >= prediction_delay:
                        # Обрезаем изображение руки
                        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                                    max(0, x - offset):min(img.shape[1], x + w + offset)]
                        
                        if imgCrop.size > 0:
                            # Предобработка изображения
                            img_array, _ = preprocess_image(imgCrop)
                            
                            # Получение предсказаний
                            predictions = model.predict(img_array, verbose=0)[0]
                            predictions_queue.append(predictions)
                            
                            last_prediction_time = current_time
                    
                    # Получаем усредненные предсказания
                    if predictions_queue:
                        avg_predictions = get_averaged_prediction(predictions_queue)
                        if avg_predictions[0][1] >= confidence_threshold:
                            img = draw_predictions(img, avg_predictions, labels, x, y)
            
            # Отображаем текущий порог уверенности
            img = put_text_with_custom_font(
                img,
                f"Confidence: {confidence_threshold:.2f}",
                (10, 30),
                24,
                (0, 255, 0)
            )
            
            # Отображаем изображение
            cv2.imshow("Kazakh Sign Language Recognition", img)
            
            # Обработка клавиш
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('+'):
                current = cv2.getTrackbarPos('Confidence Threshold',
                                           'Kazakh Sign Language Recognition')
                cv2.setTrackbarPos('Confidence Threshold',
                                 'Kazakh Sign Language Recognition',
                                 min(current + 5, 100))
            elif key == ord('-'):
                current = cv2.getTrackbarPos('Confidence Threshold',
                                           'Kazakh Sign Language Recognition')
                cv2.setTrackbarPos('Confidence Threshold',
                                 'Kazakh Sign Language Recognition',
                                 max(current - 5, 0))
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print("\nПрограмма завершена.") 