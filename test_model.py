import os
import tensorflow as tf
import numpy as np
import cv2
from tkinter import Tk, filedialog
import json
from cvzone.HandTrackingModule import HandDetector

# Параметры
model_path = "Model/kazakh_signs_resnet_final.resNet.keras"
labels_path = "Model/labels.txt"
config_path = "Model/model_config.json"
img_height, img_width = 224, 224
offset = 20

def load_model_and_labels():
    # Загрузка меток
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Файл с метками не найден по пути: {labels_path}")

    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    # Проверка соответствия меток
    num_classes = len(labels)
    if num_classes == 0:
        raise ValueError("Файл с метками пуст.")

    # Загрузка модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print("Модель успешно загружена.")
    
    return model, labels

def preprocess_image(img):
    # Преобразование в RGB если нужно
    if len(img.shape) == 2:  # Если изображение в оттенках серого
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Если изображение с альфа-каналом
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
    # Словарь для конвертации латинских меток в кириллицу
    label_mapping = {
        'AE': 'Ә', 'GH': 'Ғ', 'NG': 'Ң', 'OE': 'Ө',
        'U_': 'Ұ', 'UE': 'Ү', 'H_': 'Һ', 'I_': 'І',
        'Y_': 'Ы', 'E_': 'Э', 'ZH': 'Ж', 'CH': 'Ч',
        'SH': 'Ш', 'SCH': 'Щ', 'YU': 'Ю', 'YA': 'Я',
        'TB': 'Ъ', 'MS': 'Ь'
    }
    return label_mapping.get(latin_label, latin_label)

def show_predictions(predictions, labels, original_img):
    # Получаем топ-5 предсказаний
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    
    print("\nРезультаты распознавания:")
    print("-" * 50)
    
    # Отображаем изображение и предсказания
    plt_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Input Image", plt_img)
    
    # Создаем окно для результатов
    result_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    for i, idx in enumerate(top_5_indices):
        label = labels[idx]
        cyrillic = get_cyrillic_label(label)
        confidence = predictions[0][idx] * 100
        
        # Выводим в консоль
        print(f"{cyrillic} ({label}): {confidence:.2f}%")
        
        # Добавляем текст на изображение
        text = f"{cyrillic} ({label}): {confidence:.2f}%"
        cv2.putText(result_img, text, (10, 40 + i * 50),
                   cv2.FONT_HERSHEY_COMPLEX, 0.7,
                   (0, 0, 0), 2)
    
    cv2.imshow("Predictions", result_img)

def predict_from_file():
    try:
        # Загрузка модели и меток
        model, labels = load_model_and_labels()
        
        # Создаем окно выбора файла
        Tk().withdraw()
        image_path = filedialog.askopenfilename(
            title="Выберите изображение для распознавания",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if not image_path:
            print("Файл не выбран")
            return
        
        # Загрузка и предобработка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Предобработка изображения
        img_array, original_img = preprocess_image(img)
        
        # Получение предсказаний
        predictions = model.predict(img_array, verbose=0)
        
        # Показываем результаты
        show_predictions(predictions, labels, original_img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

def predict_from_camera():
    try:
        # Загрузка модели и меток
        model, labels = load_model_and_labels()
        
        # Инициализация камеры и детектора рук
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1, detectionCon=0.8)
        
        print("\nУправление:")
        print("SPACE - сделать снимок")
        print("Q - выход")
        
        while True:
            success, img = cap.read()
            if not success:
                print("Ошибка при получении кадра с камеры")
                break
            
            # Находим руки на изображении
            hands, img = detector.findHands(img)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Проверяем размер области руки
                if w < 50 or h < 50:
                    cv2.putText(img, "Move hand closer", (x, y - 10),
                               cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    # Рисуем прямоугольник вокруг руки
                    cv2.rectangle(img, (x - offset, y - offset),
                                (x + w + offset, y + h + offset), (255, 0, 255), 4)
            
            # Отображаем изображение с камеры
            cv2.imshow("Camera", img)
            
            # Обработка клавиш
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            elif key == ord(' ') and hands:  # Пробел для захвата
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Обрезаем изображение руки
                imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                             max(0, x - offset):min(img.shape[1], x + w + offset)]
                
                if imgCrop.size > 0:
                    # Предобработка изображения
                    img_array, original_img = preprocess_image(imgCrop)
                    
                    # Получение предсказаний
                    predictions = model.predict(img_array, verbose=0)
                    
                    # Показываем результаты
                    show_predictions(predictions, labels, original_img)
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    while True:
        print("\nВыберите режим:")
        print("1 - Распознать изображение из файла")
        print("2 - Распознать жест с камеры")
        print("0 - Выход")
        
        choice = input("Ваш выбор: ")
        
        if choice == "1":
            predict_from_file()
        elif choice == "2":
            predict_from_camera()
        elif choice == "0":
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
    
    print("\nПрограмма завершена.")