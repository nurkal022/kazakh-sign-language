import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
import json

# Казахский алфавит и настройки для сбора данных
KAZAKH_ALPHABET = [
    'А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е',  'Ж', 'З', 'И', 
    'Й', 'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С', 
    'Т', 'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ',
    'Ы', 'І', 'Ь', 'Э', 'Ю', 'Я'
]

# Словарь соответствия кириллических букв и латинских названий папок
FOLDER_MAPPING = {
    'А': 'A', 'Ә': 'AE', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Ғ': 'GH',
    'Д': 'D', 'Е': 'E',  'Ж': 'ZH', 'З': 'Z', 'И': 'I',
    'Й': 'Y', 'К': 'K', 'Қ': 'Q', 'Л': 'L', 'М': 'M', 'Н': 'N',
    'Ң': 'NG', 'О': 'O', 'Ө': 'OE', 'П': 'P', 'Р': 'R', 'С': 'S',
    'Т': 'T', 'У': 'U', 'Ұ': 'U_', 'Ү': 'UE', 'Ф': 'F', 'Х': 'H',
    'Һ': 'H_', 'Ц': 'TS', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SCH', 'Ъ': 'TB',
    'Ы': 'Y_', 'І': 'I_', 'Ь': 'MS', 'Э': 'E_', 'Ю': 'YU', 'Я': 'YA'
}

# Создаем обратное отображение (из латиницы в кириллицу)
REVERSE_MAPPING = {v: k for k, v in FOLDER_MAPPING.items()}

# Параметры
IMAGES_PER_CLASS = 100  # Желаемое количество изображений для каждой буквы
BASE_FOLDER = "train_data2"
CONFIG_FILE = "dataset_progress.json"

class DatasetCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1, detectionCon=0.8)
        self.offset = 20
        self.imgSize = 224
        self.min_hand_size = 50
        self.capture_delay = 0.5
        self.last_capture_time = 0
        self.imgWhite = None
        
        # Создаем базовую директорию
        os.makedirs(BASE_FOLDER, exist_ok=True)
        
        # Создаем все необходимые поддиректории
        for letter in KAZAKH_ALPHABET:
            folder_name = FOLDER_MAPPING[letter]
            os.makedirs(os.path.join(BASE_FOLDER, folder_name), exist_ok=True)
        
        # Загружаем или создаем прогресс
        self.progress = self.load_progress()
        
        # Выводим текущий прогресс
        print("\nТекущий прогресс:")
        for letter, count in self.progress.items():
            if count > 0:
                folder_name = FOLDER_MAPPING.get(letter, letter)
                print(f"Буква {letter} ({folder_name}): {count}/{IMAGES_PER_CLASS}")
        
        # Текущая буква для сбора данных
        self.current_letter = self.get_next_letter()
        self.counter = self.progress.get(self.current_letter, 0) if self.current_letter else 0
        
        if self.current_letter:
            print(f"\nНачинаем сбор данных с буквы: {self.current_letter} (папка: {FOLDER_MAPPING.get(self.current_letter, '')})")
            print("Управление:")
            print("S - сохранить изображение")
            print("N - пропустить текущую букву")
            print("R - повторить последнее сохранение")
            print("Q - выход")
        else:
            print("\nВсе данные собраны!")

    def load_progress(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    # Конвертируем латинские ключи в кириллические, если нужно
                    converted_progress = {}
                    for key, value in progress.items():
                        if key in REVERSE_MAPPING:  # Если ключ - латинский
                            converted_progress[REVERSE_MAPPING[key]] = value
                        elif key in FOLDER_MAPPING:  # Если ключ - кириллический
                            converted_progress[key] = value
                    return converted_progress
            except Exception as e:
                print(f"Ошибка при загрузке прогресса: {str(e)}")
                return {letter: 0 for letter in KAZAKH_ALPHABET}
        return {letter: 0 for letter in KAZAKH_ALPHABET}

    def save_progress(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении прогресса: {str(e)}")

    def get_next_letter(self):
        for letter in KAZAKH_ALPHABET:
            if self.progress.get(letter, 0) < IMAGES_PER_CLASS:
                return letter
        return None

    def preprocess_hand_image(self, imgWhite):
        imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        imgWhite = cv2.GaussianBlur(imgWhite, (3, 3), 0)
        
        lab = cv2.cvtColor(imgWhite, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        imgWhite = cv2.merge((cl, a, b))
        imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_LAB2RGB)
        imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_RGB2BGR)
        
        return imgWhite

    def resize_and_pad(self, img_crop, target_size):
        h, w = img_crop.shape[:2]
        aspect = h / w
        
        if aspect > 1:
            new_h = target_size
            new_w = int(target_size / aspect)
            resized = cv2.resize(img_crop, (new_w, new_h))
            padded = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
            pad_w = (target_size - new_w) // 2
            padded[:, pad_w:pad_w + new_w] = resized
        else:
            new_w = target_size
            new_h = int(target_size * aspect)
            resized = cv2.resize(img_crop, (new_w, new_h))
            padded = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
            pad_h = (target_size - new_h) // 2
            padded[pad_h:pad_h + new_h, :] = resized
        
        return padded

    def create_info_display(self, img):
        # Создаем информационную панель
        info_height = 120
        info_img = np.ones((info_height, img.shape[1], 3), dtype=np.uint8) * 255
        
        # Текущая буква и прогресс
        folder_name = FOLDER_MAPPING.get(self.current_letter, '')
        cv2.putText(info_img, f"Буква: {self.current_letter} ({folder_name})", (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(info_img, f"Прогресс: {self.counter}/{IMAGES_PER_CLASS}", (10, 60),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        
        # Общий прогресс
        total_progress = sum(self.progress.values())
        total_needed = len(KAZAKH_ALPHABET) * IMAGES_PER_CLASS
        progress_percent = (total_progress / total_needed) * 100
        cv2.putText(info_img, f"Общий прогресс: {progress_percent:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        
        # Объединяем с основным изображением
        return np.vstack([info_img, img])

    def save_image(self, force_save=False):
        if self.imgWhite is None:
            print("Ошибка: нет изображения для сохранения")
            return False
            
        try:
            # Получаем латинское название папки для текущей буквы
            folder_name = FOLDER_MAPPING[self.current_letter]
            # Создаем директорию для текущей буквы
            letter_folder = os.path.join(BASE_FOLDER, folder_name)
            os.makedirs(letter_folder, exist_ok=True)
            
            # Сохраняем изображение
            filename = os.path.join(letter_folder, f"Image_{time.time():.3f}.jpg")
            success = cv2.imwrite(filename, self.imgWhite)
            
            if success:
                self.counter += 1
                self.progress[self.current_letter] = self.counter
                self.save_progress()
                
                print(f"Сохранено изображение {self.counter}/{IMAGES_PER_CLASS} для буквы {self.current_letter} (папка: {folder_name})")
                
                # Проверяем, нужно ли перейти к следующей букве
                if self.counter >= IMAGES_PER_CLASS:
                    prev_letter = self.current_letter
                    self.current_letter = self.get_next_letter()
                    self.counter = self.progress.get(self.current_letter, 0) if self.current_letter else 0
                    
                    if self.current_letter:
                        print(f"\nЗавершен сбор данных для буквы {prev_letter}")
                        print(f"Переход к букве: {self.current_letter} (папка: {FOLDER_MAPPING[self.current_letter]})")
                return True
            else:
                print(f"Ошибка при сохранении изображения в папку {folder_name}")
                return False
                
        except Exception as e:
            print(f"Ошибка при сохранении изображения: {str(e)}")
            self.counter -= 1
            return False

    def skip_to_next_letter(self):
        """Функция для пропуска текущей буквы и перехода к следующей"""
        if self.current_letter is None:
            print("Нет доступных букв для сбора данных")
            return False
            
        prev_letter = self.current_letter
        prev_folder = FOLDER_MAPPING[prev_letter]
        
        # Получаем следующую букву
        self.current_letter = self.get_next_letter()
        
        if self.current_letter is None:
            print("\nВсе буквы собраны!")
            return False
            
        # Обновляем счетчик для новой буквы
        self.counter = self.progress.get(self.current_letter, 0)
        current_folder = FOLDER_MAPPING[self.current_letter]
        
        print(f"\nПропущена буква: {prev_letter} (папка: {prev_folder})")
        print(f"Переход к букве: {self.current_letter} (папка: {current_folder})")
        print(f"Текущий прогресс для буквы {self.current_letter}: {self.counter}/{IMAGES_PER_CLASS}")
        return True

    def run(self):
        last_saved_image = None  # Для функции повтора
        
        while True:
            if self.current_letter is None:
                print("Сбор данных завершен для всех букв!")
                break

            success, img = self.cap.read()
            if not success:
                print("Ошибка при захвате кадра")
                break

            hands, img = self.detector.findHands(img)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                if w < self.min_hand_size or h < self.min_hand_size:
                    cv2.putText(img, "Move hand closer", (x, y - 10),
                               cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    self.imgWhite = None
                else:
                    try:
                        imgCrop = img[max(0, y - self.offset):min(img.shape[0], y + h + self.offset),
                                    max(0, x - self.offset):min(img.shape[1], x + w + self.offset)]
                        
                        if imgCrop.size > 0:
                            self.imgWhite = self.resize_and_pad(imgCrop, self.imgSize)
                            self.imgWhite = self.preprocess_hand_image(self.imgWhite)
                            
                            cv2.imshow("ImageCrop", imgCrop)
                            cv2.imshow("ImageWhite", self.imgWhite)
                            
                            # Отображаем прямоугольник вокруг руки
                            cv2.rectangle(img, (x - self.offset, y - self.offset),
                                        (x + w + self.offset, y + h + self.offset),
                                        (255, 0, 255), 4)
                    
                    except Exception as e:
                        print(f"Ошибка обработки изображения: {str(e)}")
                        self.imgWhite = None
                        continue

            # Создаем информационный дисплей
            display_img = self.create_info_display(img)
            cv2.imshow("Data Collection", display_img)
            
            key = cv2.waitKey(1)
            
            current_time = time.time()
            if key == ord("s") and (current_time - self.last_capture_time) >= self.capture_delay:
                if self.save_image():
                    self.last_capture_time = current_time
                    last_saved_image = self.imgWhite.copy() if self.imgWhite is not None else None
            
            elif key == ord("r") and last_saved_image is not None:  # Повторить последнее сохранение
                self.imgWhite = last_saved_image.copy()
                if self.save_image():
                    print("Повторное сохранение успешно")
            
            elif key == ord("n"):  # Пропустить текущую букву
                if not self.skip_to_next_letter():
                    break  # Если все буквы собраны, завершаем программу
            
            elif key == ord("q"):
                print("\nЗавершение сбора данных...")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nИтоговая статистика:")
        for letter, count in self.progress.items():
            if count > 0:  # Показываем только буквы, для которых есть данные
                folder_name = FOLDER_MAPPING[letter]
                print(f"Буква {letter} ({folder_name}): {count}/{IMAGES_PER_CLASS}")

if __name__ == "__main__":
    collector = DatasetCollector()
    collector.run()