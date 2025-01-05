import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import json

# Параметры
data_dir = 'train_data2'
model_save_path = os.path.join("Model", "kazakh_signs_resnet.3.0.keras")
img_height, img_width = 224, 224
batch_size = 32  # Увеличен размер батча
epochs = 10  # Увеличено количество эпох
validation_split = 0.2
initial_learning_rate = 0.001  # Увеличен начальный learning rate

# Создание директории Model если её нет
os.makedirs("Model", exist_ok=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def preprocess_input(x):
    try:
        # Преобразование в float32
        x = tf.cast(x, tf.float32)
        
        # Нормализация значений в диапазон [0, 1]
        x = x / 255.0
        
        # Преобразование в оттенки серого и обратно для уменьшения шума
        x = tf.image.rgb_to_grayscale(x)
        x = tf.image.grayscale_to_rgb(x)
        
        # Применяем предобработку ResNet
        x = tf.keras.applications.resnet_v2.preprocess_input(x * 255.0)
        
        return x
    except Exception as e:
        print(f"Ошибка в preprocess_input: {str(e)}")
        return x

# Улучшенный генератор данных с аугментацией
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,        # Угол поворота
    zoom_range=0.15,         # Зум
    width_shift_range=0.15,  # Сдвиг по ширине
    height_shift_range=0.15, # Сдвиг по высоте
    brightness_range=[0.85, 1.15],  # Яркость
    fill_mode="constant",
    cval=255,
    horizontal_flip=False,   
    vertical_flip=False,
    validation_split=validation_split
)

print("\nЗагрузка данных для обучения...")
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nНайдено классов: {num_classes}")
print("Классы:", train_generator.class_indices)

# Получаем метки классов
labels = list(train_generator.class_indices.keys())

def create_model():
    # Базовая модель ResNet50V2
    base_model = ResNet50V2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Замораживаем только начальные слои
    for layer in base_model.layers[:-60]:  # Размораживаем больше слоев
        layer.trainable = False
    
    # Создаем входной слой
    inputs = layers.Input(shape=(img_height, img_width, 3))
    
    # Применяем базовую модель
    x = base_model(inputs)
    
    # Добавляем слои для обработки признаков
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    # Первый блок с остаточным соединением
    block1 = layers.Dense(2048, kernel_regularizer=l2(0.01))(x)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('relu')(block1)
    block1 = layers.Dropout(0.5)(block1)
    
    # Второй блок с остаточным соединением
    block2 = layers.Dense(1024, kernel_regularizer=l2(0.01))(block1)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('relu')(block2)
    block2 = layers.Dropout(0.4)(block2)
    
    # Третий блок
    block3 = layers.Dense(512, kernel_regularizer=l2(0.01))(block2)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Activation('relu')(block3)
    block3 = layers.Dropout(0.3)(block3)
    
    # Объединяем все блоки
    x = layers.Concatenate()([block1, block2, block3])
    
    # Добавляем финальный слой с меньшим количеством нейронов
    x = layers.Dense(256, kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Выходной слой
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Создаем модель
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Создание модели
model = create_model()

# Компиляция модели с более высоким learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Улучшенные колбэки
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=25,  # Увеличено терпение
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,  # Более мягкое уменьшение learning rate
    patience=10,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Обучение модели
print("\nНачало обучения...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Построение графиков обучения
plt.figure(figsize=(15, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

# График функции потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Потери модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

# Сохранение графиков
plt.savefig(os.path.join("Model", "training_history.png"))
plt.close()

# Оценка модели
print("\nОценка финальной модели...")
loss, accuracy = model.evaluate(validation_generator)
print(f'Точность на проверочных данных: {accuracy * 100:.2f}%')

# Анализ ошибок на валидационном наборе
print("\nАнализ результатов на валидационном наборе...")
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes

# Создаем матрицу ошибок
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_classes, predicted_classes)

# Создаем отчет о классификации
print("\nСоздание отчета о классификации...")
report = classification_report(true_classes, predicted_classes, 
                            target_names=labels, 
                            output_dict=True)

# Анализ уверенности предсказаний
print("\nАнализ уверенности предсказаний:")
prediction_confidences = np.max(predictions, axis=1)
print(f"Средняя уверенность: {np.mean(prediction_confidences):.3f}")
print(f"Минимальная уверенность: {np.min(prediction_confidences):.3f}")
print(f"Максимальная уверенность: {np.max(prediction_confidences):.3f}")

# Сохраняем отчет в JSON
print("\nСохранение отчета...")
with open(os.path.join("Model", "classification_report.json"), 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

# Визуализация матрицы ошибок
print("\nСоздание визуализации матрицы ошибок...")
plt.figure(figsize=(20, 20))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha='right')
plt.yticks(tick_marks, labels)
plt.tight_layout()
plt.savefig(os.path.join("Model", "confusion_matrix.png"), bbox_inches='tight')
plt.close()

# Анализ наихудших предсказаний
print("\nАнализ наихудших предсказаний...")
class_accuracies = {}
class_confidences = {}
for i, label in enumerate(labels):
    mask = true_classes == i
    if np.any(mask):
        class_accuracies[label] = np.mean(predicted_classes[mask] == true_classes[mask])
        class_confidences[label] = np.mean(prediction_confidences[mask])

# Сортируем классы по точности
sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])

print("\nТоп-5 наихудших классов:")
for label, acc in sorted_accuracies[:5]:
    conf = class_confidences[label]
    print(f"{label}: Точность = {acc*100:.2f}%, Средняя уверенность = {conf*100:.2f}%")

# Сохранение меток
labels_save_path = os.path.join("Model", "labels.txt")
with open(labels_save_path, 'w', encoding='utf-8') as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"\nМетки сохранены в {labels_save_path}")

# Сохранение финальной модели
final_model_path = os.path.join("Model", "kazakh_signs_resnet_final.resNet.keras")
model.save(final_model_path)
print(f"\nФинальная модель сохранена в {final_model_path}")

# Сохранение конфигурации модели
model_config = {
    'preprocessing': {
        'target_size': [img_height, img_width],
        'rescale': 1./255,
    },
    'training_params': {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': initial_learning_rate,
    },
    'class_accuracies': {label: float(acc) for label, acc in class_accuracies.items()},
    'class_confidences': {label: float(conf) for label, conf in class_confidences.items()}
}

print("\nСохранение конфигурации модели...")
with open(os.path.join("Model", "model_config.json"), 'w', encoding='utf-8') as f:
    json.dump(model_config, f, indent=4, ensure_ascii=False)

print("\nОбучение и анализ завершены успешно!")