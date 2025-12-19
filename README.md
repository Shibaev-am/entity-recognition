# BERT NER with Triton Inference Server

Этот проект реализует полный цикл MLOps для задачи Named Entity Recognition (NER) с использованием BERT: обучение, экспорт в ONNX, деплой через Triton Inference Server и UI на Streamlit.

## 1. Подготовка окружения

Для начала установите зависимости через Poetry и настройте Docker.

```bash
# Установка Python зависимостей
poetry install

# Активация окружения (опционально, команды ниже используют poetry run)
source $(poetry env info --path)/bin/activate
```

Для работы Triton Server на GPU убедитесь, что установлен [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## 2. Обучение и экспорт модели

Все команды проекта доступны через единую точку входа `ner.commands`.

### 2.1. Обучение модели
```bash
# Запуск обучения (параметры конфигурируются в configs/config.yaml)
python -m ner.commands train

# С переопределением параметров Hydra
python -m ner.commands train trainer.max_epochs=5 model.lr=1e-5
```
Чекпоинт модели сохранится в папку `models/`.

### 2.2. Экспорт в ONNX
Скрипт автоматически найдет последний чекпоинт и сконвертирует его.
```bash
python -m ner.commands to-onnx
```
Это создаст файлы `models/model.onnx` и `models/model.onnx.data`.

### 2.3. Локальный инференс
```bash
python -m ner.commands infer
```

## 3. Подготовка Triton Model Repository

Используйте команду для автоматического копирования модели:

```bash
python -m ner.commands prepare-triton
```

Или выполните вручную:
```bash
# Создаем структуру папок
mkdir -p model_repository/bert_ner/1

# Копируем ONNX модель
cp models/model.onnx models/model.onnx.data model_repository/bert_ner/1/

# (Конфиг model_repository/bert_ner/config.pbtxt уже должен быть создан в репозитории)
```

## Структура проекта

```
.
├── ner/                # Основной пакет проекта
│   ├── __init__.py
│   ├── commands.py     # Единая точка входа для всех команд
│   ├── dataset.py      # Загрузка и обработка данных
│   ├── model.py        # Модель BERT NER
│   └── utils.py        # Вспомогательные функции
├── apps/               # Streamlit приложения (UI)
│   ├── local_run.py    # Локальный инференс (HuggingFace Pipeline)
│   └── triton_run.py   # Клиент для Triton Inference Server
├── configs/            # Конфигурации Hydra (.yaml)
├── data/               # Данные для обучения
├── model_repository/   # Репозиторий моделей для Triton
│   └── bert_ner/       # Конфигурация и веса модели
│       ├── 1/          # Версия модели
│       │   └── model.onnx
│       └── config.pbtxt
├── models/             # Локальные артефакты обучения (.ckpt, .onnx)
├── outputs/            # Логи обучения (Hydra)
└── scripts/            # Скрипты (train, export)
```

## 4. Запуск Triton Inference Server

Для запуска сервера используется Docker. Убедитесь, что вы находитесь в корне проекта.

**Команда запуска (GPU):**
```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

**Разбор команды:**
*   `--gpus all`: Использовать все доступные GPU.
*   `--rm`: Удалить контейнер после остановки.
*   `-p 8000:8000`: Проброс HTTP порта (для REST API).
*   `-p 8001:8001`: Проброс GRPC порта.
*   `-p 8002:8002`: Проброс порта метрик.
*   `-v $(pwd)/model_repository:/models`: Монтирование локальной папки `model_repository` внутрь контейнера в `/models`.
*   `nvcr.io/nvidia/tritonserver:24.05-py3`: Образ Docker (версия должна поддерживать версию Opset вашего ONNX файла).

**Вариант запуска на CPU:**
Если GPU недоступен, просто уберите флаг `--gpus all`.

Дождитесь в логах сообщения: `Started HTTPService at 0.0.0.0:8000`.

## 5. Запуск UI приложения

В отдельном терминале запустите Streamlit приложение:

```bash
# Демо с Triton Inference Server
python -m ner.commands demo-triton

# Или локальное демо (без Triton)
python -m ner.commands demo-local
```

Приложение будет доступно по адресу: `http://localhost:8501`.

## Все доступные команды

Справка по всем командам:

```bash
python -m ner.commands --help
```

### `train` — Обучение модели

Запускает обучение BERT NER модели. Параметры конфигурируются в `configs/config.yaml` или передаются через Hydra overrides.

```bash
# Базовый запуск
python -m ner.commands train

# С переопределением параметров
python -m ner.commands train trainer.max_epochs=5 model.lr=1e-5

# Изменение batch_size и количества воркеров
python -m ner.commands train data.batch_size=32 data.num_workers=4
```

### `to-onnx` — Экспорт в ONNX формат

Конвертирует обученный чекпоинт (`.ckpt`) в ONNX формат для деплоя.

```bash
# Базовый запуск (найдёт последний чекпоинт автоматически)
python -m ner.commands to-onnx

# С указанием другой директории моделей
python -m ner.commands to-onnx paths.model_save_dir=./other_models
```

### `infer` — Локальный инференс

Тестовый запуск инференса на ONNX модели с примером текста.

```bash
python -m ner.commands infer
```

### `prepare-triton` — Подготовка Triton Model Repository

Копирует ONNX модель в структуру папок, необходимую для Triton Inference Server.

```bash
python -m ner.commands prepare-triton
```

### `demo-local` — Локальное Streamlit демо

Запускает веб-интерфейс для NER с использованием локальной ONNX модели (без Triton).

```bash
python -m ner.commands demo-local
```

Приложение будет доступно по адресу: `http://localhost:8501`

### `demo-triton` — Streamlit демо с Triton

Запускает веб-интерфейс для NER с отправкой запросов на Triton Inference Server.

```bash
# Сначала запустите Triton Server, затем:
python -m ner.commands demo-triton
```

Приложение будет доступно по адресу: `http://localhost:8501`

---

| Команда | Описание |
|---------|----------|
| `train` | Обучение модели BERT NER |
| `to-onnx` | Экспорт чекпоинта в ONNX формат |
| `infer` | Тестовый инференс на ONNX модели |
| `prepare-triton` | Подготовка model_repository для Triton |
| `demo-local` | Запуск локального Streamlit демо |
| `demo-triton` | Запуск Streamlit демо с Triton |

## Разработка

См. раздел "Структура проекта" выше.
