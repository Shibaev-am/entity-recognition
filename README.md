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

Если у вас еще нет обученной модели и ONNX файла, выполните следующие шаги:

### 2.1. Обучение модели
```bash
# Запуск обучения (параметры конфигурируются в configs/config.yaml)
python scripts/train.py
```
Чекпоинт модели сохранится в папку `models/`.

### 2.2. Экспорт в ONNX
Скрипт автоматически найдет последний чекпоинт и сконвертирует его.
```bash
python scripts/to_onnx.py
```
Это создаст файлы `models/model.onnx` и `models/model.onnx.data`.

## 3. Подготовка Triton Model Repository

Необходимо скопировать модель в структуру папок, понятную Triton Server.

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
├── apps/               # Streamlit приложения (UI)
│   ├── demo_local.py   # Локальный инференс (HuggingFace Pipeline)
│   └── demo_triton.py  # Клиент для Triton Inference Server
├── configs/            # Конфигурации Hydra (.yaml)
├── data/               # Данные для обучения
├── model_repository/   # Репозиторий моделей для Triton
│   └── bert_ner/       # Конфигурация и веса модели
│       ├── 1/          # Версия модели
│       │   └── model.onnx
│       └── config.pbtxt
├── models/             # Локальные артефакты обучения (.ckpt, .onnx)
├── outputs/            # Логи обучения (Hydra)
├── scripts/            # Скрипты (train, export)
└── src/                # Исходный код (model, dataset, utils)
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

В отдельном терминале запустите Streamlit приложение, которое будет отправлять запросы к Triton.

```bash
poetry run streamlit run apps/demo_triton.py
```

Приложение будет доступно по адресу: `http://localhost:8501`.

## Разработка

См. раздел "Структура проекта" выше.
