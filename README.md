# Goal-relevant scene graphs for robot manipulation

Репозиторий содержит код эксперимента для сравнения трёх методов построения
графов роботических сцен:

- **OUR** — goal-guided выделение и трекинг объектов манипуляции;
- **SG-Ego** — адаптированный baseline с ограниченной генерацией триплетов;
- **SVG2** — адаптированный baseline с нативным трекингом и последующим
  выбором task-relevant треков.

Во всех методах итоговая оценка ограничена четырьмя ролями:

```text
robot
manipulated_object
initial_support
target
```

Фоновые объекты и любые связи с ними не участвуют в precision/recall.

## Как устроен эксперимент

```text
RGB frames + planning goal
            │
            ├── OUR ──────┐
            ├── SG-Ego ───┼── нормализация к четырём ролям
            └── SVG2 ─────┘                │
                                           ├── OUR | SG-Ego | SVG2 visualization
                                           └── human + VLM evaluation
```

### Адаптация SG-Ego

Planning goal передаётся в captioning prompt. Генерация триплетов ограничена
роботом, объектом манипуляции, начальной опорой и целью. Grounding и temporal
consolidation работают только с этими триплетами.

### Адаптация SVG2

SVG2 сначала выполняет обычный class-agnostic tracking. После описания треков
goal-conditioned selector назначает максимум один существующий трек каждой из
четырёх ролей. Связи строятся только между выбранными треками.

## Структура репозитория

```text
benchmark/                               запуск, оценка и визуализация
baselines/sg-ego/                        адаптированный SG-Ego
baselines/svg2/                          адаптированный SVG2
sharerobot_sg/unified_scene_graph_pipeline/  OUR segmentation pipeline
```

Сгенерированные результаты, видео, веса и API-ключи в Git не включены.

## Требования

- Linux;
- Docker Engine и Docker Compose v2 либо `docker-compose` v1;
- NVIDIA GPU и NVIDIA Container Toolkit;
- драйвер с поддержкой CUDA 12.8;
- доступ к Hugging Face для скачивания моделей;
- OpenAI-compatible multimodal API для стадий 5–6 SVG2.

Проверка GPU в Docker:

```bash
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

Подробности: [benchmark/DOCKER.md](benchmark/DOCKER.md).

## Формат входных данных

Runner получает не произвольную папку с видео, а JSON-манифест. Для каждой
записи каталог `relative_path` должен содержать численно упорядоченные кадры:

```text
SOURCE_ROOT/
└── dataset_name/
    └── episode_001/
        └── images/
            ├── frame_000000.png
            ├── frame_000001.png
            └── ...
```

Минимальная запись манифеста:

```json
{
  "episodes": [
    {
      "relative_path": "dataset_name/episode_001",
      "dataset_name": "dataset_name",
      "planning_goal": "pick up the red cup and place it on the tray",
      "frame_count": 30
    }
  ]
}
```

Готовый шаблон: [benchmark/manifest.example.json](benchmark/manifest.example.json).

## Быстрый запуск SG-Ego и SVG2

### 1. Настроить ключи и кэши

```bash
cp benchmark/docker.env.example benchmark/docker.env
chmod 600 benchmark/docker.env
```

Заполнить `benchmark/docker.env` локальными параметрами API и путями к кешам.

`Qwen/Qwen3.8-27B` вызывается по API и внутри Compose не запускается.

### 2. Подключить датасет

Большой датасет не нужно копировать в репозиторий. Например:

```bash
export BENCHMARK_HOST_SOURCE_ROOT=/datasets/sharerobot_planning_manipulation_selected
export BENCHMARK_SOURCE_ROOT=/input/scenes
export BENCHMARK_MANIFEST=/input/scenes/manifest.json
export BENCHMARK_WORK_ROOT=baseline_runs/sharerobot
```

`BENCHMARK_HOST_SOURCE_ROOT` монтируется read-only внутрь контейнера как
`/input/scenes`.

Если манифест хранится в репозитории, можно указать:

```bash
export BENCHMARK_MANIFEST=benchmark/my_manifest.json
```

### 3. Собрать образы

```bash
./benchmark/docker-run.sh build
./benchmark/docker-run.sh smoke
```

### 4. Подготовить MP4

```bash
./benchmark/docker-run.sh prepare
```

MP4 создаются из `images/frame_%06d.png` и сохраняются в
`$BENCHMARK_WORK_ROOT/videos/`.

### 5. Smoke test

Первые несколько записей манифеста:

```bash
./benchmark/docker-run.sh sg-ego --limit 1
./benchmark/docker-run.sh svg2 --limit 1
```

Конкретный эпизод:

```bash
./benchmark/docker-run.sh svg2 --episode dataset_name/episode_001
```

### 6. Полный запуск

```bash
./benchmark/docker-run.sh all
```

Или методы отдельно:

```bash
./benchmark/docker-run.sh sg-ego
./benchmark/docker-run.sh svg2
```

Для манифеста из 100 или 1000 видео можно добавить защиту от неправильного
размера:

```bash
./benchmark/docker-run.sh all --expected-count 100
./benchmark/docker-run.sh all --expected-count 1000
```

Runner работает stage-wise, держит модель текущей стадии загруженной между
видео и пропускает уже готовые артефакты. После ошибки достаточно повторить ту
же команду.

## Результаты baseline

```text
$BENCHMARK_WORK_ROOT/
├── videos/
├── sg_ego/
│   ├── captions/goal_roles/
│   ├── frame_graphs/goal_roles/
│   └── video_graphs/goal_roles/
└── svg2/
    └── <dataset__episode>/
        ├── stage1_masks.json
        ├── stage2_tracks.json
        ├── stage3_tracks_clean.json
        ├── stage4_descriptions.json
        ├── stage5_scene_graph.json
        └── stage6_scene_graph.json
```

SVG2 stages:

1. SAM2 mask generation;
2. SAM2 tracking;
3. cleanup;
4. DAM object captioning;
5. API-based structuring;
6. goal-role selection и API-based spatial/temporal relationships.

При смене API-модели достаточно пересчитать stages 5–6; stages 1–4 от неё не
зависят.

## Трёхпанельная визуализация

После завершения трёх методов:

```bash
./benchmark/docker-run.sh visualize
```

Для одного эпизода:

```bash
./benchmark/docker-run.sh visualize \
  --episode dataset_name/episode_001
```

Результат `OUR | SG-EGO | SVG2` сохраняется в:

```text
$BENCHMARK_WORK_ROOT/comparisons/
```

## Оценка качества

### Обычная ручная reference-разметка

Если готова независимая покадровая разметка `task_role_graph_v1`:

```bash
docker compose --env-file benchmark/docker.env \
  -f benchmark/compose.yaml run --rm svg2 \
  python benchmark/evaluate.py \
  --manifest "$BENCHMARK_MANIFEST" \
  --reference-root reference_graphs \
  --ours-root our_results \
  --sg-ego-root "$BENCHMARK_WORK_ROOT/sg_ego" \
  --svg2-root "$BENCHMARK_WORK_ROOT/svg2" \
  --output "$BENCHMARK_WORK_ROOT/metrics.json"
```

Считаются micro/macro precision, recall и F1 отдельно для role nodes и
направленных `(subject, predicate, object)` triplets.

### Гибридная оценка 1000 видео

Предлагаемый протокол:

- 80 случайных стратифицированных видео проверяет человек;
- 20 high-disagreement видео образуют challenge subset;
- 25 из этих 100 получает второй аннотатор;
- все 1000 видео проверяет слепой VLM judge;
- решения VLM калибруются на human-primary subset;
- для статьи основной результат берётся с human-primary, полный
  VLM-calibrated результат используется как дополнительный.

Подготовить candidate pool и зафиксировать выборку:

```bash
export EVAL_ROOT="$BENCHMARK_WORK_ROOT/evaluation_1000"

./benchmark/docker-run.sh eval-prepare \
  --human-size 100 \
  --challenge-size 20 \
  --double-annotation-size 25
```

Запустить VLM сначала на одном видео, затем на всех:

```bash
./benchmark/docker-run.sh eval-vlm --limit 1 --workers 1
./benchmark/docker-run.sh eval-vlm --workers 2
```

Запустить интерфейс первого аннотатора:

```bash
./benchmark/docker-run.sh eval-annotate annotator1
```

UI будет доступен на `http://localhost:8765`. Для удалённого сервера:

```bash
ssh -L 8765:localhost:8765 USER@SERVER
```

Второй аннотатор размечает только frozen overlap:

```bash
ANNOTATION_PORT=8766 ./benchmark/docker-run.sh \
  eval-annotate annotator2 --only-double
```

Построить итоговый JSON и CSV:

```bash
./benchmark/docker-run.sh eval-report annotator1 \
  --second-annotator annotator2
```

Подробное описание: [benchmark/HYBRID_EVALUATION.md](benchmark/HYBRID_EVALUATION.md).

## Замер времени

Для чистого замера используйте новый `BENCHMARK_WORK_ROOT`, иначе resumable
runner пропустит существующие результаты.

```bash
export BENCHMARK_WORK_ROOT="baseline_runs/timing_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCHMARK_WORK_ROOT/logs"

./benchmark/docker-run.sh prepare --limit 10

set -o pipefail
/usr/bin/time -f $'\nTOTAL_WALL_SECONDS=%e\nCPU_PERCENT=%P\nMAX_RAM_KB=%M' \
  ./benchmark/docker-run.sh all --limit 10 \
  2>&1 | tee "$BENCHMARK_WORK_ROOT/logs/full_pipeline.log"
```

Оценить старый запуск по времени записи артефактов:

```bash
./benchmark/docker-run.sh estimate-time \
  --session-gap-hours 0.5 \
  --json-output "$BENCHMARK_WORK_ROOT/logs/runtime_estimate.json"
```

Это приблизительная оценка: точное wall-clock время даёт только
`/usr/bin/time` во время запуска.

## OUR segmentation pipeline

Самостоятельный запуск OUR-сегментации описан в
[sharerobot_sg/unified_scene_graph_pipeline/README.md](sharerobot_sg/unified_scene_graph_pipeline/README.md).

Короткая последовательность:

```bash
cd sharerobot_sg/unified_scene_graph_pipeline
export SEGMENTATION_ENV=/datasets/goal_guided_segmentation.runtime.env

./run.sh build
./run.sh prepare --output /runs/example \
  --video /datasets/input/example.mp4 \
  --planning-goal "place the red block in the metal bowl"

./run.sh start-qwen
./run.sh wait-qwen
./run.sh entities --output /runs/example
./run.sh stop-qwen

./run.sh sam --output /runs/example
./run.sh visualize --output /runs/example
```

## Тесты

Benchmark unit tests не требуют GPU:

```bash
PYTHONPATH=benchmark python3 -m unittest discover \
  -s benchmark -p 'test_*.py' -v
```

## Частые проблемы

- `unknown or invalid runtime name: nvidia` — не настроен NVIDIA Container
  Toolkit либо Docker daemon не был перезапущен.
- `Permission denied` в outputs/logs — старые файлы созданы root-контейнером;
  runner теперь запускается с UID/GID текущего пользователя. Для кэша можно
  указать новый user-owned `HF_CACHE_DIR` или `TORCH_CACHE_DIR`.
- Модель снова скачивается — убедитесь, что host cache смонтирован и внутри
  контейнера `HF_HOME=/models/huggingface`, `TORCH_HOME=/models/torch`.
- API timeout — уменьшите `SVG2_API_CONCURRENCY` до `1` или `2` и повторите
  запуск: готовые стадии будут пропущены.
- Невалидный JSON от Qwen — pipeline выполняет text-only repair и повторяет
  запрос до трёх раз.

## Безопасность и лицензии

Не коммитьте `benchmark/docker.env`, runtime env-файлы, модельные веса или
датасеты. В репозитории присутствует код сторонних проектов; их лицензии и
атрибуция сохранены в соответствующих каталогах. Перед публичным релизом своей
части проекта выберите лицензию и проверьте условия распространения моделей и
датасета.
