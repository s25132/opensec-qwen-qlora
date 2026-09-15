# OpenSec Triage — Qwen3-0.6B + QLoRA

Kompletny projekt edukacyjny do lokalnego dostrojenia modelu `Qwen/Qwen3-0.6B`
na zbiorze `tegridydev/opensec-triage`. Model klasyfikuje alert bezpieczeństwa
do jednej z siedmiu klas:

- `malicious`
- `suspicious`
- `benign`
- `expected_admin`
- `authorised_testing`
- `misconfiguration`
- `unknown`

Trening używa konfiguracji `edge_trainer_chat`, przygotowanej przez autorów
zbioru dla konwersacyjnego SFT. Ewaluacja korzysta z odpowiadającej jej
konfiguracji `edge_trainer` i nigdy nie przekazuje modelowi prawidłowej
odpowiedzi z testu.

## Wymagania

- Python 3.11 lub 3.12
- NVIDIA GPU z CUDA; ustawienia domyślne są konserwatywne dla RTX 3050 4 GB
- działająca instalacja PyTorch z CUDA
- około 5–10 GB wolnego miejsca na model, cache i checkpointy
- Windows z WSL2/Ubuntu jest zalecany, gdy `bitsandbytes` sprawia problemy

## 1. Środowisko

Nie przeinstalowuj PyTorch, jeśli masz już wersję zgodną z CUDA.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/WSL
# .venv\Scripts\Activate.ps1      # PowerShell

pip install -r requirements.txt
python scripts_check_gpu.py
```

Jeżeli `CUDA dostępna` wynosi `False`, nie rozpoczynaj treningu QLoRA.

## 2. Sprawdzenie danych

```bash
python -m src.inspect_data
```

Program pokaże liczność splitów, etykiet oraz przykładową rozmowę. Dane zostaną
pobrane automatycznie z Hugging Face przy pierwszym użyciu.

## 3. Baseline bez LLM

```bash
python -m src.baseline
```

Baseline TF-IDF + regresja logistyczna jest ważny: zbiór jest syntetyczny i
silnie szablonowy. Jeżeli prosty model osiąga wynik podobny do Qwena, nie należy
interpretować wyniku LLM jako dowodu rozumowania cyberbezpieczeństwa.

Wyniki trafiają do `outputs/baseline/`.

## 4. Smoke test treningu

Najpierw uruchom krótki test całego pipeline'u:

```bash
python -m src.train --smoke-test
```

Smoke test używa ograniczonej liczby rekordów, jednej epoki i zapisuje adapter w:

```text
outputs/qwen3-opensec-smoke/final_adapter
```

Nie oceniaj jakości modelu na podstawie smoke testu. Jego celem jest wyłącznie
sprawdzenie środowiska, pamięci GPU i poprawności kodu.

## 5. Pełny trening

```bash
python -m src.train
```

Domyślne parametry:

| Parametr | Wartość |
|---|---:|
| Model | `Qwen/Qwen3-0.6B` |
| Kwantyzacja | NF4, 4-bit |
| LoRA rank | 8 |
| Epoki | 2 |
| Batch GPU | 1 |
| Gradient accumulation | 8 |
| Maksymalna długość | 256 tokenów |
| Learning rate | `2e-4` |

Można je nadpisać:

```bash
python -m src.train --epochs 1 --max-length 192 --lora-r 4
```

Gdy wystąpi `CUDA out of memory`, kolejno:

1. zamknij aplikacje używające GPU;
2. zmniejsz `--max-length` z 256 do 192 lub 128;
3. zmniejsz `--lora-r` z 8 do 4;
4. pozostaw `--batch-size 1`;
5. uruchom proces ponownie po zwolnieniu pamięci GPU.

## 6. Ewaluacja modelu bazowego

Najpierw zmierz Qwena bez adaptera:

```bash
python -m src.evaluate --output-dir outputs/eval-base
```

## 7. Ewaluacja modelu smoke

```bash
python -m src.evaluate --adapter outputs/qwen3-opensec-smoke/final_adapter --output-dir outputs/eval-base-smoke
```

## 8. Ewaluacja modelu dostrojonego

```bash
python -m src.evaluate --adapter outputs/qwen3-opensec/final_adapter --output-dir outputs/eval-finetuned
```

Powstaną:

- `metrics.json` — accuracy, macro F1 i liczba niepoprawnych wyjść;
- `predictions.csv` — tekst, prawidłowa klasa i predykcja;
- `classification_report.txt` — precision/recall/F1 dla każdej klasy;
- `confusion_matrix.png` — macierz pomyłek.

Porównuj model bazowy i dostrojony na dokładnie tym samym `test`.

## 9. Pojedyncza klasyfikacja

```bash
python -m src.predict \
  --adapter outputs/qwen3-opensec/final_adapter \
  --text "A privileged user disabled endpoint monitoring. No approved change was found."
```

Tryb interaktywny:

```bash
python -m src.predict --adapter outputs/qwen3-opensec/final_adapter
```

Zakończ przez pusty wiersz lub `Ctrl+C`.

## 9. Testy projektu

```bash
pytest -q
```

Testy nie pobierają modelu ani danych.

## Struktura

```text
opensec-qwen-qlora/
├── README.md
├── requirements.txt
├── scripts_check_gpu.py
├── src/
│   ├── baseline.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── inspect_data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_helpers.py
└── outputs/
```

## Ograniczenia

- OpenSec Triage jest zbiorem syntetycznym, a nie korpusem realnych incydentów.
- Wysoki wynik może wynikać z rozpoznawania szablonów i słów kluczowych.
- Wynik nie dowodzi gotowości modelu do pracy w SOC.
- Przed zastosowaniem praktycznym potrzebny jest osobny zbiór przypadków
  ocenionych przez analityków bezpieczeństwa.
- Nie łącz `edge_trainer` i `edge_trainer_chat`: są to dwa formaty tych samych
  rekordów, a nie niezależne dane.

Licencja zbioru danych: CC BY 4.0. Licencję i aktualną kartę zbioru należy
sprawdzić przed publikacją modelu lub wyników.
