# Label Power

電力設備 YOLO bounding-box 桌面標註平台。支援逐類別標註、逐圖片總檢查、明確審閱狀態、資料集續接、小物件放大，以及需要人工接受的模型／光流候選框。

**儲存不等於審閱完成；空標籤不等於已確認無物件。** 圖片、標註、影片與模型權重不包含在此儲存庫。

## 安裝與啟動

需求：Python 3.11+、Tk 桌面環境、[uv](https://docs.astral.sh/uv/getting-started/installation/)。專案以 `.python-version` 固定 Python 3.11，並提交 `uv.lock`。

```bash
git clone https://github.com/1122-gggggg/label_power.git
cd label_power
uv sync --locked
uv run python scripts/annotation_tool.py
```

未指定資料集時，視窗會依序讓你選擇圖片目錄、標籤目錄與類別檔。圖片目錄需直接包含圖片；類別檔每行一個類別，行號從 0 起算。預設 `classes.txt` 為原有 15 類電力設備，**不要任意調換類別順序**。

也可直接指定路徑：

```bash
uv run python scripts/annotation_tool.py \
  --images-dir /path/to/images/all \
  --labels-dir /path/to/labels/all \
  --classes-file classes.txt
```

Linux 需可用的 X11／桌面顯示。若系統 Python 缺少 Tk，可使用 uv 提供的 Python；若要使用系統 Python，先安裝該系統對應的 Tk 套件，再以 `uv sync --python /path/to/python --locked` 建立環境。中文介面優先使用 Noto Sans CJK TC／Noto Sans TC；舊式 X11 字型環境使用可用的中文字型。

## 審閱規則

每個「圖片 × 類別」獨立記錄：

| 狀態 | 意義 |
|---|---|
| `pending` | 尚未明確審閱 |
| `edited` | 已修改，或原有確認已失效，需重新審閱 |
| `positive` | 人工明確確認，且該類別有已儲存框 |
| `negative` | 人工明確確認，且該類別無框 |

- 畫框、移動、縮放、刪除、復原、重做與接受候選框會自動儲存，但**不會自動確認**。
- 編輯只使受影響類別的確認失效，其他類別維持原有確認。
- 「確認本類 + 下一張」確認目前類別；逐圖片模式則確認整張圖片的全部類別。
- 「確認本類無物件」只作用於目前類別；該類仍有框時會拒絕操作，不會偷偷刪框。
- 「略過／下一張」、切換類別、單純瀏覽皆不增加完成率，也不自動建立空標籤。
- 確認範圍內仍有未處理候選框時，必須先接受或拒絕。
- YOLO 檔不存候選框、信心或審閱狀態；審閱狀態另存 JSON。

### 操作與快捷鍵

| 操作 | 控制 |
|---|---|
| 畫框 | 空白處左鍵拖曳 |
| 移動／調整框 | 拖曳框／控制點 |
| 平移／縮放 | 右鍵拖曳／滾輪 |
| 符合視窗 | `F` |
| 選取框置中放大 | 框選取後雙擊，或「已接受框」頁籤按鈕 |
| 上一張／略過下一張 | `←`／`→`；`P` 上一張 |
| 明確確認並下一張 | `N` |
| 確認本類無物件 | `M` |
| 下一／上一類別 pass | `J`／`K`，不會確認上一類 |
| 切類別 | `[`／`]`、`0`–`9`；全部類別皆可從列表選取 |
| 儲存／復原／重做 | `S`／`Ctrl+Z`／`Ctrl+Y` |
| 刪除選取框 | `Delete`／`Backspace` |

文字輸入欄位不會觸發標註快捷鍵。右側主要操作固定可見；類別、框、候選與輔助工具使用可縮放頁籤，長列表有捲軸。

「輔助工具」可開關局部放大鏡與十字準星。放大鏡使用原圖像素；不會用放大後的螢幕座標寫標籤。類別列表提供中文提示；**英文名稱與 classes.txt 的順序才是訓練類別依據**。「已審閱範例」只從已明確確認的真實標註裁切，不拿模型猜測或未審框充當範例。

## 資料集管理、續接與篩選

「資料集」選單可開啟／儲存 JSON 設定、切換圖片／標籤目錄、重新載入、查看遷移通知及匯出。

設定檔範例（相對路徑以設定檔所在目錄為基準，不是 shell 工作目錄）：

```json
{
  "version": 1,
  "images_dir": "images/all",
  "labels_dir": "labels/all",
  "classes_file": "classes.txt"
}
```

```bash
uv run python scripts/annotation_tool.py --dataset /path/to/dataset.json
```

預設檔案位置：

```text
LABELS_DIR/
  image_name.txt
  .annotation/
    review.json
    review.session.json
    .history/
  .history/
```

可用 `--pass-state /path/to/review.json` 指定審閱檔；工作階段檔放在其旁。啟動預設恢復圖片、類別、模式、篩選、視窗與視角；`--no-resume` 停用，`--start`／`--start-class` 可覆寫起點（索引从 0 開始）。

篩選包括 `All`、`Unreviewed`、`Edited`、`Confirmed empty`、`Confirmed`。逐類別模式依目前類別篩選；逐圖片模式檢查整張圖片的全部類別。跳轉欄位接受從 1 開始的圖片編號或完整檔名。

審閱檔綁定圖片／標籤目錄、類別順序與圖片檔案資訊。新增、刪除或變更圖片會提示；換目錄或類別順序不會沿用錯誤完成率。重新開啟／載入審閱狀態時，也會以每類框的內容指紋檢查外部標籤修改。

### 舊標註工具遷移

原有 YOLO 標籤可直接讀取。原工具的 v1 `annotation_pass_state.json` 可能把「已有框」誤當成人工審閱，因此不能把它當成可靠的完成證據。

```bash
uv run python scripts/annotation_tool.py \
  --images-dir /path/to/images/all \
  --labels-dir /path/to/labels/all \
  --pass-state /path/to/reports/annotation_pass_state.json
```

開啟時保留舊審閱檔的版本備份；仍存在圖片的舊 reviewed 轉成 `edited`，不存在圖片的記錄退出目前進度並顯示通知。**標註框不會因遷移而刪除，但需要重新明確確認。** 不指定舊審閱檔則從新進度開始。

## 安全存檔與錯誤恢復

- 標籤嚴格檢查類別、有限數值、正面積與完整框邊界，拒絕 `NaN`／`Inf`、未知類別及越界；全圖框可使用 `[0, width] × [0, height]` 的連續座標邊界。
- 格式錯誤顯示檔名與行號，該圖片禁止編輯／覆寫，不會無聲丟棄錯誤行。
- 寫入使用同目錄唯一暫存檔、fsync、原子替換與每個檔案的版本備份。
- `.history/<filename>/` 保留最近 10 份先前版本；同內容不重寫。穩定的 `.<filename>.lock` 檔是寫入鎖，請勿在程式使用中刪除。
- 多視窗或外部編輯造成內容衝突時，拒絕覆蓋。寫入失敗會保留記憶體中的未儲存修改並阻止直接切圖／關閉。

若外部編輯已修復標籤，或要放棄衝突的未儲存修改，使用「資料集 → 重新載入目前圖片」。有未儲存修改或候選框時會先要求確認放棄，再讀取磁碟上的標籤及最新審閱狀態。

如需還原備份，先關閉使用該資料集的所有視窗，保留目前檔案，再將所需 `.bak` 複製回原路徑，重新開啟並檢查。審閱指紋會要求不一致的類別重新確認。

## 模型預標與影格傳播（選用）

```bash
uv sync --locked --extra vision
uv run python scripts/annotation_tool.py \
  --dataset /path/to/dataset.json \
  --model /path/to/trusted_weights.pt \
  --confidence 0.25
```

也可從「輔助工具」選取權重。預設鎖定 CPU 版 PyTorch，避免為基本標註工具安裝 CUDA；基本畫框、審閱、續接、驗證不需要 vision extra。

**只載入自己訓練或可信來源的 `.pt`；PyTorch 權重可能包含可執行內容。** 不自動下載模型，不將權重上傳 GitHub。

模型類別依名稱精確匹配（忽略大小寫、底線／空白差異），**不假設模型 class id 等於資料集 class id**。需要時可指定映射 JSON：

```json
{
  "high voltage insulator": "high voltage insulator",
  "transformer box": 1,
  "utility pole": 3
}
```

```bash
uv run python scripts/annotation_tool.py --dataset /path/to/dataset.json \
  --model /path/to/trusted_weights.pt --model-mapping /path/to/mapping.json
```

明確映射存在時，每個實際輸出的模型類別都必須有有效映射；未知／歧義名稱會拒絕整批結果，不會猜 id。映射只是類別對應，不能讓不相容的模型憑空學會新物件。

影格傳播從上一張已接受框提取特徵，執行實際 LK 光流、前後向一致性及 RANSAC 仿射估計。只接受同一影片序列的 `prefix_frame_timestampms` 檔名；缺乏紋理、對應不足或不穩定的大幅運動會拒絕，不以原框複製充當追蹤結果。

兩種功能皆在背景運算。黃色虛線框只進入「待確認候選」頁籤；選取後可接受或拒絕。離開圖片會要求確認是否丟棄剩餘候選；圖片已切換時，不會把舊背景結果加到新圖片。接受後可正常調框，仍須另行確認審閱。**候選信心不是人工標註正確率。**

## 驗證與完整審閱匯出

嚴格格式檢查，不改寫資料：

```bash
uv run python scripts/verify_yolo_labels.py \
  --images-dir /path/to/images/all \
  --labels-dir /path/to/labels/all \
  --classes-file classes.txt
```

訓練前額外要求每個圖片／類別都明確確認：

```bash
uv run python scripts/verify_yolo_labels.py \
  --images-dir /path/to/images/all \
  --labels-dir /path/to/labels/all \
  --classes-file classes.txt --require-reviewed
```

指定過其他審閱檔時，加上 `--pass-state /path/to/review.json`。主程式也提供 `--selftest --require-reviewed`，搭配資料集路徑或 `--dataset`，不開 GUI。

驗證器區分缺少標籤、合法空標籤、非法標籤及未審閱狀態。退出碼：`0` 通過、`1` 資料／完成度不符、`2` 設定或審閱檔無法讀取。讀取舊進度也不會在驗證過程中遷移或建立檔案。

「資料集 → 匯出已完整審閱資料」只允許**全部圖片 × 全部類別**已明確確認且與實際框一致的資料集。選擇不存在的新目錄；不覆蓋既有目錄。驗證與複製完成後才發布輸出，包含 `images/all`、`labels/all`、`classes.txt`、`data.yaml` 及審閱摘要。

匯出**不自動將同一批影格同時充當 train/val**；`data.yaml` 是完整標註集入口，訓練前仍需建立獨立資料分割，避免相鄰影格造成驗證洩漏。

## 效能與回歸驗證

- 圖片由背景執行緒解碼，預載鄰近影格；RGB 快取採有容量上限的 LRU。
- `--cache-mb 128` 控制快取容量，不是整個程式的 RSS 上限；目前顯示圖片、解碼中結果與模型另占記憶體。
- 只重取可視影像區域；互動期間快速重繪，停止後恢復高品質。
- 重繪請求合併，保留 Canvas 背景影像項目；放大鏡不重建所有框，視窗外的框不建立繪圖項目。
- 「說明 → 效能量測」顯示圖片請求到首次重繪、實際重繪耗時及快取資訊。

本地隔離測試使用相同的 6 張 **3840×2160 壓力測試圖（由真實 720p 影格上採樣，不代表原始 4K 拍攝）**及來回切圖。一次量測中，首次繪製中位數由原工具約 69.7 ms 降至約 39.0 ms；數值依硬體、字型、磁碟、快取與框數改變，不是效能保證。

```bash
uv sync --locked --extra vision
uv run --no-sync pytest -q
uv run --no-sync ruff check scripts tests
uv run --no-sync ruff format --check scripts tests
```

回歸測試包含 YOLO 數值／邊界性質測試、原子替換失敗、寫入衝突、審閱轉換／遷移、資料集隔離、外部標籤變更、快取與回呼執行緒、實際光流平移／不同解析度，以及完整審閱匯出。GitHub Actions 使用鎖定依賴執行相同檢查；模型推論另以本地可信真實權重操作驗證，不在 CI 下載權重。

## 1.0 更新

以明確審閱取代「已有框即完成」；加入嚴格解析、原子備份與衝突保護、固定可見操作區、中文類別提示／真實範例、放大鏡、重做、逐圖片總檢查、資料集續接／篩選／跳轉、背景圖片載入、需人工接受的 YOLO／光流候選，以及完整審閱驗證／匯出。
