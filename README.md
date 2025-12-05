# Streamlit Diffusers 文生圖（sd-turbo）

使用 Hugging Face `stabilityai/sd-turbo`（輕量版 Stable Diffusion Turbo）與 diffusers 套件的 Streamlit 應用。無需 API Token，於 Streamlit Cloud 以 CPU 執行即可產生圖片。

## 線上展示
- Streamlit: https://43onov79zc27hey78w7tzk.streamlit.app/

## 本地快速開始
1. `python -m venv .venv && .\.venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## 部署到 Streamlit Cloud
1. 將此資料夾推送到 GitHub。
2. 在 [Streamlit Cloud](https://streamlit.io/cloud) 建立新專案並選擇該 repo。
3. 將 **Main file path** 設為 `app.py`（預設指令 `streamlit run app.py` 即可）。
4. 首次啟動會下載模型權重；之後因為快取會更快。

> 附帶 `runtime.txt`，將 Python 版本鎖定為 3.11（目前 `python-3.11.6`），避免在 Streamlit Cloud 編譯 Rust 相依。
- `transformers==4.37.2` 與 `tokenizers==0.15.2`（有預編譯 wheel），`diffusers==0.27.2` 相容 sd-turbo。
- `huggingface_hub==0.20.3` 鎖版本以保留 diffusers 0.27.x 仍需的 `cached_download`。

## 模型與參數
- 模型：`stabilityai/sd-turbo`，透過 `AutoPipelineForText2Image`。
- 建議採樣步數 1–2，guidance scale 接近 0 時效果最佳。
- 輸出解析度固定 512x512，以符合 Streamlit Cloud CPU/RAM 限制。
- Torch 鎖為 `2.5.1`（目前在 Streamlit Cloud 可用的最新版本），避免安裝失敗。

## 介面說明
- **主控制區**：提示詞輸入、採樣步數、guidance scale、隨機種子切換與生成按鈕，可下載 PNG。
- **Token 計數**：顯示 text encoder 目前 token 數（預設上限 20），提醒提示詞長度。
- **範例提示**：展示 `example.png`（城市街角）與 `example2.png`（彩虹貓），並給出建議的提示詞與步數/引導值。
- **歷史紀錄**：保留最近 6 張生成結果，含提示詞與設定，可一鍵重播。
- **快速提示**：提供幾組常用預設，方便初次嘗試。

## 檔案說明
- `app.py`：Streamlit UI 與圖片生成邏輯。
- `requirements.txt`：Cloud 與本地的 Python 依賴。
- `runtime.txt`：指定 Streamlit Cloud 的 Python 版本。
- `example.png`、`example2.png`：示例輸出，展示於範例區塊。
- `GPT_Chat.md`：依需求保留的對話紀錄。

## 參考資料
- https://github.com/yenlung/AI-Demo/blob/master/%E3%80%90Demo08%E3%80%91%E7%94%A8diffusers%E5%A5%97%E4%BB%B6%E7%94%9F%E6%88%90%E5%9C%96%E5%83%8F.ipynb
