# 專案摘要

本專案在 Streamlit Cloud 部署了一個基於 Hugging Face diffusers 的輕量化 Stable Diffusion Turbo（`stabilityai/sd-turbo`）文生圖應用，主打免 API token、CPU 可運行與快速冷啟。介面以中文導向，提供提示詞、反向提示、步數、引導強度、隨機種子等控制，並收錄兩組範例（閱讀角落、彩虹煙火）與對應的示意圖。安裝相依採固定版本策略（torch 2.5.1、diffusers 0.27.2、transformers 4.37.2、tokenizers 0.15.2、huggingface_hub 0.20.3），搭配 `runtime.txt` 鎖定 Python 3.11.6，避免在 Cloud 端編譯或相依衝突。生成後會計算並展示 Prompt token 的相對重要性（清理特殊字元），同時提供表格與條狀圖視覺化，方便分析提示詞影響。應用亦保留安全檢查器關閉的提醒，方便依需求決定是否加上內容審核。整體目標是提供教學友善、部署容易且回應迅速的文生圖示範服務。
