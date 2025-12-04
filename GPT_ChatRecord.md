## Chat Record

- User: 對話紀錄都要存在GPT_ChatRecord.md內
- User: 請參考"https://github.com/yenlung/AI-Demo/blob/master/%E3%80%90Demo03%E3%80%91%E5%92%8C_AI_PK_%E7%9C%8B%E8%AA%B0%E6%AF%94%E8%BC%83%E6%9C%83%E8%AA%8D_IVE_%E6%88%90%E5%93%A1.ipynb"來建立一閣專案
- Assistant: 已下載參考 notebook（demo.ipynb）、建立 DeepFace+Gradio 專案（app.py、requirements.txt、README.md），並產生 photos/ 下的 IVE 成員資料夾。
- User: 請幫我修改成能佈署在streamlit的版本，須確保使用的模型能透過網路直接使用不須藉由token
- Assistant: 新增 Streamlit 版本（streamlit_app.py），更新 requirements.txt 加入 streamlit，並於 README.md 補充 Streamlit 部署方式與模型免 token 說明。
- User: 完整的對話紀錄都要進行儲存，另外請幫我git push到"https://github.com/064718-lgtm/hw4.git"
- Assistant: 已更新 GPT_ChatRecord.md，初始化 Git 專案，提交並推送 main 分支到 https://github.com/064718-lgtm/hw4.git。
- User: ImportError: This app has encountered an error... (DeepFace import leads to cv2 ImportError on Streamlit Cloud)
- Assistant: 更新 requirements.txt 改用 opencv-python-headless 並在 README.md 註明，以解決 Streamlit Cloud 環境下的 cv2 匯入問題。
- User: 那個報錯是在使用streamlit連接git去佈署到網路上時出現的
- Assistant: 已確認錯誤來源於 Streamlit Cloud 部署，改用 opencv-python-headless 並建議重新部署以套用新依賴。
- User: 部署後出現 ValueError（RetinaFace 對 Keras 3 不相容）
- Assistant: 將 requirements.txt 鎖定 tensorflow==2.15.0、keras==2.15.0 以避開 Keras 3，並在 README.md 註記原因。
