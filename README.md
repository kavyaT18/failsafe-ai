# FailSafe – Setup Guide

## Project Structure
```
failsafe/
├── backend/
│   ├── main.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── .env
│   └── failsafe_model3.pkl   ← copy your saved model here
└── frontend/
    ├── package.json
    ├── public/index.html
    └── src/
        ├── App.js / App.css
        ├── index.js / index.css
        └── components/
            ├── SinglePredict.js
            └── BulkPredict.js
```

## Step 1 – Export model from Colab
Add this cell at the end of your notebook and download the file:
```python
from google.colab import files
files.download('failsafe_model3.pkl')
```
Then copy `failsafe_model3.pkl` into `backend/`.

## Step 2 – Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Edit `.env`:
- `MODEL_PATH=failsafe_model3.pkl`
- `ANTHROPIC_API_KEY=sk-ant-...`  (optional, enables LLM intervention plans)

API docs: http://localhost:8000/docs

## Step 3 – Frontend
```bash
cd frontend
npm install
npm start
```
Opens at http://localhost:3000

## Notes
- The `feature_eng` in utils.py exactly mirrors your Colab code.
- SHAP runs on the XGBoost step only (extracts `model.named_steps['xgb']`).
  If your pipeline step name differs, update the key in `utils.py → get_shap_reasons`.
- Threshold is 0.5 by default; change `model.predict` threshold by adjusting 
  `predict_proba` cutoff in `main.py` if needed.
