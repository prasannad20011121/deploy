# AI Medical Report Generator

This app uses Streamlit to provide AI-powered medical image diagnosis and report generation.

## How to Deploy on Streamlit Cloud

1. **Fork or clone this repo to your GitHub account.**
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud) and click 'New app'.**
3. **Select your repo and set the main file as `streamlit_app.py`.**
4. **Ensure all model files are present in the `models/` folder.**
5. **Streamlit Cloud will automatically install dependencies from `requirements.txt`.**

## File Structure
```
streamlit_app.py
requirements.txt
models/
    bone_model.keras
    brain_model.keras
    breast_model.keras
    kidney_model.keras
    main_model.keras
```

## Requirements
See `requirements.txt` for required Python packages.

## Notes
- If you use Google Gemini, make sure your API key is set as a Streamlit Cloud secret.
- All models must be included in the repo for deployment.
