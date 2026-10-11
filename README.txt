ELECTRICAL FAULT DETECTION - STREAMLIT STARTER

Files:
- app.py: Streamlit user interface and prediction code
- requirements.txt: Python packages
- EXPORT_FROM_COLAB.txt: cell to run in your existing Colab notebook

STEP 1 — Export your models from Colab
1. Open your existing notebook and run the training cells successfully.
2. Add the code in EXPORT_FROM_COLAB.txt in a new cell at the end.
3. Run that cell. It downloads fault_models.joblib to your computer.

STEP 2 — Put the files in one folder
Keep these files together:
- app.py
- requirements.txt
- fault_models.joblib

STEP 3 — Install Python packages
Open Terminal / Command Prompt in that folder and run:
    python -m pip install -r requirements.txt

If your system uses python3 instead:
    python3 -m pip install -r requirements.txt

STEP 4 — Start the website
Run:
    python -m streamlit run app.py

Streamlit will show a local URL, usually:
    http://localhost:8501

Open that URL in your browser. Keep the terminal open while using the app.

IMPORTANT
- The feature order is Ia, Ib, Ic, Va, Vb, Vc, Im, Vm, Pai, Pav, Fi, Fv.
- SVM and KNN use StandardScaler; Decision Tree and Random Forest do not.
- Keep the same Python/scikit-learn environment where possible when saving/loading models.
- Only load joblib/pickle model files that you trust.
- The initial input values are only for interface testing; use real readings with the same units as the training data.
- This demo is not a certified real-time protection system.
