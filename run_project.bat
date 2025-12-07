@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo ====================================
echo   NEWS IR SYSTEM — FULL PIPELINE
echo ====================================

echo.
echo Activating virtual environment...
call .\.venv\Scripts\activate

echo.
echo Running preprocessing...
python -m src.preprocessing.load_and_process
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Preprocessing failed.
    exit /b
)

echo.
echo Building TF-IDF index...
python -m src.indexing.build_tfidf
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: TF-IDF build failed.
    exit /b
)

echo.
echo Building BM25 index...
python -m src.indexing.bm25_search build
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: BM25 build failed.
    exit /b
)

echo.
echo Building Sentence Embeddings and FAISS index...
python -m src.semantic.build_embeddings_faiss
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: FAISS build failed.
    exit /b
)

echo.
echo Training LDA topic model...
python -m src.topic_modeling.train_lda
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: LDA training failed.
    exit /b
)

echo.
echo Generating LDA visualization...
python -m src.topic_modeling.visualize_topics
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: LDA visualization failed.
    exit /b
)

echo.
echo ====================================
echo     ALL STEPS COMPLETED SUCCESSFULLY
echo ====================================
echo.
echo Starting Streamlit app...
streamlit run demo/streamlit_app.py
