# AI Assignment OCR and Grader

A two-part Python project using the Google Gemini API to:
1.  Extract text from scanned handwritten assignments (PDF format).
2.  Automatically score the extracted text (out of 10) and provide personalized feedback.

## Features

* Processes multi-page PDFs containing scanned images.
* Uses Gemini Vision for optical character recognition (OCR), including handwriting.
* Uses Gemini's text generation model for automated scoring and feedback.
* Outputs structured results (scores and feedback) to a CSV file.

## Workflow

1.  Place scanned assignment `.pdf` files into the `Input` folder.
2.  Run the **OCR script** (e.g., `pdf_to_text.py`) to convert PDFs to `.txt` files in the `Output` folder.
3.  Run the **Scoring script** (`score_assignments.py`) to read `.txt` files, evaluate them using Gemini, and save results to `assignment_scores.csv`.

## Setup

1.  **API Key:** Set your Google Gemini API key as an environment variable:
    ```bash
    # Linux/macOS
    export GEMINI_API_KEY='YOUR_API_KEY'
    # Windows (Command Prompt)
    set GEMINI_API_KEY=YOUR_API_KEY
    # Windows (PowerShell)
    $env:GEMINI_API_KEY='YOUR_API_KEY'
    ```

2.  **Dependencies:** Install the required Python libraries using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Folders:** Ensure you have two empty folders named `Input` and `Output` in the same directory as the scripts.

## Usage

1.  **Place PDFs:** Copy your assignment PDF files into the `Input` folder.
2.  **Run OCR Script:** Execute the first script (replace `your_ocr_script_name.py` with its actual name):
    ```bash
    python pdf.py
    ```
    *(Wait for it to complete. It will create `.txt` files in the `Output` folder.)*
3.  **Customize Scoring (Important!):** Open `score_assignments.py` and modify the `get_evaluation_prompt` function. Add specific details about your assignment, grading criteria, or subject matter to get relevant scores and feedback.
4.  **Run Scoring Script:** Execute the second script:
    ```bash
    python score.py
    ```
5.  **Check Results:** Look for the scores and feedback printed in the console and check the `assignment_scores.csv` file for a summary.

---

*Note: The accuracy of OCR and the quality of scoring/feedback depend heavily on the input image quality and the detail provided in the scoring prompt.*
