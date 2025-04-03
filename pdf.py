import os
import sys
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from PIL import Image # Still useful for getting MIME type from bytes if needed
import io             # To handle image bytes in memory
import fitz           # PyMuPDF for handling PDFs

# --- Configuration ---
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"
# Use a model that supports image analysis (like gemini-pro-vision or newer)
# Using gemini-1.5-flash as it's generally available and efficient
MODEL_NAME = "gemini-1.5-flash"
# Define the input extension we are looking for
INPUT_EXTENSION = '.pdf'
# Define the intermediate image format (and its MIME type)
# PNG is lossless and well-supported
INTERMEDIATE_IMAGE_FORMAT = "png"
INTERMEDIATE_MIME_TYPE = "image/png"
# --- End Configuration ---

# Note: get_mime_type function is no longer needed as we control the intermediate format

def extract_text_from_image_data(image_data, mime_type, model, page_num, total_pages, pdf_filename):
    """
    Extracts text from image data in memory using the Gemini API.

    Args:
        image_data (bytes): The image data as bytes.
        mime_type (str): The MIME type of the image data (e.g., "image/png").
        model (genai.GenerativeModel): The configured Gemini model instance.
        page_num (int): The current page number (1-based).
        total_pages (int): Total number of pages in the PDF.
        pdf_filename (str): The name of the source PDF file for logging.

    Returns:
        str: The extracted text, or None if an error occurred or no text found.
    """
    print(f"  Processing Page {page_num}/{total_pages} of {pdf_filename}...")

    try:
        # Prepare the content parts for the API request
        image_part = {
            "mime_type": mime_type,
            "data": image_data
        }
        # Updated prompt for potentially better handwriting recognition
        prompt_part = "Extract all handwritten and printed text visible in this image. Preserve the general layout if possible, but focus on accurate transcription. Provide only the extracted text."

        # Make the API call
        contents = [prompt_part, image_part]
        # Increase timeout potential for complex images/handwriting
        request_options = {"timeout": 120} # 120 seconds timeout
        response = model.generate_content(contents, request_options=request_options)


        # --- Handle potential API blocking or errors ---
        if not response.candidates:
            print(f"    Warning: No content generated for page {page_num} of {pdf_filename}, possibly due to safety filters or other issues.")
            try:
                print(f"    Prompt Feedback: {response.prompt_feedback}")
            except (AttributeError, ValueError):
                print("    (No detailed prompt feedback available)")
            return None

        # --- Extract text from the response ---
        try:
            if response.candidates[0].content.parts:
                extracted_text = response.candidates[0].content.parts[0].text
                print(f"    Text extracted successfully from page {page_num}.")
                return extracted_text.strip() # Remove leading/trailing whitespace
            else:
                 # Fallback check for simpler response structures
                 if hasattr(response, 'text') and response.text:
                    print(f"    Text extracted directly from response.text for page {page_num}.")
                    return response.text.strip()
                 else:
                    print(f"    Warning: No text part found in the response structure for page {page_num} of {pdf_filename}.")
                    # print(f"    Full Response Candidate: {response.candidates[0]}") # Uncomment for detailed debugging
                    return None

        except (AttributeError, IndexError, ValueError) as e:
            print(f"    Error parsing response for page {page_num} of {pdf_filename}: {e}")
            # print(f"    Full Response Candidate: {response.candidates[0]}") # Uncomment for detailed debugging
            return None

    except google_exceptions.GoogleAPIError as e:
        # Specific check for potential timeouts or resource exhaustion
        if isinstance(e, google_exceptions.DeadlineExceeded):
             print(f"    API Error (Timeout) processing page {page_num} of {pdf_filename}: {e}")
        elif isinstance(e, google_exceptions.ResourceExhausted):
             print(f"    API Error (Resource Exhausted/Rate Limit) processing page {page_num} of {pdf_filename}: {e}")
        else:
             print(f"    API Error processing page {page_num} of {pdf_filename}: {e}")
        return None
    except Exception as e:
        print(f"    An unexpected error occurred processing page {page_num} of {pdf_filename}: {e}")
        return None


def process_pdf(pdf_path, output_folder, model):
    """
    Processes a single PDF file, extracts text from each page, and saves combined text.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Path to the folder where the output text file will be saved.
        model (genai.GenerativeModel): The configured Gemini model instance.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    pdf_filename = os.path.basename(pdf_path)
    print(f"\nProcessing PDF: {pdf_filename}...")
    all_pages_text = []
    processed_successfully = False # Track overall success for this PDF

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"Found {total_pages} page(s).")

        # Iterate through each page
        for page_num, page in enumerate(doc):
            current_page_index = page_num + 1 # 1-based index for display

            try:
                # Render page to an image (pixmap) - increase resolution for better OCR if needed
                # Default is 96 DPI. Higher DPI uses more memory/time. Try 150 or 200 if needed.
                pix = page.get_pixmap(dpi=150)
                # Convert pixmap to image bytes in the chosen format
                img_data = pix.tobytes(output=INTERMEDIATE_IMAGE_FORMAT)

                # Extract text from the image data
                extracted_text = extract_text_from_image_data(
                    img_data,
                    INTERMEDIATE_MIME_TYPE,
                    model,
                    current_page_index,
                    total_pages,
                    pdf_filename
                )

                if extracted_text is not None:
                    # Add a separator between pages for clarity in the output file
                    page_separator = f"\n\n--- Page {current_page_index} ---\n\n"
                    all_pages_text.append(page_separator + extracted_text)
                else:
                    # Log that a page was skipped but continue processing others
                    print(f"    Skipping text from page {current_page_index} due to extraction issues.")
                    all_pages_text.append(f"\n\n--- Page {current_page_index} (Error extracting text) ---\n\n")


            except Exception as page_e:
                print(f"  Error processing page {current_page_index} of {pdf_filename}: {page_e}")
                all_pages_text.append(f"\n\n--- Page {current_page_index} (Error processing page) ---\n\n")
                # Continue to the next page

        # Close the document
        doc.close()

        # Combine text from all pages
        final_text = "".join(all_pages_text).strip()

        if not final_text:
             print(f"Warning: No text could be extracted from any page of {pdf_filename}.")
             # Decide if an empty file should be created or not. Here, we won't.
             return False # Indicate failure if no text was extracted at all

        # Construct output path
        base_name = os.path.splitext(pdf_filename)[0]
        output_filename = f"{base_name}.txt"
        output_path = os.path.join(output_folder, output_filename)

        # Save the combined text to the output file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"Successfully saved combined text to {output_path}")
            processed_successfully = True
        except IOError as e:
            print(f"Error writing output file {output_path}: {e}")

    except fitz.fitz.FileNotFoundError:
         print(f"Error: PDF file not found at {pdf_path}")
    except fitz.fitz.PasswordError:
         print(f"Error: PDF file {pdf_filename} is password-protected. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing PDF {pdf_filename}: {e}")
        # Ensure doc is closed if it was opened before the error
        try:
            if 'doc' in locals() and doc:
                doc.close()
        except Exception: pass # Ignore errors during cleanup closing

    return processed_successfully


def main():
    """
    Main function to iterate through PDFs, extract text, and save results.
    """
    # --- API Key and Client Setup ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        # Configure safety settings if needed (e.g., to be less strict for assignments)
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]
        # model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Using Gemini model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Google Generative AI client: {e}")
        sys.exit(1)

    # --- Directory Setup ---
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input directory '{INPUT_FOLDER}' not found.")
        print("Please create it and place your PDF files inside.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output directory: {OUTPUT_FOLDER}")
        except OSError as e:
            print(f"Error creating output directory '{OUTPUT_FOLDER}': {e}")
            sys.exit(1)

    # --- Process PDFs ---
    print(f"\nScanning for PDF files in '{INPUT_FOLDER}'...")
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if it's a file and has the target PDF extension
        if os.path.isfile(input_path) and file_ext == INPUT_EXTENSION:
            if process_pdf(input_path, OUTPUT_FOLDER, model):
                processed_count += 1
            else:
                skipped_count += 1
        elif os.path.isfile(input_path):
             # Optional: Log skipped non-PDF files
             # print(f"Skipping non-PDF file: {filename}")
             pass


    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} PDF files.")
    print(f"Skipped/Errored:     {skipped_count} PDF files.")
    print(f"Text files saved in:   '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()