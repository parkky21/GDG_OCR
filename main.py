import os
import sys
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from PIL import Image # Used for verifying images and getting MIME types
import mimetypes # Alternative/fallback for MIME types

# --- Configuration ---
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"
# Use gemini-pro-vision for image analysis
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Adjust as needed
# Define supported image extensions (adjust if needed)
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
# --- End Configuration ---

def get_mime_type(file_path):
    """Gets the MIME type of an image file."""
    try:
        # Try Pillow first for accurate detection
        with Image.open(file_path) as img:
            return Image.MIME.get(img.format)
    except Exception:
        # Fallback to mimetypes module based on extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("image/"):
            return mime_type
        else:
            # If detection fails, make a reasonable guess or return None
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".jpg" or ext == ".jpeg": return "image/jpeg"
            if ext == ".png": return "image/png"
            if ext == ".webp": return "image/webp"
            print(f"  Warning: Could not determine MIME type for {os.path.basename(file_path)}. Skipping.")
            return None

def extract_text_from_image(image_path, model):
    """
    Extracts text from a single image using the Gemini API.

    Args:
        image_path (str): The path to the image file.
        model (genai.GenerativeModel): The configured Gemini model instance.

    Returns:
        str: The extracted text, or None if an error occurred or no text found.
    """
    print(f"Processing {os.path.basename(image_path)}...")

    mime_type = get_mime_type(image_path)
    if not mime_type:
        return None # Skip if we can't determine MIME type

    try:
        # Read image data
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Prepare the content parts for the API request
        image_part = {
            "mime_type": mime_type,
            "data": image_data
        }
        prompt_part = "Extract all text visible in this image. Provide only the extracted text."

        # Make the API call (non-streaming for simpler text extraction)
        # Contents should be a list containing the prompt and the image
        contents = [prompt_part, image_part]
        response = model.generate_content(contents)

        # --- Handle potential API blocking or errors ---
        if not response.candidates:
            print(f"  Warning: No content generated for {os.path.basename(image_path)}, possibly due to safety filters or other issues.")
            # Log detailed feedback if available
            try:
                 print(f"  Prompt Feedback: {response.prompt_feedback}")
            except (AttributeError, ValueError):
                 print("  (No detailed prompt feedback available)")
            return None

        # --- Extract text from the response ---
        # Access the text part correctly for vision models
        try:
            # Check if the response has parts, typical for Gemini Pro Vision
            if response.candidates[0].content.parts:
                extracted_text = response.candidates[0].content.parts[0].text
                print(f"  Text extracted successfully.")
                return extracted_text.strip() # Remove leading/trailing whitespace
            else:
                 # Sometimes the text might be directly in response.text if the model simplifies
                 # Or if there was an issue and only a text error message came back
                 if hasattr(response, 'text'):
                      print(f"  Extracted text directly from response.text (may indicate simpler response or issue).")
                      return response.text.strip()
                 else:
                    print(f"  Warning: No text part found in the response structure for {os.path.basename(image_path)}.")
                    print(f"  Full Response Candidate: {response.candidates[0]}")
                    return None

        except (AttributeError, IndexError, ValueError) as e:
            print(f"  Error parsing response for {os.path.basename(image_path)}: {e}")
            print(f"  Full Response Candidate: {response.candidates[0]}") # Helps debugging
            return None

    except FileNotFoundError:
        print(f"  Error: Image file not found at {image_path}")
        return None
    except google_exceptions.GoogleAPIError as e:
        print(f"  API Error processing {os.path.basename(image_path)}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred processing {os.path.basename(image_path)}: {e}")
        return None


def main():
    """
    Main function to iterate through images, extract text, and save results.
    """
    # --- API Key and Client Setup ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        sys.exit(1) # Exit if key is missing

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Using Gemini model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Google Generative AI client: {e}")
        sys.exit(1)

    # --- Directory Setup ---
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input directory '{INPUT_FOLDER}' not found.")
        print("Please create it and place your images inside.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created output directory: {OUTPUT_FOLDER}")
        except OSError as e:
            print(f"Error creating output directory '{OUTPUT_FOLDER}': {e}")
            sys.exit(1)

    # --- Process Images ---
    print(f"\nScanning for images in '{INPUT_FOLDER}'...")
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if it's a file and has a supported image extension
        if os.path.isfile(input_path) and file_ext in SUPPORTED_EXTENSIONS:

            # Construct output path (replace image ext with .txt)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.txt"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # Extract text using the helper function
            extracted_text = extract_text_from_image(input_path, model)

            if extracted_text is not None:
                # Save the text to the output file
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)
                    print(f"  Successfully saved text to {output_path}")
                    processed_count += 1
                except IOError as e:
                    print(f"  Error writing output file {output_path}: {e}")
                    skipped_count += 1
            else:
                 # Error occurred or no text found during extraction
                 print(f"  Skipping save for {filename} due to extraction issues.")
                 skipped_count += 1
        elif os.path.isfile(input_path):
             print(f"Skipping non-supported file: {filename}")


    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} images.")
    print(f"Skipped/Errored:      {skipped_count} files.")
    print(f"Text files saved in:  '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    main()