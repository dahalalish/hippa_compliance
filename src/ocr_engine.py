import os
import platform
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

# ----------------------------
# Automatically set Tesseract path for Windows
# ----------------------------
if platform.system() == "Windows":
    # Default install path for Tesseract on Windows
    default_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path
    else:
        raise FileNotFoundError(
            f"Tesseract not found at {default_tesseract_path}. "
            "Please install Tesseract OCR or update the path."
        )

# ----------------------------
# Extract text from image
# ----------------------------
def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return pytesseract.image_to_string(Image.open(image_path))

# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdf2image + Tesseract OCR."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text
