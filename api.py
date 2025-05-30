from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from omegaconf import OmegaConf
import shutil
import json # For reading json file content
import os
import uuid
import logging
from PIL import Image

from chat import DOLPHIN
# Functions from demo_page.py and utils.py will be called
# We need to ensure they are available in the Python path or copy/adapt them.
# For now, attempting direct import path assuming they are structured as modules.
from demo_page import process_page
# setup_output_dirs is called by save_outputs, so not directly needed here.
from utils.utils import prepare_image, parse_layout_string, process_coordinates, save_outputs, ImageDimensions, map_to_original_coordinates, adjust_box_edges
from utils.markdown_utils import MarkdownConverter

# Setup logging
logging.basicConfig(level=logging.INFO) # Configure basic logging
logger = logging.getLogger(__name__)

# Load configuration
logger.info("Loading DOLPHIN model configuration...")
cfg = OmegaConf.load("./config/Dolphin.yaml")

# Create FastAPI app instance
app = FastAPI()

# Initialize DOLPHIN model
logger.info("Initializing DOLPHIN model...")
model = DOLPHIN(cfg)
logger.info("DOLPHIN model initialized.")

# Example of how to potentially store in app state (though not strictly necessary for global)
# app.state.model = model

TEMP_UPLOADS_DIR = "temp_uploads"
RESULTS_DIR = "results"
DEFAULT_MAX_BATCH_SIZE = 4

@app.on_event("startup")
async def startup_event():
    """Create temporary and results directories on startup."""
    logger.info(f"Creating temporary directory: {TEMP_UPLOADS_DIR}")
    os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)
    logger.info(f"Creating results directory: {RESULTS_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/process/")
async def process_file_endpoint(
    file: UploadFile = File(...),
    output_format: str = Query("json", enum=["json", "markdown"], description="Format for the output: 'json' or 'markdown'")
):
    """
    Accepts a file upload, saves it temporarily, processes it using DOLPHIN model,
    and returns the results in the specified format (JSON or Markdown).

    - **file**: The image file to process.
    - **output_format**: Query parameter to specify the desired output format.
        - 'json': Returns detailed JSON results of the recognition.
        - 'markdown': Returns the recognized content in Markdown format.
    """
    upload_file_path = None
    request_id = str(uuid.uuid4()) # Generate request_id early for logging

    logger.info(f"Processing request {request_id}. Output format: {output_format}. Input filename: {file.filename}")

    try:
        # Save uploaded file temporarily
        file_extension = os.path.splitext(file.filename)[1]
        # Basic check for common image file extensions
        if file_extension.lower() not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            logger.warning(f"Request {request_id}: Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid file type. Supported types: PNG, JPG, JPEG, BMP, TIFF. Got: {file_extension}")

        unique_upload_filename = f"{request_id}{file_extension}"
        upload_file_path = os.path.join(TEMP_UPLOADS_DIR, unique_upload_filename)

        logger.info(f"Request {request_id}: Saving uploaded file to {upload_file_path}")
        try:
            with open(upload_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to save uploaded file {upload_file_path}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

        # Define save directory for results for this specific file
        output_save_dir = os.path.join(RESULTS_DIR, request_id)
        logger.info(f"Request {request_id}: Creating output directory {output_save_dir}")
        try:
            os.makedirs(output_save_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Request {request_id}: Failed to create output directory {output_save_dir}. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create output directory: {str(e)}")

        # Call process_page (DOLPHIN model processing)
        logger.info(f"Request {request_id}: Starting DOLPHIN model processing for {upload_file_path}")
        try:
            json_path, _ = process_page( # recognition_results not directly used now
                image_path=upload_file_path,
                model=model,
                save_dir=output_save_dir,
                max_batch_size=DEFAULT_MAX_BATCH_SIZE
            )
            logger.info(f"Request {request_id}: DOLPHIN model processing completed. JSON path: {json_path}")
        except FileNotFoundError as e: # Specific to process_page if it expects files that aren't there
            logger.error(f"Request {request_id}: File not found during DOLPHIN model processing. Error: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"File not found during model processing: {str(e)}")
        except Exception as e:
            logger.error(f"Request {request_id}: Error during DOLPHIN model processing. Error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during DOLPHIN model processing: {str(e)}")

        # Determine output based on output_format
        if output_format == "markdown":
            base_filename = os.path.splitext(unique_upload_filename)[0]
            markdown_file_path = os.path.join(output_save_dir, "markdown", f"{base_filename}.md")
            logger.info(f"Request {request_id}: Attempting to read Markdown output from {markdown_file_path}")

            if not os.path.exists(markdown_file_path):
                logger.error(f"Request {request_id}: Markdown file not found at {markdown_file_path}")
                raise HTTPException(status_code=404, detail=f"Markdown file not found: {markdown_file_path}")
            try:
                with open(markdown_file_path, "r", encoding="utf-8") as md_file:
                    markdown_content = md_file.read()
                logger.info(f"Request {request_id}: Successfully read Markdown file. Returning content.")
                return PlainTextResponse(content=markdown_content, media_type="text/markdown")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to read Markdown file {markdown_file_path}. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to read Markdown file: {str(e)}")

        else: # Default to JSON
            logger.info(f"Request {request_id}: Attempting to read JSON output from {json_path}")
            if not os.path.exists(json_path):
                logger.error(f"Request {request_id}: JSON results file not found at {json_path}")
                raise HTTPException(status_code=404, detail=f"JSON results file not found: {json_path}")
            try:
                with open(json_path, "r", encoding="utf-8") as f_json:
                    json_content = json.load(f_json)
                logger.info(f"Request {request_id}: Successfully read and parsed JSON file. Returning content.")
                return JSONResponse(content=json_content)
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to read or parse JSON results file {json_path}. Error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to read or parse JSON results file: {str(e)}")

    except HTTPException as e: # Re-raise HTTPExceptions to be handled by FastAPI
        raise e
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Request {request_id}: An unexpected error occurred. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if file:
            try:
                file.file.close()
            except Exception as e:
                logger.warning(f"Request {request_id}: Error closing uploaded file stream. Error: {e}", exc_info=True)
        
        # Clean up the uploaded temporary file
        if upload_file_path and os.path.exists(upload_file_path):
            try:
                os.remove(upload_file_path)
                logger.info(f"Request {request_id}: Successfully removed temporary upload file {upload_file_path}")
            except Exception as e:
                logger.error(f"Request {request_id}: Failed to remove temporary upload file {upload_file_path}. Error: {e}", exc_info=True)
        # Optionally, clean up the uploaded temporary file if it's no longer needed
        # For now, keeping it for debugging, but in production, you might delete it:
        # if upload_file_path and os.path.exists(upload_file_path):
        #     os.remove(upload_file_path)


@app.get("/")
async def root():
    return {"message": "DOLPHIN API is running"}

# Further endpoints will be added in subsequent steps.
