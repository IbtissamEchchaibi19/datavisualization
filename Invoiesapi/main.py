from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional, Union
import pandas as pd
import uvicorn
import json
import hashlib
from fastapi.middleware.wsgi import WSGIMiddleware

# Import your existing classes
from invoice_extractor import InvoiceExtractor
from dashboard import app as dash_app

# Initialize FastAPI
app = FastAPI(title="Invoice Processing API", version="1.0.0")

# Mount the Dash app - this makes it accessible at /dash_app/
app.mount("/dash_app", WSGIMiddleware(dash_app.server))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic models for request/response
class DeleteInvoiceRequest(BaseModel):
    invoice_ids: List[str]  # List of invoice IDs to delete

class DeleteInvoiceResponse(BaseModel):
    message: str
    deleted_invoices: List[dict]
    not_found_invoices: List[str]
    total_deleted: int
    remaining_records: int
    errors: Optional[List[str]] = None

# Global variables
INVOICES_DIR = "invoices"
CSV_FILE = "invoice_data.csv"
PROCESSED_FILES_TRACKER = "processed_files.json"
extractor = InvoiceExtractor()

# Ensure directories exist
os.makedirs(INVOICES_DIR, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file to track if it's been processed"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_files_tracker() -> dict:
    """Load the tracker of processed files"""
    if os.path.exists(PROCESSED_FILES_TRACKER):
        try:
            with open(PROCESSED_FILES_TRACKER, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processed files tracker: {e}")
            return {}
    return {}

def save_processed_files_tracker(tracker: dict):
    """Save the tracker of processed files"""
    try:
        with open(PROCESSED_FILES_TRACKER, 'w') as f:
            json.dump(tracker, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files tracker: {e}")

def is_file_processed(file_path: str) -> bool:
    """Check if a file has already been processed"""
    tracker = load_processed_files_tracker()
    filename = os.path.basename(file_path)
    
    if filename not in tracker:
        return False
    
    try:
        current_hash = get_file_hash(file_path)
        return tracker[filename] == current_hash
    except Exception as e:
        print(f"Error checking file hash for {filename}: {e}")
        return False

def mark_file_as_processed(file_path: str):
    """Mark a file as processed in the tracker"""
    tracker = load_processed_files_tracker()
    filename = os.path.basename(file_path)
    try:
        file_hash = get_file_hash(file_path)
        tracker[filename] = file_hash
        save_processed_files_tracker(tracker)
    except Exception as e:
        print(f"Error marking file as processed {filename}: {e}")

def remove_file_from_tracker(filename: str):
    """Remove a file from the processed files tracker"""
    tracker = load_processed_files_tracker()
    if filename in tracker:
        del tracker[filename]
        save_processed_files_tracker(tracker)
        print(f"Removed {filename} from processed files tracker")

def get_invoice_records_by_ids(invoice_ids: List[str]) -> tuple:
    """
    Get invoice records from CSV by invoice IDs
    Returns: (found_records, not_found_ids)
    """
    if not os.path.exists(CSV_FILE):
        return [], invoice_ids
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Check if the CSV has an invoice_id column (adjust based on your CSV structure)
        # You might need to adjust this based on how your invoice IDs are stored
        if 'invoice_id' not in df.columns:
            # If there's no invoice_id column, you might need to use filename or another identifier
            # This is an example - adjust based on your actual CSV structure
            print("Warning: 'invoice_id' column not found in CSV. Using 'filename' instead.")
            id_column = 'filename' if 'filename' in df.columns else df.columns[0]
        else:
            id_column = 'invoice_id'
        
        found_records = []
        not_found_ids = []
        
        for invoice_id in invoice_ids:
            matching_records = df[df[id_column].astype(str) == str(invoice_id)]
            if not matching_records.empty:
                found_records.extend(matching_records.to_dict('records'))
            else:
                not_found_ids.append(invoice_id)
        
        return found_records, not_found_ids
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], invoice_ids

def delete_records_from_csv(invoice_ids: List[str]) -> int:
    """
    Delete records from CSV by invoice IDs
    Returns: number of deleted records
    """
    if not os.path.exists(CSV_FILE):
        return 0
    
    try:
        df = pd.read_csv(CSV_FILE)
        original_count = len(df)
        
        # Determine the ID column to use
        if 'invoice_id' not in df.columns:
            id_column = 'filename' if 'filename' in df.columns else df.columns[0]
        else:
            id_column = 'invoice_id'
        
        # Filter out records with matching IDs
        df_filtered = df[~df[id_column].astype(str).isin([str(id) for id in invoice_ids])]
        deleted_count = original_count - len(df_filtered)
        
        # Save the updated CSV
        df_filtered.to_csv(CSV_FILE, index=False)
        print(f"Deleted {deleted_count} records from CSV")
        
        return deleted_count
        
    except Exception as e:
        print(f"Error deleting records from CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating CSV: {str(e)}")

def delete_pdf_files(filenames: List[str]) -> List[str]:
    """
    Delete PDF files from the invoices directory
    Returns: list of successfully deleted filenames
    """
    deleted_files = []
    
    for filename in filenames:
        file_path = os.path.join(INVOICES_DIR, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(filename)
                print(f"Deleted file: {filename}")
                
                # Remove from processed files tracker
                remove_file_from_tracker(filename)
            else:
                print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
    
    return deleted_files

def append_to_csv(new_data: List[dict]):
    """Append new data to CSV file instead of rewriting"""
    if not new_data:
        return
    
    try:
        new_df = pd.DataFrame(new_data)
        
        if os.path.exists(CSV_FILE):
            # Append to existing CSV
            new_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            # Create new CSV with headers
            new_df.to_csv(CSV_FILE, index=False)
        
        print(f"Added {len(new_data)} records to CSV")
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        raise

def initialize_csv_if_needed():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        try:
            df = pd.DataFrame(columns=extractor.csv_fields)
            df.to_csv(CSV_FILE, index=False)
            print("Initialized CSV file with headers")
        except Exception as e:
            print(f"Error initializing CSV: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("Starting Invoice Processing API...")
    initialize_csv_if_needed()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Invoice Processing API",
        "endpoints": {
            "upload_invoices": "/upload-invoices/",
            "process_directory": "/process-directory/",
            "delete_invoices": "/delete-invoices/",
            "get_data": "/data/",
            "dashboard": "/dash_app/",
            "health": "/health/"
        }
    }

@app.get("/dashboard/")
async def get_dashboard():
    """Redirect directly to the dashboard"""
    return RedirectResponse(url="")

@app.delete("/delete-invoices/")
async def delete_invoices(request: DeleteInvoiceRequest):
    """
    Delete one or multiple invoices by their IDs
    This will remove both the PDF files and their records from the CSV
    """
    if not request.invoice_ids:
        raise HTTPException(status_code=400, detail="No invoice IDs provided")
    
    try:
        # Get invoice records that match the IDs
        found_records, not_found_ids = get_invoice_records_by_ids(request.invoice_ids)
        
        if not found_records:
            raise HTTPException(
                status_code=404, 
                detail=f"No invoices found with IDs: {', '.join(request.invoice_ids)}"
            )
        
        deleted_invoices = []
        errors = []
        
        # Delete records from CSV
        try:
            deleted_count = delete_records_from_csv(request.invoice_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting from CSV: {str(e)}")
        
        # Extract filenames from the found records and delete PDF files
        filenames_to_delete = []
        for record in found_records:
            # Adjust this based on your CSV structure - might be 'filename', 'file_name', etc.
            filename = record.get('filename') or record.get('file_name') or record.get('source_file')
            if filename:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'  # Add extension if missing
                filenames_to_delete.append(filename)
        
        # Delete PDF files
        deleted_files = delete_pdf_files(filenames_to_delete)
        
        # Prepare response data
        for record in found_records:
            invoice_id = str(record.get('invoice_id', record.get('filename', 'unknown')))
            deleted_invoices.append({
                "invoice_id": invoice_id,
                "filename": record.get('filename', 'unknown'),
                "deleted_from_csv": True,
                "deleted_pdf_file": record.get('filename') in deleted_files if record.get('filename') else False
            })
        
        # Get remaining record count
        remaining_records = 0
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                remaining_records = len(df)
        except Exception as e:
            print(f"Error counting remaining records: {e}")
        
        return DeleteInvoiceResponse(
            message=f"Successfully deleted {len(deleted_invoices)} invoice(s)",
            deleted_invoices=deleted_invoices,
            not_found_invoices=not_found_ids,
            total_deleted=len(deleted_invoices),
            remaining_records=remaining_records,
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting invoices: {str(e)}")

@app.get("/invoices/")
async def list_invoices():
    """
    List all invoices with their IDs for reference
    This helps users know which IDs they can delete
    """
    if not os.path.exists(CSV_FILE):
        return {
            "message": "No invoice data found",
            "invoices": [],
            "total_count": 0
        }
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        # Determine the ID column
        if 'invoice_id' not in df.columns:
            id_column = 'filename' if 'filename' in df.columns else df.columns[0]
        else:
            id_column = 'invoice_id'
        
        # Create a list of invoices with their basic info
        invoices = []
        for _, row in df.iterrows():
            invoice_info = {
                "id": str(row[id_column]),
                "filename": row.get('filename', 'unknown'),
            }
            # Add other relevant fields if they exist
            for field in ['invoice_number', 'date', 'total_amount', 'vendor']:
                if field in row and pd.notna(row[field]):
                    invoice_info[field] = row[field]
            
            invoices.append(invoice_info)
        
        return {
            "message": f"Found {len(invoices)} invoice(s)",
            "invoices": invoices,
            "total_count": len(invoices)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing invoices: {str(e)}")

@app.post("/upload-invoices/")
async def upload_invoices(files: Union[List[UploadFile], UploadFile] = File(...)):
    """
    Upload and process invoice PDFs (supports both single and multiple files)
    This endpoint handles both single file upload and multiple file upload
    """
    # Convert single file to list for uniform processing
    if isinstance(files, UploadFile):
        files = [files]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_files = []
    total_new_records = 0
    errors = []
    all_new_data = []
    skipped_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            errors.append(f"{file.filename}: Only PDF files are allowed")
            continue
        
        file_path = os.path.join(INVOICES_DIR, file.filename)
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Check if this file has already been processed
            if is_file_processed(file_path):
                skipped_files.append({
                    "filename": file.filename,
                    "reason": "Already processed (no changes detected)"
                })
                continue
            
            # Process the new/changed file
            print(f"Processing new/changed file: {file.filename}")
            invoice_data = extractor.process_invoice(file_path)
            
            if invoice_data:
                all_new_data.extend(invoice_data)
                mark_file_as_processed(file_path)
                
                processed_files.append({
                    "filename": file.filename,
                    "records_added": len(invoice_data)
                })
                total_new_records += len(invoice_data)
                print(f"Successfully processed {file.filename}: {len(invoice_data)} records")
            else:
                errors.append(f"{file.filename}: No data extracted")
                
        except Exception as e:
            error_msg = f"{file.filename}: {str(e)}"
            errors.append(error_msg)
            print(f"Error processing {file.filename}: {e}")
            
            # Clean up file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    # Append all new data to CSV in one operation
    if all_new_data:
        try:
            append_to_csv(all_new_data)
            print(f"Added {len(all_new_data)} total records to CSV")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving data to CSV: {str(e)}")
    
    # Get current total records
    total_records = 0
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            total_records = len(df)
    except Exception as e:
        print(f"Error reading CSV for count: {e}")
    
    return {
        "message": f"Upload completed. Processed {len(processed_files)} new/changed files",
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "total_new_records": total_new_records,
        "total_records": total_records,
        "errors": errors if errors else None
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    try:
        # Get processing status
        pdf_count = 0
        processed_count = 0
        unprocessed_count = 0
        
        try:
            if os.path.exists(INVOICES_DIR):
                pdf_files = [f for f in os.listdir(INVOICES_DIR) if f.endswith('.pdf')]
                pdf_count = len(pdf_files)
                
                for filename in pdf_files:
                    file_path = os.path.join(INVOICES_DIR, filename)
                    if is_file_processed(file_path):
                        processed_count += 1
                    else:
                        unprocessed_count += 1
                        
        except Exception as e:
            print(f"Warning: Could not get processing status: {e}")
        
        # Get CSV record count
        csv_records = 0
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                csv_records = len(df)
        except Exception as e:
            print(f"Warning: Could not count CSV records: {e}")
        
        return {
            "status": "healthy",
            "message": "Invoice Processing API is running",
            "dashboard_status": "mounted at /dash_app/",
            "csv_exists": os.path.exists(CSV_FILE),
            "csv_records": csv_records,
            "invoices_directory_exists": os.path.exists(INVOICES_DIR),
            "total_pdf_files": pdf_count,
            "processed_files": processed_count,
            "unprocessed_files": unprocessed_count
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

if __name__ == "__main__":
    print("Starting Invoice Processing API...")
    print("API will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Dashboard will be available at: http://localhost:8000/dash_app/")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)