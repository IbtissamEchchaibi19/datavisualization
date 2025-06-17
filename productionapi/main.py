from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import tempfile
from typing import List, Set, Optional, Union
import pandas as pd
from datetime import datetime
import uvicorn
import threading
import time
import json
import hashlib
import traceback

# Import your existing classes
from extract_production_data import HoneyProductionExtractor  # Your existing extractor
from dashboard import app as dash_app # Your existing dashboard

# Initialize FastAPI
app = FastAPI(title="Honey Production Processing API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8001", 
        "http://localhost:8051",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8051"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic models for request/response
class DeleteReportRequest(BaseModel):
    report_ids: List[str]  # List of batch numbers or report IDs to delete

class DeleteReportResponse(BaseModel):
    message: str
    deleted_reports: List[dict]
    not_found_reports: List[str]
    total_deleted: int
    remaining_records: int
    errors: Optional[List[str]] = None

# Global variables
REPORTS_DIR = "honey_production_reports"
CSV_FILE = "honey_production_data.csv"
PROCESSED_FILES_TRACKER = "processed_production_files.json"
extractor = HoneyProductionExtractor()

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# Dash app will run on port 8051
dash_thread = None

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file to track if it's been processed"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error generating hash for {file_path}: {e}")
        return ""

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
    try:
        if not os.path.exists(file_path):
            return False
            
        tracker = load_processed_files_tracker()
        filename = os.path.basename(file_path)
        
        if filename not in tracker:
            print(f"File {filename} not in tracker - will process")
            return False
        
        current_hash = get_file_hash(file_path)
        if not current_hash:
            print(f"Could not generate hash for {filename} - will process")
            return False
            
        is_same = tracker[filename] == current_hash
        print(f"File {filename}: hash match = {is_same}")
        return is_same
        
    except Exception as e:
        print(f"Error checking if file processed {file_path}: {e}")
        return False

def mark_file_as_processed(file_path: str):
    """Mark a file as processed in the tracker"""
    try:
        if not os.path.exists(file_path):
            print(f"Cannot mark non-existent file as processed: {file_path}")
            return
            
        tracker = load_processed_files_tracker()
        filename = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)
        
        if file_hash:
            tracker[filename] = file_hash
            save_processed_files_tracker(tracker)
            print(f"Marked {filename} as processed")
        else:
            print(f"Could not mark {filename} as processed - no hash generated")
    except Exception as e:
        print(f"Error marking file as processed {file_path}: {e}")

def remove_file_from_tracker(filename: str):
    """Remove a file from the processed files tracker"""
    try:
        tracker = load_processed_files_tracker()
        if filename in tracker:
            del tracker[filename]
            save_processed_files_tracker(tracker)
            print(f"Removed {filename} from processed files tracker")
    except Exception as e:
        print(f"Error removing {filename} from tracker: {e}")

def get_csv_columns():
    """Get CSV columns from extractor or use default columns"""
    try:
        # Try to get columns from extractor
        if hasattr(extractor, 'csv_fields') and extractor.csv_fields:
            return extractor.csv_fields
        elif hasattr(extractor, 'get_csv_columns'):
            return extractor.get_csv_columns()
        else:
            # Default columns if extractor doesn't specify
            return [
                'batch_number', 'report_year', 'company_name', 'location',
                'apiary_number', 'production_kg', 'extraction_date'
            ]
    except Exception as e:
        print(f"Error getting CSV columns: {e}")
        # Fallback columns
        return [
            'batch_number', 'report_year', 'company_name', 'location',
            'apiary_number', 'production_kg', 'extraction_date'
        ]

def initialize_csv_if_needed():
    """Initialize CSV file with headers if it doesn't exist"""
    try:
        if not os.path.exists(CSV_FILE):
            columns = get_csv_columns()
            df = pd.DataFrame(columns=columns)
            df.to_csv(CSV_FILE, index=False)
            print(f"Initialized CSV file with columns: {columns}")
        else:
            print(f"CSV file already exists: {CSV_FILE}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")
        raise

# Debug and Fix for CSV Update Issues

# Problem Analysis:
# 1. The API reports adding records but CSV file doesn't reflect the changes
# 2. Possible issues:
#    - CSV file permissions
#    - Race condition in file operations
#    - Incorrect CSV path being used
#    - CSV append operation failing silently
#    - File handle not being closed properly

# Add these improved functions to your FastAPI code:

def append_to_csv_with_verification(new_data: List[dict]):
    """
    Improved CSV append function with verification and better error handling
    """
    if not new_data:
        print("No data to append")
        return 0
    
    try:
        # Get record count before append
        records_before = 0
        if os.path.exists(CSV_FILE):
            try:
                df_before = pd.read_csv(CSV_FILE)
                records_before = len(df_before)
                print(f"CSV before append: {records_before} records")
            except Exception as e:
                print(f"Error reading CSV before append: {e}")
                records_before = 0
        
        # Create DataFrame from new data
        new_df = pd.DataFrame(new_data)
        print(f"New DataFrame created with {len(new_df)} records")
        print(f"New DataFrame columns: {list(new_df.columns)}")
        
        # Verify CSV file path and permissions
        csv_abs_path = os.path.abspath(CSV_FILE)
        print(f"Absolute CSV path: {csv_abs_path}")
        print(f"CSV directory exists: {os.path.exists(os.path.dirname(csv_abs_path))}")
        print(f"CSV file exists: {os.path.exists(csv_abs_path)}")
        
        if os.path.exists(csv_abs_path):
            # Check file permissions
            print(f"CSV file readable: {os.access(csv_abs_path, os.R_OK)}")
            print(f"CSV file writable: {os.access(csv_abs_path, os.W_OK)}")
        
        # Append to CSV with explicit flushing
        if os.path.exists(CSV_FILE):
            # Append mode
            print("Appending to existing CSV...")
            new_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            # Create new file
            print("Creating new CSV...")
            new_df.to_csv(CSV_FILE, index=False)
        
        # Force file system sync
        import sys
        if hasattr(os, 'sync'):
            os.sync()  # Unix/Linux
        elif sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(-1)
        
        # Verify the append worked
        time.sleep(0.1)  # Small delay to ensure file system sync
        
        records_after = 0
        if os.path.exists(CSV_FILE):
            try:
                df_after = pd.read_csv(CSV_FILE)
                records_after = len(df_after)
                print(f"CSV after append: {records_after} records")
            except Exception as e:
                print(f"Error reading CSV after append: {e}")
                raise
        
        records_added = records_after - records_before
        print(f"Records actually added: {records_added}")
        
        if records_added != len(new_data):
            raise Exception(f"Mismatch: Expected to add {len(new_data)} records, but only {records_added} were added")
        
        return records_added
        
    except Exception as e:
        print(f"Error in append_to_csv_with_verification: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def get_current_csv_count():
    """Get current record count from CSV with error handling"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            count = len(df)
            print(f"Current CSV count: {count}")
            return count
        else:
            print("CSV file doesn't exist")
            return 0
    except Exception as e:
        print(f"Error reading CSV count: {e}")
        return -1  # Return -1 to indicate error

# Updated upload endpoint with better debugging
@app.post("/upload-reports-debug/")
async def upload_reports_with_debug(files: Union[List[UploadFile], UploadFile] = File(...)):
    """
    Upload and process honey production report PDFs with detailed debugging
    """
    # Convert single file to list for uniform processing
    if isinstance(files, UploadFile):
        files = [files]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    print(f"\n=== UPLOAD WITH DEBUG STARTED: {len(files)} files ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"CSV file path: {os.path.abspath(CSV_FILE)}")
    print(f"Reports directory: {os.path.abspath(REPORTS_DIR)}")
    
    # Check initial CSV state
    initial_count = get_current_csv_count()
    print(f"Initial CSV count: {initial_count}")
    
    processed_files = []
    total_new_records = 0
    errors = []
    all_new_data = []
    skipped_files = []
    
    # Ensure CSV is initialized
    try:
        initialize_csv_if_needed()
        post_init_count = get_current_csv_count()
        print(f"Post-initialization CSV count: {post_init_count}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing CSV: {str(e)}")
    
    for file in files:
        print(f"\n--- Processing file: {file.filename} ---")
        
        if not file.filename.endswith('.pdf'):
            error_msg = f"{file.filename}: Only PDF files are allowed"
            errors.append(error_msg)
            print(error_msg)
            continue
        
        file_path = os.path.join(REPORTS_DIR, file.filename)
        
        try:
            # Save the uploaded file
            print(f"Saving file to: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            print(f"File saved successfully, size: {file_size} bytes")
            
            # Check if this file has already been processed
            if is_file_processed(file_path):
                print(f"File {file.filename} already processed - skipping")
                skipped_files.append({
                    "filename": file.filename,
                    "reason": "Already processed (no changes detected)"
                })
                continue
            
            # Process the new/changed file
            print(f"Processing new/changed file: {file.filename}")
            try:
                report_data = extractor.process_report(file_path)
                print(f"Extractor returned: {len(report_data) if report_data else 0} records")
                
                if report_data and len(report_data) > 0:
                    print(f"Sample record: {report_data[0]}")
                    all_new_data.extend(report_data)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "records_added": len(report_data)
                    })
                    total_new_records += len(report_data)
                    print(f"Successfully processed {file.filename}: {len(report_data)} records")
                    
                else:
                    error_msg = f"{file.filename}: No data extracted from PDF"
                    errors.append(error_msg)
                    print(error_msg)
                    
            except Exception as e:
                error_msg = f"{file.filename}: Error processing PDF - {str(e)}"
                errors.append(error_msg)
                print(f"Error processing {file.filename}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            error_msg = f"{file.filename}: Error saving file - {str(e)}"
            errors.append(error_msg)
            print(f"Error handling {file.filename}: {e}")
    
    # CSV Update with verification
    print(f"\n--- CSV UPDATE WITH VERIFICATION ---")
    print(f"Total new records to add: {len(all_new_data)}")
    
    pre_append_count = get_current_csv_count()
    print(f"Pre-append CSV count: {pre_append_count}")
    
    if all_new_data:
        try:
            records_actually_added = append_to_csv_with_verification(all_new_data)
            print(f"Records actually added to CSV: {records_actually_added}")
            
            # Mark files as processed only if CSV update succeeded
            if records_actually_added > 0:
                for file in files:
                    if file.filename.endswith('.pdf'):
                        file_path = os.path.join(REPORTS_DIR, file.filename)
                        if os.path.exists(file_path) and not is_file_processed(file_path):
                            mark_file_as_processed(file_path)
                            print(f"Marked {file.filename} as processed")
            
        except Exception as e:
            print(f"Critical error saving to CSV: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error saving data to CSV: {str(e)}")
    else:
        print("No new data to add to CSV")
    
    # Final verification
    final_count = get_current_csv_count()
    print(f"Final CSV count: {final_count}")
    actual_records_added = final_count - initial_count if initial_count >= 0 and final_count >= 0 else 0
    
    print(f"\n=== UPLOAD DEBUG COMPLETED ===")
    print(f"Expected new records: {total_new_records}")
    print(f"Actual records added: {actual_records_added}")
    print(f"Initial count: {initial_count}")
    print(f"Final count: {final_count}")
    
    return {
        "message": f"Upload completed. Processed {len(processed_files)} new/changed files",
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "total_new_records": total_new_records,
        "actual_records_added": actual_records_added,
        "csv_counts": {
            "initial": initial_count,
            "final": final_count,
            "difference": actual_records_added
        },
        "total_records": final_count,
        "errors": errors if errors else None,
        "debug_info": {
            "csv_file_path": os.path.abspath(CSV_FILE),
            "working_directory": os.getcwd(),
            "csv_exists": os.path.exists(CSV_FILE)
        }
    }

# Additional debugging endpoint
@app.get("/verify-csv-state/")
async def verify_csv_state():
    """
    Verify the current state of the CSV file
    """
    csv_info = {
        "csv_file_path": os.path.abspath(CSV_FILE),
        "csv_exists": os.path.exists(CSV_FILE),
        "working_directory": os.getcwd(),
        "reports_dir": os.path.abspath(REPORTS_DIR),
        "reports_dir_exists": os.path.exists(REPORTS_DIR)
    }
    
    if os.path.exists(CSV_FILE):
        try:
            stat_info = os.stat(CSV_FILE)
            csv_info.update({
                "file_size": stat_info.st_size,
                "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "readable": os.access(CSV_FILE, os.R_OK),
                "writable": os.access(CSV_FILE, os.W_OK)
            })
            
            # Read CSV content
            df = pd.read_csv(CSV_FILE)
            csv_info.update({
                "record_count": len(df),
                "columns": list(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum()
            })
            
            if len(df) > 0:
                csv_info["last_5_records"] = df.tail(5).to_dict('records')
                
        except Exception as e:
            csv_info["read_error"] = str(e)
    
    return csv_info

# Immediate fix: Replace the append_to_csv function in your existing code
def append_to_csv_fixed(new_data: List[dict]):
    """Fixed version of append_to_csv with proper error handling"""
    if not new_data:
        return
    
    try:
        # Get absolute path
        csv_path = os.path.abspath(CSV_FILE)
        print(f"Appending to CSV at: {csv_path}")
        
        # Create DataFrame
        new_df = pd.DataFrame(new_data)
        print(f"Created DataFrame with {len(new_df)} records")
        
        # Check if file exists and append accordingly
        if os.path.exists(csv_path):
            # Read existing CSV to verify structure
            existing_df = pd.read_csv(csv_path)
            print(f"Existing CSV has {len(existing_df)} records")
            
            # Ensure column alignment
            if set(new_df.columns) != set(existing_df.columns):
                print(f"Column mismatch - Existing: {list(existing_df.columns)}, New: {list(new_df.columns)}")
                # Reorder new_df columns to match existing
                new_df = new_df.reindex(columns=existing_df.columns, fill_value='')
            
            # Append to existing file
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                new_df.to_csv(f, header=False, index=False)
            
        else:
            # Create new file
            new_df.to_csv(csv_path, index=False)
        
        # Verify the append worked
        verification_df = pd.read_csv(csv_path)
        print(f"CSV now contains {len(verification_df)} records")
        
    except Exception as e:
        print(f"Error in append_to_csv_fixed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise
def delete_records_from_csv(report_ids: List[str]) -> int:
    """
    Delete records from CSV by batch numbers
    Returns: number of deleted records
    """
    if not os.path.exists(CSV_FILE):
        print("CSV file doesn't exist - nothing to delete")
        return 0
    
    try:
        df = pd.read_csv(CSV_FILE)
        original_count = len(df)
        print(f"Original CSV has {original_count} records")
        
        # Filter out records with matching batch numbers
        df_filtered = df[~df['batch_number'].astype(str).isin([str(id) for id in report_ids])]
        deleted_count = original_count - len(df_filtered)
        
        # Save the updated CSV
        df_filtered.to_csv(CSV_FILE, index=False)
        print(f"Deleted {deleted_count} records from CSV, {len(df_filtered)} remaining")
        
        return deleted_count
        
    except Exception as e:
        print(f"Error deleting records from CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating CSV: {str(e)}")

def delete_pdf_files(filenames: List[str]) -> List[str]:
    """
    Delete PDF files from the reports directory
    Returns: list of successfully deleted filenames
    """
    deleted_files = []
    
    for filename in filenames:
        file_path = os.path.join(REPORTS_DIR, filename)
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

def run_dash_app():
    """Run Dash app in a separate thread"""
    try:
        dash_app.run(host='0.0.0.0', port=8051, debug=False)
    except Exception as e:
        print(f"Error running Dash app: {e}")

def start_dash_app():
    """Start Dash app in background thread"""
    global dash_thread
    if dash_thread is None or not dash_thread.is_alive():
        dash_thread = threading.Thread(target=run_dash_app, daemon=True)
        dash_thread.start()
        time.sleep(2)
        print("Dash app started on port 8051")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("Starting Honey Production Processing API...")
    try:
        initialize_csv_if_needed()
        start_dash_app()
        print("Startup completed successfully")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Honey Production Processing API",
        "endpoints": {
            "upload_reports": "/upload-reports/",
            "delete_reports": "/delete-reports/",
            "get_reports": "/reports/",
            "get_data": "/data/",
            "dashboard": "/dashboard/",
            "health": "/health/"
        }
    }

@app.get("/dashboard/")
async def get_dashboard(request: Request):
    """Get dashboard URL"""
    start_dash_app()
    
    host = request.headers.get("host", "localhost:8001")
    base_url = f"http://{host.split(':')[0]}:8051"
    
    return {
        "dashboard_url": base_url,
        "message": "Dashboard is running",
        "iframe_endpoint": "/dashboard-iframe/"
    }

@app.delete("/delete-reports/")
async def delete_reports(request: DeleteReportRequest):
    """
    Delete one or multiple production reports by their batch numbers
    This will remove both the PDF files and their records from the CSV
    """
    if not request.report_ids:
        raise HTTPException(status_code=400, detail="No report IDs provided")
    
    try:
        print(f"Attempting to delete reports: {request.report_ids}")
        
        # Get report records that match the IDs
        found_records, not_found_ids = get_report_records_by_ids(request.report_ids)
        
        if not found_records:
            raise HTTPException(
                status_code=404, 
                detail=f"No reports found with IDs: {', '.join(request.report_ids)}"
            )
        
        deleted_reports = []
        errors = []
        
        # Delete records from CSV
        try:
            deleted_count = delete_records_from_csv(request.report_ids)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting from CSV: {str(e)}")
        
        # Extract filenames from the found records and delete PDF files
        filenames_to_delete = []
        for record in found_records:
            # Try to determine the filename from the batch number or other fields
            batch_number = record.get('batch_number', 'unknown')
            # You might need to adjust this based on your naming convention
            potential_filename = f"{batch_number}.pdf"
            filenames_to_delete.append(potential_filename)
        
        # Delete PDF files
        deleted_files = delete_pdf_files(filenames_to_delete)
        
        # Prepare response data
        for record in found_records:
            batch_number = str(record.get('batch_number', 'unknown'))
            deleted_reports.append({
                "batch_number": batch_number,
                "apiary_number": record.get('apiary_number', 'unknown'),
                "location": record.get('location', 'unknown'),
                "deleted_from_csv": True,
                "deleted_pdf_file": f"{batch_number}.pdf" in deleted_files
            })
        
        # Get remaining record count
        remaining_records = 0
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                remaining_records = len(df)
        except Exception as e:
            print(f"Error counting remaining records: {e}")
        
        return DeleteReportResponse(
            message=f"Successfully deleted {len(deleted_reports)} report(s)",
            deleted_reports=deleted_reports,
            not_found_reports=not_found_ids,
            total_deleted=len(deleted_reports),
            remaining_records=remaining_records,
            errors=errors if errors else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting reports: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting reports: {str(e)}")

@app.get("/reports/")
async def list_reports():
    """
    List all production reports with their batch numbers for reference
    This helps users know which IDs they can delete
    """
    if not os.path.exists(CSV_FILE):
        return {
            "message": "No production report data found",
            "reports": [],
            "total_count": 0
        }
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Listed reports from CSV with {len(df)} records")
        
        if df.empty:
            return {
                "message": "CSV file exists but contains no data",
                "reports": [],
                "total_count": 0
            }
        
        # Group by batch_number to get unique reports
        if 'batch_number' not in df.columns:
            print(f"Warning: batch_number column missing. Available columns: {list(df.columns)}")
            return {
                "message": "CSV structure error - missing batch_number column",
                "reports": [],
                "total_count": 0
            }
        
        unique_reports = df.groupby('batch_number').agg({
            'report_year': 'first' if 'report_year' in df.columns else lambda x: 'N/A',
            'company_name': 'first' if 'company_name' in df.columns else lambda x: 'N/A',
            'location': lambda x: ', '.join(x.unique()) if 'location' in df.columns else 'N/A',
            'apiary_number': 'count',
            'production_kg': 'sum' if 'production_kg' in df.columns else lambda x: 0,
            'extraction_date': 'first' if 'extraction_date' in df.columns else lambda x: 'N/A'
        }).reset_index()
        
        reports = []
        for _, row in unique_reports.iterrows():
            report_info = {
                "batch_number": str(row['batch_number']),
                "report_year": row.get('report_year', 'N/A'),
                "company_name": row.get('company_name', 'N/A'),
                "locations": row.get('location', 'N/A'),
                "total_apiaries": row.get('apiary_number', 0),
                "total_production_kg": row.get('production_kg', 0),
                "extraction_date": row.get('extraction_date', 'N/A')
            }
            reports.append(report_info)
        
        return {
            "message": f"Found {len(reports)} unique report(s)",
            "reports": reports,
            "total_count": len(reports),
            "total_apiary_records": len(df)
        }
        
    except Exception as e:
        print(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

@app.post("/upload-reports/")
async def upload_reports(files: Union[List[UploadFile], UploadFile] = File(...)):
    """
    Upload and process honey production report PDFs (supports both single and multiple files)
    This endpoint handles both single file upload and multiple file upload
    """
    # Convert single file to list for uniform processing
    if isinstance(files, UploadFile):
        files = [files]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    print(f"\n=== UPLOAD STARTED: {len(files)} files ===")
    
    processed_files = []
    total_new_records = 0
    errors = []
    all_new_data = []
    skipped_files = []
    
    # Ensure CSV is initialized
    try:
        initialize_csv_if_needed()
    except Exception as e:
        print(f"Error initializing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing CSV: {str(e)}")
    
    for file in files:
        print(f"\n--- Processing file: {file.filename} ---")
        
        if not file.filename.endswith('.pdf'):
            error_msg = f"{file.filename}: Only PDF files are allowed"
            errors.append(error_msg)
            print(error_msg)
            continue
        
        file_path = os.path.join(REPORTS_DIR, file.filename)
        
        try:
            # Save the uploaded file
            print(f"Saving file to: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            print(f"File saved successfully, size: {file_size} bytes")
            
            # Check if this file has already been processed
            if is_file_processed(file_path):
                print(f"File {file.filename} already processed - skipping")
                skipped_files.append({
                    "filename": file.filename,
                    "reason": "Already processed (no changes detected)"
                })
                continue
            
            # Process the new/changed file
            print(f"Processing new/changed file: {file.filename}")
            try:
                report_data = extractor.process_report(file_path)
                print(f"Extractor returned: {len(report_data) if report_data else 0} records")
                
                if report_data and len(report_data) > 0:
                    print(f"Sample record: {report_data[0]}")
                    all_new_data.extend(report_data)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "records_added": len(report_data)
                    })
                    total_new_records += len(report_data)
                    print(f"Successfully processed {file.filename}: {len(report_data)} records")
                    
                    # Mark as processed only after successful processing
                    mark_file_as_processed(file_path)
                else:
                    error_msg = f"{file.filename}: No data extracted from PDF"
                    errors.append(error_msg)
                    print(error_msg)
                    
            except Exception as e:
                error_msg = f"{file.filename}: Error processing PDF - {str(e)}"
                errors.append(error_msg)
                print(f"Error processing {file.filename}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            error_msg = f"{file.filename}: Error saving file - {str(e)}"
            errors.append(error_msg)
            print(f"Error handling {file.filename}: {e}")
            
            # Clean up file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Cleaned up failed file: {file_path}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up file: {cleanup_error}")
    
    # Append all new data to CSV in one operation
    print(f"\n--- CSV Update Phase ---")
    print(f"Total new records to add: {len(all_new_data)}")
    
    if all_new_data:
        try:
            append_to_csv(all_new_data)
            print(f"Successfully added {len(all_new_data)} records to CSV")
        except Exception as e:
            print(f"Critical error saving to CSV: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving data to CSV: {str(e)}")
    else:
        print("No new data to add to CSV")
    
    # Get current total records for response
    total_records = 0
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            total_records = len(df)
            print(f"CSV now contains {total_records} total records")
    except Exception as e:
        print(f"Error reading CSV for final count: {e}")
    
    print(f"\n=== UPLOAD COMPLETED ===")
    print(f"Processed: {len(processed_files)} files")
    print(f"Skipped: {len(skipped_files)} files")
    print(f"Errors: {len(errors)}")
    print(f"New records: {total_new_records}")
    print(f"Total records: {total_records}")
    
    return {
        "message": f"Upload completed. Processed {len(processed_files)} new/changed files",
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "total_new_records": total_new_records,
        "total_records": total_records,
        "errors": errors if errors else None
    }

@app.get("/data/")
async def get_data():
    """Get all production data from CSV"""
    if not os.path.exists(CSV_FILE):
        return {
            "message": "No data available",
            "data": [],
            "total_records": 0
        }
    
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Retrieved {len(df)} records from CSV")
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        return {
            "message": f"Retrieved {len(data)} records",
            "data": data,
            "total_records": len(data),
            "unique_batches": df['batch_number'].nunique() if 'batch_number' in df.columns else 0
        }
    except Exception as e:
        print(f"Error reading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    try:
        dashboard_status = "running" if dash_thread and dash_thread.is_alive() else "stopped"
        
        # Get processing status
        pdf_count = 0
        processed_count = 0
        unprocessed_count = 0
        
        try:
            if os.path.exists(REPORTS_DIR):
                pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
                pdf_count = len(pdf_files)
                
                for filename in pdf_files:
                    file_path = os.path.join(REPORTS_DIR, filename)
                    if is_file_processed(file_path):
                        processed_count += 1
                    else:
                        unprocessed_count += 1
                        
        except Exception as e:
            print(f"Warning: Could not get processing status: {e}")
        
        # Get CSV record count
        csv_records = 0
        csv_exists = os.path.exists(CSV_FILE)
        try:
            if csv_exists:
                df = pd.read_csv(CSV_FILE)
                csv_records = len(df)
        except Exception as e:
            print(f"Warning: Could not count CSV records: {e}")
        
        return {
            "status": "healthy",
            "message": "Honey Production Processing API is running",
            "dashboard_status": dashboard_status,
            "csv_exists": csv_exists,
            "csv_records": csv_records,
            "reports_directory_exists": os.path.exists(REPORTS_DIR),
            "total_pdf_files": pdf_count,
            "processed_files": processed_count,
            "unprocessed_files": unprocessed_count,
            "csv_file_path": os.path.abspath(CSV_FILE),
            "reports_dir_path": os.path.abspath(REPORTS_DIR)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

# Add these diagnostic endpoints to your FastAPI code to debug the issue

@app.post("/force-reprocess/")
async def force_reprocess_files(files: Union[List[UploadFile], UploadFile] = File(...)):
    """
    Force reprocess files by clearing their processed status first
    This will help us debug what's actually happening during processing
    """
    # Convert single file to list for uniform processing
    if isinstance(files, UploadFile):
        files = [files]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    print(f"\n=== FORCE REPROCESS STARTED: {len(files)} files ===")
    
    # Clear processed status for all uploaded files
    for file in files:
        if file.filename.endswith('.pdf'):
            remove_file_from_tracker(file.filename)
            print(f"Cleared processed status for: {file.filename}")
    
    # Now process them normally (copy the upload logic but with more debugging)
    processed_files = []
    total_new_records = 0
    errors = []
    all_new_data = []
    
    # Ensure CSV is initialized
    try:
        initialize_csv_if_needed()
        print(f"CSV initialized. Current record count: {len(pd.read_csv(CSV_FILE)) if os.path.exists(CSV_FILE) else 0}")
    except Exception as e:
        print(f"Error initializing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing CSV: {str(e)}")
    
    for file in files:
        print(f"\n--- FORCE Processing file: {file.filename} ---")
        
        if not file.filename.endswith('.pdf'):
            error_msg = f"{file.filename}: Only PDF files are allowed"
            errors.append(error_msg)
            continue
        
        file_path = os.path.join(REPORTS_DIR, file.filename)
        
        try:
            # Save the uploaded file
            print(f"Saving file to: {file_path}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_size = os.path.getsize(file_path)
            print(f"File saved successfully, size: {file_size} bytes")
            
            # FORCE process without checking if already processed
            print(f"FORCE processing file: {file.filename}")
            
            try:
                # Debug: Check if extractor exists and has methods
                print(f"Extractor type: {type(extractor)}")
                print(f"Extractor methods: {[method for method in dir(extractor) if not method.startswith('_')]}")
                
                # Call the extractor
                report_data = extractor.process_report(file_path)
                print(f"Extractor returned data type: {type(report_data)}")
                print(f"Extractor returned record count: {len(report_data) if report_data else 0}")
                
                if report_data and len(report_data) > 0:
                    print(f"First record structure: {report_data[0]}")
                    print(f"First record keys: {list(report_data[0].keys()) if isinstance(report_data[0], dict) else 'Not a dict'}")
                    
                    all_new_data.extend(report_data)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "records_added": len(report_data)
                    })
                    total_new_records += len(report_data)
                    print(f"Successfully processed {file.filename}: {len(report_data)} records")
                    
                else:
                    error_msg = f"{file.filename}: Extractor returned no data"
                    errors.append(error_msg)
                    print(error_msg)
                    
            except Exception as e:
                error_msg = f"{file.filename}: Error during extraction - {str(e)}"
                errors.append(error_msg)
                print(f"Error processing {file.filename}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
        except Exception as e:
            error_msg = f"{file.filename}: Error saving file - {str(e)}"
            errors.append(error_msg)
            print(f"Error handling {file.filename}: {e}")
    
    # Debug CSV append
    print(f"\n--- CSV APPEND DEBUG ---")
    print(f"Records to append: {len(all_new_data)}")
    
    if all_new_data:
        print(f"Sample record for CSV: {all_new_data[0]}")
        
        # Check CSV before append
        csv_before_count = 0
        if os.path.exists(CSV_FILE):
            df_before = pd.read_csv(CSV_FILE)
            csv_before_count = len(df_before)
            print(f"CSV before append: {csv_before_count} records")
            print(f"CSV columns: {list(df_before.columns)}")
        
        try:
            # Manual CSV append with detailed logging
            new_df = pd.DataFrame(all_new_data)
            print(f"New DataFrame shape: {new_df.shape}")
            print(f"New DataFrame columns: {list(new_df.columns)}")
            
            # Append to CSV
            if os.path.exists(CSV_FILE):
                new_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
                print("Appended to existing CSV")
            else:
                new_df.to_csv(CSV_FILE, index=False)
                print("Created new CSV")
            
            # Verify append
            df_after = pd.read_csv(CSV_FILE)
            csv_after_count = len(df_after)
            print(f"CSV after append: {csv_after_count} records")
            print(f"Records actually added: {csv_after_count - csv_before_count}")
            
            # Mark files as processed only if CSV update succeeded
            if csv_after_count > csv_before_count:
                for file in files:
                    if file.filename.endswith('.pdf'):
                        file_path = os.path.join(REPORTS_DIR, file.filename)
                        mark_file_as_processed(file_path)
                        print(f"Marked {file.filename} as processed")
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error saving data to CSV: {str(e)}")
    
    # Final count
    total_records = 0
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            total_records = len(df)
    except Exception as e:
        print(f"Error reading final CSV count: {e}")
    
    return {
        "message": f"Force reprocess completed. Processed {len(processed_files)} files",
        "processed_files": processed_files,
        "total_new_records": total_new_records,
        "total_records": total_records,
        "errors": errors if errors else None,
        "debug_info": {
            "all_new_data_count": len(all_new_data),
            "csv_exists": os.path.exists(CSV_FILE),
            "csv_path": os.path.abspath(CSV_FILE)
        }
    }

@app.get("/debug-status/")
async def debug_status():
    """
    Get detailed debug information about the current state
    """
    debug_info = {
        "csv_file": {
            "exists": os.path.exists(CSV_FILE),
            "path": os.path.abspath(CSV_FILE),
            "size": os.path.getsize(CSV_FILE) if os.path.exists(CSV_FILE) else 0,
            "record_count": 0,
            "columns": []
        },
        "reports_directory": {
            "exists": os.path.exists(REPORTS_DIR),
            "path": os.path.abspath(REPORTS_DIR),
            "pdf_files": []
        },
        "processed_files_tracker": {
            "exists": os.path.exists(PROCESSED_FILES_TRACKER),
            "path": os.path.abspath(PROCESSED_FILES_TRACKER),
            "tracked_files": {}
        },
        "extractor_info": {
            "type": str(type(extractor)),
            "methods": [method for method in dir(extractor) if not method.startswith('_')],
            "has_process_report": hasattr(extractor, 'process_report'),
            "has_csv_fields": hasattr(extractor, 'csv_fields')
        }
    }
    
    # CSV details
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            debug_info["csv_file"]["record_count"] = len(df)
            debug_info["csv_file"]["columns"] = list(df.columns)
            if len(df) > 0:
                debug_info["csv_file"]["sample_record"] = df.iloc[0].to_dict()
        except Exception as e:
            debug_info["csv_file"]["error"] = str(e)
    
    # Reports directory
    if os.path.exists(REPORTS_DIR):
        try:
            pdf_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')]
            debug_info["reports_directory"]["pdf_files"] = pdf_files
            debug_info["reports_directory"]["pdf_count"] = len(pdf_files)
        except Exception as e:
            debug_info["reports_directory"]["error"] = str(e)
    
    # Processed files tracker
    debug_info["processed_files_tracker"]["tracked_files"] = load_processed_files_tracker()
    
    # Test extractor
    if hasattr(extractor, 'csv_fields'):
        try:
            debug_info["extractor_info"]["csv_fields"] = extractor.csv_fields
        except:
            debug_info["extractor_info"]["csv_fields"] = "Error accessing csv_fields"
    
    return debug_info

@app.post("/clear-processed-tracker/")
async def clear_processed_tracker():
    """
    Clear the processed files tracker to force reprocessing of all files
    """
    try:
        if os.path.exists(PROCESSED_FILES_TRACKER):
            os.remove(PROCESSED_FILES_TRACKER)
            print("Cleared processed files tracker")
        
        # Create empty tracker
        save_processed_files_tracker({})
        
        return {
            "message": "Processed files tracker cleared successfully",
            "tracker_path": os.path.abspath(PROCESSED_FILES_TRACKER)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing tracker: {str(e)}")

@app.post("/test-extractor/")
async def test_extractor(file: UploadFile = File(...)):
    """
    Test the extractor directly without any file tracking
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    # Save to temp location
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Testing extractor on: {temp_path}")
        
        # Test extractor
        result = extractor.process_report(temp_path)
        
        return {
            "message": "Extractor test completed",
            "filename": file.filename,
            "temp_path": temp_path,
            "result_type": str(type(result)),
            "result_length": len(result) if result else 0,
            "result_data": result[:2] if result and len(result) > 0 else None,  # First 2 records
            "extractor_type": str(type(extractor))
        }
        
    except Exception as e:
        return {
            "message": "Extractor test failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    print("Starting Honey Production Processing API...")
    print("API will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("Dashboard will be available at: http://localhost:8051")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)