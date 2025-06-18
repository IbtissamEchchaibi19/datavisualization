from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional, Union
import pandas as pd
import uvicorn
import threading
import time
import json
import hashlib
import traceback

# Import your existing classes
from extract_production_data import HoneyProductionExtractor  # Your existing extractor

# DASHBOARD IMPORT - UNCHANGED
try:
    from dashboard import app as dash_app
    DASHBOARD_AVAILABLE = True
    print("✓ Dashboard app imported successfully")
except ImportError as e:
    print(f"✗ Dashboard import error: {e}")
    print("Creating a dummy dashboard app...")
    
    try:
        import dash
        from dash import html
        
        dash_app = dash.Dash(__name__)
        dash_app.layout = html.Div([
            html.H1("Dashboard Not Available"),
            html.P("The dashboard module could not be imported.")
        ])
        DASHBOARD_AVAILABLE = False
    except:
        dash_app = None
        DASHBOARD_AVAILABLE = False
except Exception as e:
    print(f"✗ Dashboard import failed: {e}")
    dash_app = None
    DASHBOARD_AVAILABLE = False

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
    report_ids: List[str]

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

# Global variables for dashboard
dash_thread = None

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# FIXED: Define the exact CSV column order from your extractor
CSV_COLUMNS = [
    'batch_number', 'report_year', 'company_name', 'apiary_number', 
    'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
    'beshara_kg', 'production_kg', 'num_production_hives', 
    'production_per_hive_kg', 'num_hive_supers', 'harvest_date',
    'efficiency_ratio', 'waste_percentage', 'extraction_date'
]

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
            return False
        
        current_hash = get_file_hash(file_path)
        if not current_hash:
            return False
            
        return tracker[filename] == current_hash
        
    except Exception as e:
        print(f"Error checking if file processed {file_path}: {e}")
        return False

def mark_file_as_processed(file_path: str):
    """Mark a file as processed in the tracker"""
    try:
        if not os.path.exists(file_path):
            return
            
        tracker = load_processed_files_tracker()
        filename = os.path.basename(file_path)
        file_hash = get_file_hash(file_path)
        
        if file_hash:
            tracker[filename] = file_hash
            save_processed_files_tracker(tracker)
            print(f"Marked {filename} as processed")
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

def initialize_csv_if_needed():
    """FIXED: Initialize CSV file with proper column structure"""
    if not os.path.exists(CSV_FILE):
        try:
            # Create empty DataFrame with ALL expected columns in correct order
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(CSV_FILE, index=False)
            print(f"Initialized CSV file with {len(CSV_COLUMNS)} columns")
        except Exception as e:
            print(f"Error initializing CSV: {e}")

def ensure_data_consistency(data_dict: dict) -> dict:
    """FIXED: Ensure all required columns are present with proper default values"""
    consistent_data = {}
    
    for column in CSV_COLUMNS:
        if column in data_dict and data_dict[column] is not None:
            # Convert to appropriate type and handle NaN
            value = data_dict[column]
            if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                consistent_data[column] = None
            else:
                consistent_data[column] = value
        else:
            # Set appropriate default for missing columns
            if column in ['batch_number', 'company_name', 'apiary_number', 'location']:
                consistent_data[column] = 'Unknown'
            elif column == 'report_year':
                consistent_data[column] = 2024
            elif column == 'extraction_date':
                consistent_data[column] = pd.Timestamp.now().strftime('%Y-%m-%d')
            else:
                consistent_data[column] = None
    
    return consistent_data

def append_to_csv(new_data: List[dict]):
    """FIXED: Append new data with proper column alignment and validation"""
    if not new_data:
        return
    
    try:
        print(f"DEBUG: Appending {len(new_data)} records to CSV")
        
        # Ensure data consistency for all records
        consistent_data = []
        for record in new_data:
            consistent_record = ensure_data_consistency(record)
            consistent_data.append(consistent_record)
            print(f"DEBUG: Processed record - batch: {consistent_record.get('batch_number')}, location: {consistent_record.get('location')}, production: {consistent_record.get('production_kg')}")
        
        # Create DataFrame with proper column order
        new_df = pd.DataFrame(consistent_data, columns=CSV_COLUMNS)
        
        # Debug: Check for NaN values before saving
        nan_columns = new_df.columns[new_df.isna().any()].tolist()
        if nan_columns:
            print(f"WARNING: Found NaN values in columns: {nan_columns}")
        
        if os.path.exists(CSV_FILE):
            # Read existing CSV to ensure column compatibility
            try:
                existing_df = pd.read_csv(CSV_FILE)
                print(f"DEBUG: Existing CSV has {len(existing_df)} records with columns: {list(existing_df.columns)}")
                
                # Check if columns match
                if list(existing_df.columns) != CSV_COLUMNS:
                    print("WARNING: Column mismatch detected - rewriting CSV structure")
                    # Rewrite entire CSV with correct structure
                    combined_df = pd.concat([existing_df.reindex(columns=CSV_COLUMNS), new_df], ignore_index=True)
                    combined_df.to_csv(CSV_FILE, index=False)
                else:
                    # Append normally
                    new_df.to_csv(CSV_FILE, mode='a', header=False, index=False)
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
                # If can't read existing, write new
                new_df.to_csv(CSV_FILE, index=False)
        else:
            # Create new CSV with headers
            new_df.to_csv(CSV_FILE, index=False)
        
        print(f"SUCCESS: Added {len(new_data)} records to CSV")
        
        # Verify the data was written correctly
        try:
            verification_df = pd.read_csv(CSV_FILE)
            print(f"VERIFICATION: CSV now has {len(verification_df)} total records")
            
            # Check last few records for NaN issues
            last_records = verification_df.tail(len(new_data))
            nan_check = last_records.isna().sum()
            if nan_check.sum() > 0:
                print(f"WARNING: NaN values found in newly added records:")
                for col, count in nan_check.items():
                    if count > 0:
                        print(f"  {col}: {count} NaN values")
            else:
                print("SUCCESS: No NaN values in newly added records")
                
        except Exception as e:
            print(f"Error verifying CSV: {e}")
            
    except Exception as e:
        print(f"ERROR: Failed to append to CSV: {e}")
        traceback.print_exc()
        raise

def get_report_records_by_ids(report_ids: List[str]) -> tuple:
    """Get report records by their IDs and return found/not found lists"""
    if not os.path.exists(CSV_FILE):
        return [], report_ids
    
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty or 'batch_number' not in df.columns:
            return [], report_ids
        
        # Convert report_ids to strings for comparison
        report_ids_str = [str(id) for id in report_ids]
        
        # Find matching records
        matching_records = df[df['batch_number'].astype(str).isin(report_ids_str)]
        found_batch_numbers = matching_records['batch_number'].astype(str).unique().tolist()
        
        # Determine which IDs were not found
        not_found_ids = [id for id in report_ids_str if id not in found_batch_numbers]
        
        # Convert matching records to list of dictionaries
        found_records = matching_records.to_dict('records')
        
        return found_records, not_found_ids
        
    except Exception as e:
        print(f"Error getting report records: {e}")
        return [], report_ids

def delete_records_from_csv(report_ids: List[str]) -> int:
    """Delete records from CSV by batch numbers"""
    if not os.path.exists(CSV_FILE):
        return 0
    
    try:
        df = pd.read_csv(CSV_FILE)
        original_count = len(df)
        
        # Filter out records with matching batch numbers
        df_filtered = df[~df['batch_number'].astype(str).isin([str(id) for id in report_ids])]
        deleted_count = original_count - len(df_filtered)
        
        # Save the updated CSV
        df_filtered.to_csv(CSV_FILE, index=False)
        print(f"Deleted {deleted_count} records from CSV")
        
        return deleted_count
        
    except Exception as e:
        print(f"Error deleting records from CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating CSV: {str(e)}")

def delete_pdf_files(filenames: List[str]) -> List[str]:
    """Delete PDF files from the reports directory"""
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

# DASHBOARD FUNCTIONS - UNCHANGED
def run_dash_app():
    """Run Dash app in a separate thread"""
    try:
        if dash_app is not None:
            dash_app.run(host='0.0.0.0', port=8051, debug=False)
        else:
            print("Dash app is None - cannot start")
    except Exception as e:
        print(f"Error running Dash app: {e}")

def start_dash_app():
    """Start Dash app in background thread"""
    global dash_thread
    
    if not DASHBOARD_AVAILABLE or dash_app is None:
        print("Dashboard not available - skipping start")
        return
        
    if dash_thread is None or not dash_thread.is_alive():
        dash_thread = threading.Thread(target=run_dash_app, daemon=True)
        dash_thread.start()
        time.sleep(2)  # Wait for dashboard to start
        print("Dash app started on port 8051")
    else:
        print("Dash app already running")

# STARTUP EVENT
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("Starting Honey Production Processing API...")
    start_dash_app()
    initialize_csv_if_needed()

# ROUTES - UNCHANGED EXCEPT UPLOAD
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
        "message": "Dashboard is running" if DASHBOARD_AVAILABLE else "Dashboard not available",
        "iframe_endpoint": "/dashboard-iframe/",
        "dashboard_available": DASHBOARD_AVAILABLE
    }

@app.get("/dashboard-iframe/")
async def dashboard_iframe():
    """Serve dashboard in iframe"""
    if not DASHBOARD_AVAILABLE:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Dashboard Not Available</h1>
                <p>The dashboard module could not be imported.</p>
            </body>
        </html>
        """)
    
    return HTMLResponse("""
    <html>
        <head>
            <title>Production Dashboard</title>
            <style>
                body { margin: 0; padding: 0; }
                iframe { width: 100%; height: 100vh; border: none; }
            </style>
        </head>
        <body>
            <iframe src="http://localhost:8051/"></iframe>
        </body>
    </html>
    """)

@app.delete("/delete-reports/")
async def delete_reports(request: DeleteReportRequest):
    """Delete one or multiple production reports by their batch numbers"""
    if not request.report_ids:
        raise HTTPException(status_code=400, detail="No report IDs provided")
    
    try:
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
            batch_number = record.get('batch_number', 'unknown')
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
        raise HTTPException(status_code=500, detail=f"Error deleting reports: {str(e)}")

@app.get("/reports/")
async def list_reports():
    """List all production reports with their batch numbers for reference"""
    if not os.path.exists(CSV_FILE):
        return {
            "message": "No production report data found",
            "reports": [],
            "total_count": 0
        }
    
    try:
        df = pd.read_csv(CSV_FILE)
        
        if df.empty:
            return {
                "message": "CSV file exists but contains no data",
                "reports": [],
                "total_count": 0
            }
        
        # Group by batch_number to get unique reports
        if 'batch_number' not in df.columns:
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
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

@app.post("/upload-reports/")
async def upload_reports(files: Union[List[UploadFile], UploadFile] = File(...)):
    """FIXED: Upload and process honey production report PDFs with proper data handling"""
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
        
        file_path = os.path.join(REPORTS_DIR, file.filename)
        
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
            try:
                report_data = extractor.process_report(file_path)
                
                if report_data and len(report_data) > 0:
                    print(f"DEBUG: Extractor returned {len(report_data)} records for {file.filename}")
                    
                    # FIXED: Validate and clean the data before adding
                    valid_records = []
                    for record in report_data:
                        # Debug the record structure
                        print(f"DEBUG: Raw record: {record}")
                        
                        # Ensure the record has all required fields
                        clean_record = ensure_data_consistency(record)
                        valid_records.append(clean_record)
                        
                        print(f"DEBUG: Cleaned record - batch: {clean_record.get('batch_number')}, production: {clean_record.get('production_kg')}")
                    
                    all_new_data.extend(valid_records)
                    mark_file_as_processed(file_path)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "records_added": len(valid_records)
                    })
                    total_new_records += len(valid_records)
                    print(f"SUCCESS: Processed {file.filename}: {len(valid_records)} valid records")
                else:
                    errors.append(f"{file.filename}: No data extracted from PDF")
                    print(f"WARNING: No data extracted from {file.filename}")
                    
            except Exception as e:
                error_msg = f"{file.filename}: Error processing PDF - {str(e)}"
                errors.append(error_msg)
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                
        except Exception as e:
            error_msg = f"{file.filename}: Error saving file - {str(e)}"
            errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            
            # Clean up file on error
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    # FIXED: Append all new data to CSV with proper validation
    if all_new_data:
        try:
            print(f"DEBUG: About to append {len(all_new_data)} records to CSV")
            append_to_csv(all_new_data)
            print(f"SUCCESS: Added {len(all_new_data)} total records to CSV")
        except Exception as e:
            error_msg = f"Error saving data to CSV: {str(e)}"
            print(f"ERROR: {error_msg}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_msg)
    
    # Get current total records
    total_records = 0
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            total_records = len(df)
            print(f"DEBUG: CSV now contains {total_records} total records")
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
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict('records')
        
        return {
            "message": f"Retrieved {len(data)} records",
            "data": data,
            "total_records": len(data),
            "unique_batches": df['batch_number'].nunique() if 'batch_number' in df.columns else 0
        }
    except Exception as e:
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
        try:
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                csv_records = len(df)
        except Exception as e:
            print(f"Warning: Could not count CSV records: {e}")
        
        return {
            "status": "healthy",
            "message": "Honey Production Processing API is running",
            "dashboard_status": dashboard_status,
            "dashboard_available": DASHBOARD_AVAILABLE,
            "csv_exists": os.path.exists(CSV_FILE),
            "csv_records": csv_records,
            "reports_directory_exists": os.path.exists(REPORTS_DIR),
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
    print("Starting Honey Production Processing API...")
    print("API will be available at: http://localhost:8001")
    print("API Documentation: http://localhost:8001/docs")
    print("Dashboard will be available at: http://localhost:8051")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)