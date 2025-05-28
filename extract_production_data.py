import os
import re
import pandas as pd
import fitz  # PyMuPDF
from datetime import datetime
import glob
import csv
import camelot
import numpy as np

class HoneyProductionExtractor:
    def __init__(self):
        # Define key field patterns to extract from honey production reports
        self.patterns = {
            'batch_number': r'Batch:\s*([A-Z0-9]+)',
            'report_title': r'SIDR\s+(\d{4})\s*-\s*([^-]+)',
            'company_name': r'([A-Z\s]+LLC)',
            'year': r'SIDR\s+(\d{4})',
            'report_type': r'APIARIES\s+HONEY\s+PRODUCTION\s+MAP',
        }
        
        # Define table headers that we expect to find
        self.table_headers = [
            'Apiary Number', 'Gross weight', 'Drum weight', 'Net weight', 
            'Beshara', 'Production', 'No. of Production Hives', 
            'Production per hive', 'No. of Hive Supers', 'Harvest Date'
        ]
        
        # UAE Emirates and locations for location extraction
        self.uae_locations = [
            "Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", 
            "Fujairah", "Ras Al Khaimah", "Kalba", "Masafi", "Hatta",
            "Khor Fakkan", "Al farfar", "UAQ", "RAK", "Dibba", "Taweeh"
        ]
        
        # Output CSV fields - added harvest_date
        self.csv_fields = [
            'batch_number', 'report_year', 'company_name', 'apiary_number', 
            'location', 'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg',
            'beshara_kg', 'production_kg', 'num_production_hives', 
            'production_per_hive_kg', 'num_hive_supers', 'harvest_date',
            'efficiency_ratio', 'waste_percentage', 'extraction_date'
        ]

    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from PDF using PyMuPDF"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def extract_header_info(self, text):
        """Extract header information from the report"""
        header_info = {}
        
        for field, pattern in self.patterns.items():
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if field == 'report_title':
                        # This pattern has 2 groups
                        if len(match.groups()) >= 2:
                            header_info['year'] = match.group(1)
                            header_info['title'] = match.group(2).strip()
                        else:
                            header_info[field] = match.group(0).strip()
                    elif field == 'report_type':
                        # This pattern has no capture groups, use group(0)
                        header_info[field] = match.group(0).strip()
                    else:
                        # For patterns with single capture group
                        if len(match.groups()) >= 1:
                            header_info[field] = match.group(1).strip()
                        else:
                            header_info[field] = match.group(0).strip()
                else:
                    header_info[field] = None
            except Exception as e:
                print(f"Error extracting {field}: {e}")
                header_info[field] = None
                
        return header_info

    def extract_location_from_text(self, text):
        """Extract location from apiary text"""
        for location in self.uae_locations:
            if location.lower() in text.lower():
                return location
        return "Unknown"

    def clean_numeric_value(self, value):
        """Clean and convert numeric values"""
        if pd.isna(value) or value == '' or value is None:
            return None
        
        # Convert to string if not already
        value_str = str(value).strip()
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.-]', '', value_str)
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def parse_harvest_date(self, date_str):
        """Parse harvest date from various formats"""
        if not date_str or pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Common date patterns found in the PDFs
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})',  # DD/MM/YY or DD/MM/YYYY
            r'(\d{1,2})-(\d{1,2})-(\d{2,4})',  # DD-MM-YY or DD-MM-YYYY
            r'(\d{4})/(\d{1,2})/(\d{1,2})',    # YYYY/MM/DD
            r'(\d{4})-(\d{1,2})-(\d{1,2})',    # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    if pattern.startswith(r'(\d{4})'):  # Year first format
                        year, month, day = match.groups()
                    else:  # Day first format (DD/MM/YY)
                        day, month, year = match.groups()
                    
                    # Convert 2-digit year to 4-digit
                    year = int(year)
                    if year < 100:
                        if year > 50:  # Assume 1950-1999
                            year += 1900
                        else:  # Assume 2000-2050
                            year += 2000
                    
                    # Create date object
                    parsed_date = datetime(year, int(month), int(day))
                    return parsed_date.strftime('%Y-%m-%d')
                    
                except (ValueError, TypeError) as e:
                    print(f"Error parsing date {date_str}: {e}")
                    continue
        
        return None

    def parse_data_line(self, line):
        """Parse a single data line from the production table"""
        # Clean the line
        line = line.strip()
        if not line:
            return None
            
        # Skip summary/total lines
        if any(word in line.lower() for word in ['net stock', 'net beshara', 'net production', 'total', 'average', 'cleaning']):
            return None
            
        # Split by whitespace but preserve apiary info
        parts = line.split()
        if len(parts) < 8:
            return None
            
        # Extract all numbers from the line
        numbers = re.findall(r'\d+\.?\d*', line)
        if len(numbers) < 8:
            return None
            
        # Extract date pattern from the end of the line
        harvest_date = None
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})(?:\s|$)', line)
        if date_match:
            harvest_date = self.parse_harvest_date(date_match.group(0))
        
        # Determine apiary number and location
        apiary_number = ""
        location = "Unknown"
        
        # Check if line starts with number (apiary ID)
        if re.match(r'^\d+', line):
            apiary_match = re.match(r'^(\d+\s*-\s*[A-Za-z\s]+)', line)
            if apiary_match:
                apiary_number = apiary_match.group(1).strip()
                location = self.extract_location_from_text(apiary_number)
        else:
            # Line might start with location name
            for loc in self.uae_locations:
                if line.lower().startswith(loc.lower()) or f"- {loc.lower()}" in line.lower():
                    apiary_number = f"- {loc}"
                    location = loc
                    break
        
        try:
            return {
                'apiary_number': apiary_number,
                'location': location,
                'gross_weight_kg': self.clean_numeric_value(numbers[0]),
                'drum_weight_kg': self.clean_numeric_value(numbers[1]),
                'net_weight_kg': self.clean_numeric_value(numbers[2]),
                'beshara_kg': self.clean_numeric_value(numbers[3]),
                'production_kg': self.clean_numeric_value(numbers[4]),
                'num_production_hives': self.clean_numeric_value(numbers[5]),
                'production_per_hive_kg': self.clean_numeric_value(numbers[6]),
                'num_hive_supers': self.clean_numeric_value(numbers[7]) if len(numbers) > 7 else None,
                'harvest_date': harvest_date
            }
        except Exception as e:
            print(f"Error parsing line: {line} - {e}")
            return None

    def extract_production_table(self, pdf_path):
        """Extract production data from tables in the PDF"""
        production_rows = []
        
        try:
            # Extract text from PDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Split into lines and process
            lines = text.split('\n')
            
            # Find the start of the data table
            data_started = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Skip header lines
                if any(header_word in line.lower() for header_word in ['apiary', 'gross', 'weight', 'production', 'harvest']):
                    if 'apiary' in line.lower() and 'weight' in line.lower():
                        data_started = True
                    continue
                
                # Skip summary lines
                if any(summary_word in line.lower() for summary_word in ['net stock', 'net beshara', 'net production', 'total', 'average', 'cleaning']):
                    continue
                    
                if data_started:
                    # Try to parse as data line
                    parsed_row = self.parse_data_line(line)
                    if parsed_row and parsed_row['apiary_number']:
                        production_rows.append(parsed_row)
            
            # If regex parsing didn't work well, try camelot as fallback
            if len(production_rows) < 5:  # Assuming we should have more than 5 rows
                try:
                    print("Trying camelot extraction as fallback...")
                    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                    if not tables:
                        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                    
                    for table in tables:
                        df_table = table.df
                        
                        # Process each row
                        for idx, row in df_table.iterrows():
                            if len(row.values) >= 8:
                                row_text = ' '.join([str(val) for val in row.values if str(val).strip()])
                                parsed_row = self.parse_data_line(row_text)
                                if parsed_row and parsed_row['apiary_number']:
                                    production_rows.append(parsed_row)
                                    
                except Exception as e:
                    print(f"Camelot extraction failed: {e}")
                    
        except Exception as e:
            print(f"Error extracting production table from {pdf_path}: {e}")
            
        return production_rows

    def calculate_additional_metrics(self, row):
        """Calculate additional metrics for each row"""
        # Calculate efficiency ratio (production per hive / net weight * 100)
        if row['production_per_hive_kg'] and row['net_weight_kg'] and row['net_weight_kg'] > 0:
            row['efficiency_ratio'] = (row['production_per_hive_kg'] / row['net_weight_kg']) * 100
        else:
            row['efficiency_ratio'] = None
            
        # Calculate waste percentage (beshara / gross weight * 100)
        if row['beshara_kg'] and row['gross_weight_kg'] and row['gross_weight_kg'] > 0:
            row['waste_percentage'] = (row['beshara_kg'] / row['gross_weight_kg']) * 100
        else:
            row['waste_percentage'] = None
            
        return row

    def process_report(self, pdf_path):
        """Process a single honey production report PDF"""
        results = []
        
        try:
            # Extract text and header information
            text = self.extract_text_from_pdf(pdf_path)
            header_info = self.extract_header_info(text)
            
            # Extract production data
            production_data = self.extract_production_table(pdf_path)
            
            # If no production data found, create a placeholder
            if not production_data:
                print(f"Warning: No production data found in {pdf_path}")
                production_data = [{
                    'apiary_number': 'Unknown',
                    'location': 'Unknown',
                    'gross_weight_kg': None,
                    'drum_weight_kg': None,
                    'net_weight_kg': None,
                    'beshara_kg': None,
                    'production_kg': None,
                    'num_production_hives': None,
                    'production_per_hive_kg': None,
                    'num_hive_supers': None,
                    'harvest_date': None
                }]
            
            # Create a row for each apiary
            for production in production_data:
                row = {
                    'batch_number': header_info.get('batch_number', 'Unknown'),
                    'report_year': header_info.get('year', datetime.now().year),
                    'company_name': header_info.get('company_name', 'MANAHIL LLC'),
                    'apiary_number': production.get('apiary_number'),
                    'location': production.get('location'),
                    'gross_weight_kg': production.get('gross_weight_kg'),
                    'drum_weight_kg': production.get('drum_weight_kg'),
                    'net_weight_kg': production.get('net_weight_kg'),
                    'beshara_kg': production.get('beshara_kg'),
                    'production_kg': production.get('production_kg'),
                    'num_production_hives': production.get('num_production_hives'),
                    'production_per_hive_kg': production.get('production_per_hive_kg'),
                    'num_hive_supers': production.get('num_hive_supers'),
                    'harvest_date': production.get('harvest_date'),
                    'extraction_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                # Calculate additional metrics
                row = self.calculate_additional_metrics(row)
                results.append(row)
            
            print(f"Successfully extracted {len(results)} records from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            # Return at least one row with basic info
            results = [{
                'batch_number': 'Error',
                'report_year': datetime.now().year,
                'company_name': 'Unknown',
                'apiary_number': 'Error',
                'location': 'Unknown',
                'gross_weight_kg': None,
                'drum_weight_kg': None,
                'net_weight_kg': None,
                'beshara_kg': None,
                'production_kg': None,
                'num_production_hives': None,
                'production_per_hive_kg': None,
                'num_hive_supers': None,
                'harvest_date': None,
                'efficiency_ratio': None,
                'waste_percentage': None,
                'extraction_date': datetime.now().strftime('%Y-%m-%d')
            }]
        
        return results

    def process_all_reports(self, directory):
        """Process all PDF reports in the specified directory"""
        all_results = []
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        
        total_files = len(pdf_files)
        print(f"Found {total_files} PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"Processing [{i+1}/{total_files}]: {os.path.basename(pdf_file)}")
            try:
                report_data = self.process_report(pdf_file)
                all_results.extend(report_data)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
                continue
            
        return all_results

    def save_to_csv(self, data, output_path):
        """Save extracted data to a CSV file"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
                writer.writeheader()
                for row in data:
                    # Ensure all fields are present
                    clean_row = {field: row.get(field, '') for field in self.csv_fields}
                    writer.writerow(clean_row)
            
            print(f"Data saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return None

    def create_dataframe(self, data):
        """Create a pandas DataFrame from the extracted data"""
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        numeric_cols = [
            'gross_weight_kg', 'drum_weight_kg', 'net_weight_kg', 'beshara_kg',
            'production_kg', 'num_production_hives', 'production_per_hive_kg',
            'num_hive_supers', 'efficiency_ratio', 'waste_percentage'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['extraction_date', 'harvest_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df

def main():
    # Set the directory containing honey production PDFs
    reports_dir = "honey_production_reports"
    output_csv = "honey_production_data.csv"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created directory: {reports_dir}")
        print("Please place your PDF files in this directory and run the script again.")
        return None
    
    # Create extractor and process reports
    extractor = HoneyProductionExtractor()
    all_data = extractor.process_all_reports(reports_dir)
    
    if not all_data:
        print("No data extracted. Please check your PDF files.")
        return None
    
    # Save to CSV
    extractor.save_to_csv(all_data, output_csv)
    
    # Create DataFrame
    df = extractor.create_dataframe(all_data)
    
    print(f"Successfully processed {len(df)} apiary records")
    print(f"Unique batches: {df['batch_number'].nunique()}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Date range: {df['harvest_date'].min()} to {df['harvest_date'].max()}")
    
    # Show DataFrame info
    print("\nDataFrame Summary:")
    print(df.info())
    
    # Show sample data
    print("\nSample Data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()