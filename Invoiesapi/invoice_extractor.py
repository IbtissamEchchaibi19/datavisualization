import os
import re
import pandas as pd
import fitz 
from datetime import datetime
import glob
import csv
import camelot

class InvoiceExtractor:
    def __init__(self):
        # Define key field patterns to extract from invoices
        self.patterns = {
            'invoice_id': r'Tax Invoice No:\s*([\w\d-]+)',
            'invoice_date': r'Date:\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})',
            'customer_name': r'Customer Name:\s*([^\n]+)',
            'customer_id': r'Customer ID:\s*([^\n]+)',
            'customer_address': r'Address:\s*([^\n]+(?:\n[^\n]+)*?)(?:United Arab Emirates|Tel:|Ref:|Customer Type:)',
            'customer_trn': r'Customer.*?TRN:\s*(\d+)',
            'customer_type': r'Customer Type:\s*([^\n]+)',
            'payment_status': r'Payment Status:\s*([^\n]+)',
            'due_date': r'Due Date:\s*(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})',
            'total_amount': r'Total with VAT.*?AED\s+([\d,]+\.\d{2})',
            'vat_amount': r'5% Total VAT.*?AED\s+([\d,]+\.\d{2})',
            'amount_excl_vat': r'Total Excluding VAT.*?AED\s+([\d,]+\.\d{2})',
            'profit': r'Profit:\s*AED\s+([\d,]+\.\d{2})',
            'profit_margin': r'Profit Margin:\s*([\d.]+)%',
            'cost_price': r'Cost Price:\s*AED\s+([\d,]+\.\d{2})',
        }
        
        # List of UAE Emirates for location extraction
        self.uae_emirates = [
            "Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", 
            "Fujairah", "Ras Al Khaimah", "UAQ", "RAK"
        ]
        
        # Keywords to identify tables with product information
        self.table_identifiers = ['Item Description', 'QTY', 'Unit Rate', 'Amount']
        
        # Output CSV fields
        self.csv_fields = [
            'invoice_id', 'invoice_date', 'customer_name', 'customer_id', 
            'customer_location', 'customer_type', 'customer_trn', 'payment_status', 
            'due_date', 'product', 'qty', 'unit_price', 'total', 'amount_excl_vat', 
            'vat', 'profit', 'profit_margin', 'cost_price', 'days_to_payment'
        ]

    def extract_text_from_pdf(self, pdf_path):
        """Extract all text from PDF using PyMuPDF (fitz)"""
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

    def extract_fields(self, text):
        """Extract fields from text using regex patterns"""
        fields = {}

        # Updated pattern for customer_address to support multiline and flexible end
        self.patterns['customer_address'] = r'Address:\s*((?:.|\n)*?)\s*United Arab Emirates'

        # Extract fields using regex
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                fields[field] = match.group(1).strip()
            else:
                fields[field] = None

        # Extract customer location from address
        fields['customer_location'] = "Unknown"
    
        if fields.get('customer_address'):
            for emirate in self.uae_emirates:
                pattern = r'\b' + re.escape(emirate) + r'\b'
                match = re.search(pattern, fields['customer_address'], re.IGNORECASE)
                if match:
                    fields['customer_location'] = match.group(0)
                    break
        
            # If no known emirate found, try "City, P.O. Box"
            if fields['customer_location'] == "Unknown":
                location_match = re.search(r'([A-Za-z\s]+),\s*P\.O\.\s*Box', fields['customer_address'])
                if location_match:
                    potential_location = location_match.group(1).strip()
                    if len(potential_location) > 3 and potential_location.lower() not in ["box", "p.o"]:
                        fields['customer_location'] = potential_location

        # Convert date strings to standard format
        for date_field in ['invoice_date', 'due_date']:
            if fields.get(date_field):
                try:
                    date_obj = datetime.strptime(fields[date_field], "%d %b %Y")
                    fields[date_field] = date_obj.strftime("%Y-%m-%d")
                except Exception:
                    pass

        # Clean numeric values
        for num_field in ['total_amount', 'vat_amount', 'amount_excl_vat', 'profit', 'cost_price']:
            if fields.get(num_field):
                fields[num_field] = fields[num_field].replace(',', '')

        return fields

    def extract_product_table(self, pdf_path):
        """Extract product information from tables in the PDF"""
        product_rows = []
        
        try:
            # Try with PyMuPDF first
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                tables = page.find_tables()
                
                for table in tables:
                    table_data = table.extract()
                    headers = [h.lower() if h else "" for h in table_data[0]]
                    
                    # Check if this is the product table
                    if any(id_text.lower() in " ".join(headers).lower() for id_text in self.table_identifiers):
                        # Find column indices
                        desc_idx = next((i for i, h in enumerate(headers) if 'description' in h.lower()), None)
                        qty_idx = next((i for i, h in enumerate(headers) if 'qty' in h.lower() or 'quantity' in h.lower()), None)
                        price_idx = next((i for i, h in enumerate(headers) if 'rate' in h.lower() or 'price' in h.lower()), None)
                        total_idx = next((i for i, h in enumerate(headers) if 'amount' in h.lower() and 'incl' in h.lower()), None)
                        
                        if not all([desc_idx is not None, qty_idx is not None]):
                            continue
                        
                        # Process data rows
                        for row in table_data[1:]:
                            if len(row) > max(filter(None, [desc_idx, qty_idx, price_idx, total_idx])):
                                product = row[desc_idx] if desc_idx is not None else ""
                                
                                # Skip headers or empty rows
                                if not product or product.lower() in ["item description", "total", ""]:
                                    continue
                                
                                qty = row[qty_idx] if qty_idx is not None else ""
                                unit_price = row[price_idx] if price_idx is not None else ""
                                total = row[total_idx] if total_idx is not None else ""
                                
                                # Clean numeric values
                                try:
                                    qty = float(qty.replace(',', '')) if qty else None
                                    unit_price = float(unit_price.replace(',', '')) if unit_price else None
                                    total = float(total.replace(',', '')) if total else None
                                except Exception:
                                    pass
                                
                                product_rows.append({
                                    'product': product,
                                    'qty': qty,
                                    'unit_price': unit_price,
                                    'total': total
                                })
            doc.close()
            
            # If no products found with PyMuPDF, try camelot
            if not product_rows:
                tables = camelot.read_pdf(pdf_path, pages='1-end', flavor='stream')
                for table in tables:
                    headers = [h.lower() for h in table.df.iloc[0]]
                    
                    # Check if this is the product table
                    if any(id_text.lower() in " ".join(headers).lower() for id_text in self.table_identifiers):
                        # Skip header row and process data
                        for _, row in table.df.iloc[1:].iterrows():
                            product = row[0] if not pd.isna(row[0]) else ""
                            
                            # Skip headers or empty rows
                            if not product or product.lower() in ["item description", "total", ""]:
                                continue
                                
                            # Try to find quantity and price columns
                            qty_col = next((i for i, h in enumerate(headers) if 'qty' in h or 'quantity' in h), 1)
                            price_col = next((i for i, h in enumerate(headers) if 'rate' in h or 'price' in h), 2)
                            total_col = next((i for i, h in enumerate(headers) if 'amount' in h and 'incl' in h), 3)
                            
                            qty = row[qty_col] if len(row) > qty_col else ""
                            unit_price = row[price_col] if len(row) > price_col else ""
                            total = row[total_col] if len(row) > total_col else ""
                            
                            # Clean numeric values
                            try:
                                qty = float(qty.replace(',', '')) if qty and not pd.isna(qty) else None
                                unit_price = float(unit_price.replace(',', '')) if unit_price and not pd.isna(unit_price) else None
                                total = float(total.replace(',', '')) if total and not pd.isna(total) else None
                            except Exception:
                                pass
                            
                            product_rows.append({
                                'product': product,
                                'qty': qty,
                                'unit_price': unit_price,
                                'total': total
                            })
            
            # If still no products found, try to extract using regex
            if not product_rows:
                text = self.extract_text_from_pdf(pdf_path)
                product_pattern = r'(\d+)\s+([^\n]+?)\s+(?:UN|EA)\s+(\d+)\s+([\d,.]+)\s+([\d,.]+)'
                matches = re.findall(product_pattern, text)
                
                for match in matches:
                    try:
                        product_rows.append({
                            'product': match[1].strip(),
                            'qty': float(match[2]),
                            'unit_price': float(match[3].replace(',', '')),
                            'total': float(match[4].replace(',', ''))
                        })
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"Error extracting product table from {pdf_path}: {e}")
            
        return product_rows

    def process_invoice(self, pdf_path):
        """Process a single invoice PDF and extract all relevant data"""
        results = []
        
        # Extract text and fields
        text = self.extract_text_from_pdf(pdf_path)
        fields = self.extract_fields(text)
        
        # Extract product information
        products = self.extract_product_table(pdf_path)
        
        # If no products found, create a dummy record to preserve invoice data
        if not products:
            products = [{'product': 'Unknown', 'qty': None, 'unit_price': None, 'total': fields.get('total_amount')}]
        
        # Create a row for each product
        for product in products:
            row = {
                'invoice_id': fields.get('invoice_id'),
                'invoice_date': fields.get('invoice_date'),
                'customer_name': fields.get('customer_name'),
                'customer_id': fields.get('customer_id'),
                'customer_location': fields.get('customer_location'),
                'customer_type': fields.get('customer_type'),
                'customer_trn': fields.get('customer_trn'),
                'payment_status': fields.get('payment_status'),
                'due_date': fields.get('due_date'),
                'product': product.get('product'),
                'qty': product.get('qty'),
                'unit_price': product.get('unit_price'),
                'total': product.get('total'),
                'amount_excl_vat': fields.get('amount_excl_vat'),
                'vat': fields.get('vat_amount'),
                'profit': fields.get('profit'),
                'profit_margin': fields.get('profit_margin'),
                'cost_price': fields.get('cost_price')
            }
            
            # Calculate days to payment (if paid)
            if fields.get('payment_status') == 'Paid' and fields.get('invoice_date') and fields.get('due_date'):
                try:
                    invoice_date = datetime.strptime(fields.get('invoice_date'), "%Y-%m-%d")
                    due_date = datetime.strptime(fields.get('due_date'), "%Y-%m-%d")
                    row['days_to_payment'] = (due_date - invoice_date).days
                except Exception:
                    row['days_to_payment'] = None
            else:
                row['days_to_payment'] = None
                
            results.append(row)
        
        return results

    def process_all_invoices(self, directory):
        """Process all PDF invoices in the specified directory"""
        all_results = []
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        
        total_files = len(pdf_files)
        print(f"Found {total_files} PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"Processing [{i+1}/{total_files}]: {os.path.basename(pdf_file)}")
            invoice_data = self.process_invoice(pdf_file)
            all_results.extend(invoice_data)
            
        return all_results

    def save_to_csv(self, data, output_path):
        """Save extracted data to a CSV file"""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fields)
            writer.writeheader()
            for row in data:
                # Ensure all fields are present
                clean_row = {field: row.get(field, '') for field in self.csv_fields}
                writer.writerow(clean_row)
        
        print(f"Data saved to {output_path}")
        return output_path

    def create_dataframe(self, data):
        """Create a pandas DataFrame from the extracted data"""
        df = pd.DataFrame(data)
        
        # Convert date columns to datetime
        for date_col in ['invoice_date', 'due_date']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['qty', 'unit_price', 'total', 'amount_excl_vat', 'vat', 
                        'profit', 'profit_margin', 'cost_price', 'days_to_payment']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

def main():
    # Set the directory containing invoice PDFs
    invoice_dir = "invoices"
    output_csv = "invoice_data.csv"
    
    # Create extractor and process invoices
    extractor = InvoiceExtractor()
    all_data = extractor.process_all_invoices(invoice_dir)
    
    # Save to CSV
    extractor.save_to_csv(all_data, output_csv)
    
    # Create DataFrame
    df = extractor.create_dataframe(all_data)
    
    print(f"Successfully processed {len(df)} invoice line items")
    print(f"Unique invoices: {df['invoice_id'].nunique()}")
    
    # Show DataFrame info
    print("\nDataFrame Summary:")
    print(df.info())
    
    # Show sample data
    print("\nSample Data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()