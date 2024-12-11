import pandas as pd
import os
from pathlib import Path

def process_excel_files():
    # Get all .xlsx files in the current directory
    current_dir = Path(".")
    excel_files = list(current_dir.glob("*.xlsx"))

    if not excel_files:
        print("No .xlsx files found in the current directory.")
        return

    for file in excel_files:
        # Read the Excel file
        data = pd.read_excel(file)

        # Process the 'TicketNo' column
        expanded_data = data.set_index('Pnr')['TicketNo'].str.split('|', expand=True).stack().reset_index()

        # Rename the columns
        expanded_data.columns = ['Pnr', 'Index', 'TicketNo']

        # Remove quotes from 'TicketNo' and 'Pnr' columns
        expanded_data['TicketNo'] = expanded_data['TicketNo'].str.strip('"')
        expanded_data['Pnr'] = expanded_data['Pnr'].str.strip('"')

        # Drop the unnecessary 'Index' column
        expanded_data.drop(columns=['Index'], inplace=True)

        # Save the processed data to a new file
        output_file = file.stem + "_modified.xlsx"
        expanded_data.to_excel(output_file, index=False)

        print(f"Processed: {file.name} -> {output_file}")

if __name__ == "__main__":
    process_excel_files()
