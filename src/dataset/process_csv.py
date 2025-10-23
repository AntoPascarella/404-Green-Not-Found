import csv
import re
import os

def clean_prompt(prompt):
    """
    Clean and format a prompt by:
    - Removing all backslashes
    - Removing all newlines and carriage returns
    - Normalizing whitespaces to single spaces
    - Creating a single-line text
    """
    # Remove carriage returns
    text = prompt.replace('\r', '')
    
    # Remove all newlines
    text = text.replace('\n', ' ')
    
    # Remove escaped quotes (backslash before quotes)
    text = text.replace('\\"', '"')
    
    # Remove all other backslashes
    text = text.replace('\\', '')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Ensure it starts with [ and ends with ]/
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']/'):
        if text.endswith(']'):
            text += '/'
        else:
            text += ']/'
    
    return text

def process_csv(csv_file_path, output_dir='corrected_prompts'):
    """
    Process CSV file with run data and save corrected prompts to individual text files.
    
    Args:
        csv_file_path: Path to the CSV file containing run data
        output_dir: Directory where corrected prompt files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process the CSV file
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Verify required columns exist
        required_fields = ['run_id', 'run_status', 'model', 'quantization_level', 
                          'task_type', 'row_id', 'prompt']
        
        if not all(field in reader.fieldnames for field in required_fields):
            missing = [f for f in required_fields if f not in reader.fieldnames]
            raise ValueError(f"CSV file is missing required fields: {missing}")
        
        processed_count = 0
        
        for row in reader:
            run_id = row['run_id']
            row_id = row['row_id']
            prompt = row['prompt']
            
            # Clean the prompt
            cleaned_prompt = clean_prompt(prompt)
            
            
            # Generate output filename
            run_id_clean = run_id.lstrip('_')
            output_filename = f"{run_id_clean}_{row_id}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write cleaned prompt to file
            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write(cleaned_prompt)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} rows...")
        
        print(f"\nCompleted! Processed {processed_count} total rows.")
        print(f"Corrected prompts saved to: {output_dir}/")

if __name__ == "__main__":
    
    csv_file = r"C:\Users\anton\Desktop\VU\GREEN LAB\runplan_qwen_int4.csv"
    output_directory = r"C:\Users\anton\Desktop\VU\GREEN LAB\qwen-int4_prompts"
    
    process_csv(csv_file, output_directory)
