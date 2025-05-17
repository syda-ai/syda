import sys
sys.path.insert(0, '.')  # Ensure local imports work if running from root

from syda.structured import SyntheticDataGenerator
import pandas as pd

# Example: Hybrid synthetic data generation test script
import os
from dotenv import load_dotenv
load_dotenv()

from syda.structured import SyntheticDataGenerator
import pandas as pd
import random

# Example custom generator for a healthcare code
def custom_icd10_code(row, col_name):
    return random.choice(['A00', 'B01', 'C02', 'D03'])

# Schema with a custom type
schema = {
    'patient_id': 'number',
    'diagnosis_code': 'icd10_code',
    'email': 'email',
    'visit_date': 'date',
    'notes': 'text'
}
prompt = "Generate realistic synthetic patient records with ICD-10 diagnosis codes, emails, visit dates, and clinical notes."

def test_hybrid_generate_data():
    generator = SyntheticDataGenerator()
    generator.register_generator('icd10_code', custom_icd10_code)
    output_path = 'synthetic_output.csv'
    result = generator.generate_data(
        schema_dict=schema, prompt=prompt, sample_size=15, output_path=output_path)
    print(f'Generated hybrid synthetic data written to: {result}')
    assert result == output_path
    df = pd.read_csv(output_path)
    print(df)
    assert len(df) == 15
    assert set(df.columns) == set(schema.keys())
    # Check that only allowed ICD-10 codes appear
    assert set(df['diagnosis_code']).issubset({'A00', 'B01', 'C02', 'D03'})

if __name__ == '__main__':
    print('--- Testing hybrid generate_data with custom generator ---')
    test_hybrid_generate_data()
