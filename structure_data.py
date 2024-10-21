import pandas as pd
from io import StringIO
import json

# Open the CSV file in binary mode and handle encoding errors
with open('D:\\AI\\code\\raw_data.csv', 'rb') as f:

# with open('raw_data.csv', 'rb') as f:
    content = f.read().decode('Big5', errors='replace')  # 'replace' will replace invalid characters with �

# Convert the decoded content back into a file-like object to read with pandas
df = pd.read_csv(StringIO(content))

# Define a function to generate the sentence from each row
def generate_sentence(row):
    age = row['AGE']
    sex = 'male' if row['SEX'] == 'M' else 'female'
    bp_systolic = row['BPS']
    bp_diastolic = row['BPB']
    hr = row['PULSE']
    rr = row['RR']
    temp = row['TMP']
    shock_index = row['shock_index']
    gcs_e = row['GCSE']
    gcs_v = row['GCSV']
    gcs_m = row['GCSM']
    chief_complaint = row['CHIFC1']
    cardiac_arrest = row['cardiac_arrest'] * 100
    icu_transfer = row['ICU_transfer'] * 100
    inotropic_support = row['Inotropic'] * 100
    ventilation = row['Vent'] * 100
    input_text = f"This is a {age} year old, {sex} patient, presented to the ER with vital signs of BP {bp_systolic}/{bp_diastolic}, HR: {hr}bpm, RR: {rr}/min, BT: {temp} degrees, Shock Index: {shock_index} and GCS: E{gcs_e}V{gcs_v}M{gcs_m}    [Chief Complaint]: {chief_complaint}"

    input_text1 = (f"This is a {age} year old, {sex} patient, presented to the ER with vital signs of BP {bp_systolic}/{bp_diastolic}, "
                  f"HR: {hr}bpm, RR: {rr}/min, BT: {temp} degrees, Shock Index: {shock_index} and GCS: E{gcs_e}V{gcs_v}M{gcs_m}    "
                  f"[Chief Complaint]: {chief_complaint}\n\n[病史詢問來源]: pt's mother\n\n[過去及個人病史]:\n1.Major depressive disorder, recurrent, mild\n"
                  f"2.Hepatitis C\n3.Suspect cluster B personality disorder\n4.Heroin use disorder\n5.Asthma\n6.Benzodiazepines use history\n"
                  f"no alcohol or drug abuse, no smoking\n\nOutput: Cardiac Arrest={cardiac_arrest}%, ICU transfer={icu_transfer}%, "
                  f"Inotropic Support={inotropic_support}%, Mechanical Ventilation={ventilation}%")
    
    return {
        "Input": input_text,
        "Output": f"Cardiac Arrest={cardiac_arrest}%, ICU transfer={icu_transfer}%, Inotropic Support={inotropic_support}%, Mechanical Ventilation={ventilation}%"
    }

# Create a list to store the results
results = []

# Apply the function to each row in the DataFrame
for index, row in df.iterrows():
    results.append(generate_sentence(row))

# Save the results to a JSON file
with open('structured_raw_data.json', 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)

print("Data has been successfully saved to structured_raw_data.json")
