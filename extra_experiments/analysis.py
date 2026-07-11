import pandas as pd

for fname, label in [
    ('data/flows/Wednesday-WorkingHours.csv', 'Wednesday'),
    ('data/flows/Friday-WorkingHours.csv', 'Friday')]:
    df = pd.read_csv(fname, low_memory=False)
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].str.replace(' - Attempted', '', regex=False)
    mid = len(df) // 2
    train_part = df.iloc[:mid]
    test_part  = df.iloc[mid:]
    print(f'{label} — total: {len(df)}, train: {len(train_part)}, test: {len(test_part)}')
    print(f'  Train labels: {train_part["Label"].value_counts().to_dict()}')
    print(f'  Test labels:  {test_part["Label"].value_counts().to_dict()}')
    print()
