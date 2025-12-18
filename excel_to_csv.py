import pandas as pd
import os

excel_path = os.path.join('data', 'data.xlsx')
csv_path = os.path.join('data', 'data.csv')

def excel_to_csv(excel_path, csv_path):
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False)
    print(f'转换完成: {csv_path}')

if __name__ == '__main__':
    excel_to_csv(excel_path, csv_path)
