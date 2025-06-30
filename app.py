
from src.data_loader import ExcelLoader



def main():
    excel_path = "./test_data/Report_QNN-v2.36.0.250610144245_123137-auto_nightly_RC1_11_06_2025_14_01_48.xlsx"
    data = ExcelLoader().load(path = excel_path)



if __name__ == "__main__":
    main()