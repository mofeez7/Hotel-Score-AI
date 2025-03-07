import pandas as pd

class HotelReviewDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.filepath)
            print("Available columns:", self.df.columns.tolist())
            return self.df
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            return None

    def get_data(self):
        if self.df is None:
            self.load_data()
        return self.df

if __name__ == "__main__":
    dataset = HotelReviewDataset('/dataset/Hotel_Reviews.csv')
    df = dataset.load_data()