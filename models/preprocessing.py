from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dataset.dataset import HotelReviewDataset

def preprocess_data(filepath):
    dataset = HotelReviewDataset(filepath)
    df = dataset.get_data()
    if df is None:
        return None, None, None, None
    y = df['Reviewer_Score']
    X = df.drop('Reviewer_Score', axis=1)



    X['Hotel_Country'] = X['Hotel_Address'].apply(lambda x: x.split()[-1])
    le_country = LabelEncoder()
    X['Hotel_Country'] = le_country.fit_transform(X['Hotel_Country'])
    X = X.drop('Hotel_Address', axis=1)


    X = X.drop('Review_Date', axis=1)


    le_hotel = LabelEncoder()
    X['Hotel_Name'] = le_hotel.fit_transform(X['Hotel_Name'])


    le_nationality = LabelEncoder()
    X['Reviewer_Nationality'] = le_nationality.fit_transform(X['Reviewer_Nationality'])


    if 'Negative_Review' in X.columns and 'Positive_Review' in X.columns:
        X['No_Negative'] = X['Negative_Review'].apply(lambda x: 1 if 'No Negative' in x else 0)
        X['No_Positive'] = X['Positive_Review'].apply(lambda x: 1 if 'No Positive' in x else 0)
        X = X.drop(['Negative_Review', 'Positive_Review'], axis=1)
    else:
        print("Warning: Negative_Review and Positive_Review not found. Proceeding without them.")


    def parse_tags(tags_str):
        tags = eval(tags_str) if tags_str.startswith('[') else tags_str.split(',')
        return [tag.strip() for tag in tags]

    all_tags = X['Tags'].explode().value_counts()
    top_tags = all_tags.head(15).index
    for tag in top_tags:
        X[f'tag_{tag}'] = X['Tags'].apply(lambda x: 1 if tag in parse_tags(x) else 0)
    X = X.drop('Tags', axis=1)


    X['days_since_review'] = X['days_since_review'].apply(lambda x: int(x.split()[0]))


    X['lat'] = X['lat'].fillna(X['lat'].mean())
    X['lng'] = X['lng'].fillna(X['lng'].mean())


    numeric_cols = [
        'Additional_Number_of_Scoring', 'Average_Score', 'Review_Total_Negative_Word_Counts',
        'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
        'Total_Number_of_Reviews_Reviewer_Has_Given', 'days_since_review', 'lat', 'lng'
    ]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Preprocessing complete. X shape:", X.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data('/Users/mofeez/PycharmProjects/ANNsSample/dataset/Hotel_Reviews.csv')