from models.preprocessing import preprocess_data
from models.models import ANN
import matplotlib.pyplot as plt

def train_model():

    X_train, X_test, y_train, y_test = preprocess_data('/Users/mofeez/PycharmProjects/ANNsSample/dataset/Hotel_Reviews.csv')
    if X_train is None:
        return


    model = ANN(input_shape=X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)


    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")


    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model

if __name__ == "__main__":
    model = train_model()