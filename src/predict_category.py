import joblib
import pandas as pd

MODEL_PATH = "model/product_category_model.pkl"

def main():
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
    print("Type 'exit' to stop.\n")

    while True:
        title = input("Enter product title: ")

        if title.lower() == "exit":
            print("Exiting...")
            break

        user_input = pd.DataFrame({"Product Title": [title]})

        prediction = model.predict(user_input["Product Title"])[0]
        print(f"Predicted category: {prediction}")
        print("-" * 40)


if __name__ == "__main__":
    main()