from predict import predict_email

print("Spam Detection System")
print("---------------------")

while True:
    msg = input("\nEnter email (or quit): ")
    if msg.lower() == "quit":
        break

    print("Prediction:", predict_email(msg))
