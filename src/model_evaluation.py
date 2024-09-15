from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Not_Fraud", "Fraud"]))
    cm = confusion_matrix(y_test, y_pred)
    # Unpack using ravel
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp
