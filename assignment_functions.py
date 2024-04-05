import pickle as pkl
import os
def compute_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    pkl.dump(model, open('models/'+type(model).__name__+'.pkl', 'wb'))   
    return score, y_pred


def compute_accuracy(y_true, y_pred):
    correct_predictions = 0
    for true, pred in zip(y_true, y_pred):
        y_pred = y_pred.round(2)
        pred_pos = pred + (pred * 0.2)
        pred_neg = pred - (pred * 0.2)
        if true == pred:
            correct_predictions += 1
        elif(true>=pred and true<=pred_pos):
            correct_predictions += 1
        elif(true<=pred and true>=pred_neg):
            correct_predictions += 1
    accuracy = correct_predictions/len(y_true)
    return accuracy


def encode_X_split(enc, X_train, X_test, y_train):
  X_train_enc = enc.fit_transform(X_train, y_train)
  X_test_enc = enc.transform(X_test)
  return X_train_enc, X_test_enc

def encode_X(enc, X, y):
  X_enc = enc.fit_transform(X, y)
  return X_enc


def load_pkl(fname, app):
    resource_path = os.path.join(app.root_path, fname)
    with open(resource_path, 'rb') as f:
        model = pkl.load(f)
    return model