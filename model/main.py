import model
from model.model_training import TrainingModel

if __name__ == '__main__':
    words = model.words
    classes = model.classes

    training_model = TrainingModel(words, classes, model.data_x, model.data_y)
    trained_model = training_model.train()

    while True:
        command = input("Enter your command: ")
        intents = training_model.get_intent(trained_model, command)
        response = TrainingModel.get_response(intents, model.data)
        print(intents, ' : ', response)