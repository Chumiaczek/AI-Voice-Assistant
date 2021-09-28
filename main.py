import model
from model.model_training import TrainingModel
import speech_recognition as sr

def read_voice_cmd(speech):
    voice_input = ''
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = speech.listen(source=source, timeout=5, phrase_time_limit=5)
        voice_input = speech.recognize_google(audio)
        print('Input: {}'.format(voice_input))
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        print("Check your connection with Internet and try again.")
    except sr.WaitTimeoutError:
        pass
    except TimeoutError:
        pass

    return voice_input.lower()

if __name__ == '__main__':
    words = model.words
    classes = model.classes

    training_model = TrainingModel(words, classes, model.data_x, model.data_y)
    trained_model = training_model.train()

    speech = sr.Recognizer()

    while True:
        command = read_voice_cmd(speech)
        intents = training_model.get_intent(trained_model, command)
        response = TrainingModel.get_response(intents, model.data)
        print(intents, ' : ', response)