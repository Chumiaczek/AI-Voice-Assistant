import model
from model.model_training import TrainingModel
import speech_recognition as sr

def read_voice_cmd(self):
    voice_input = ''
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.speech.listen(source=source, timeout=5, phrase_time_limit=5)
        voice_input = self.speech.recognize_google(audio)
        self.logger.info('Input: {}'.format(voice_input))
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

    while True:
        command = input("Enter your command: ")
        intents = training_model.get_intent(trained_model, command)
        response = TrainingModel.get_response(intents, model.data)
        print(intents, ' : ', response)