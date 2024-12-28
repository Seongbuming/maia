import typing
from dotenv import load_dotenv
from utils.model import Model
from conversation.prompter import MAIAPrompter

load_dotenv()

def run(model_class: typing.Type[Model]):
    prompter = MAIAPrompter(model_class)
    prompter.reset()

    print("AI Assistant initialized. Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the conversation.")
                break

            response = prompter.prompt(user_input)
            print(f"AI: {response}")

        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    from models.chatgpt.core import ChatGPT
    run(ChatGPT)
