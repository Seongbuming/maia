import ask_sdk_core.utils as ask_utils
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = 'Hi.'
        return handler_input.response_builder.speak(speak_output).response

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
lambda_handler = sb.lambda_handler()
