import gradio as gr
import logging
import random
import json
import os
import os.path as osp
import copy
from utils.form import PairwiseForm
from utils.pipeline import PairwisePipeline

class ConversationForm(PairwiseForm):
    def __init__(
        self,
        model: PairwisePipeline,
        title: str,
    ):
        # Excluded 1 turn before and after
        # minimum = 1
        self.turns = 3
        self.skip_btn_visible=True

        self.scales = {
            "likert": [
                ("Strongly disagree", 1),
                ("Disagree", 2),
                ("Neither agree nor disagree", 3),
                ("Agree", 4),
                ("Strongly agree", 5),
            ],
            "comparison": [
                ("Model 1", 1),
                ("Model 2", 2),
            ],
        }
        self.situations = self.__load_scenario("hit")
        self.guidance = {
            "informed_consent": self.__load_guidance("informed_consent"),
            "experiment_description": self.__load_experiment_description("experiment_description", self.situations),
            "system_guide": self.__load_guidance("system_guide"),
            "system_usage_instruction": self.__load_guidance("system_usage_instruction"),
            "end_of_tasks": self.__load_guidance("end_of_tasks"),
        }
        self.user_temp = {}
        self.user_data = {}
        self.user_model = {}
        # self.data = {
        #     "mturk_worker_id": "N/A",
        #     "result": {
        #         "situation1": list(),
        #         "situation2": list(),
        #         "situation3": list(),
        #         "freetalk": list(),
        #         "last_answer": -1,          # -1 means None(=null)
        #         "usability_answer": list()
        #     }
        # }
        self.data_path = "results"
        self.text_input_hint="If the microphone malfunctions, use text input."
        self.evaluation_check_msg = "Please complete all survey questions."

        super().__init__(model=model, title=title)
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        self.file_handler = logging.FileHandler("hit.log")
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
    
    def __init_user_data(self, mturk_worker_id: str) -> None:
        self.user_temp[mturk_worker_id] = {
            "situation_idx": 0,
            "scenario_count": 0,
        }
        self.user_data[mturk_worker_id] = {
            "mturk_worker_id": mturk_worker_id,
            "result": {
                "situation1": list(),
                "situation2": list(),
                "situation3": list(),
                "freetalk": list(),
                "last_answer": -1,          # -1 means None(=null)
                "usability_answer": list(),
            },
        }
        user_model = copy.deepcopy(self.model)
        user_model.generate_model_1.reset()
        user_model.generate_model_2.reset()
        self.user_model[mturk_worker_id] = user_model

    def _create_form(self) -> gr.Blocks:
        with gr.Blocks(css="#chatbot") as form:
            gr.HTML(f"<h1 style=\"text-align: center;\">{self.title}</h1>")

            with gr.Tab("IRB Agreement"):
                irb_msg = gr.Markdown(self.guidance['informed_consent'])
                informed_consent = gr.Radio(
                    show_label=False,
                    choices=[("Agree", 1), ("Disagree", 0)],
                    container=False
                )
                irb_button = gr.Button("Save IRB Agreement")
            with gr.Tab("System Guide"):
                with gr.Column():
                    gr.Markdown(self.guidance['experiment_description'])
                    gr.Markdown(self.guidance['system_guide'])
            with gr.Tab("System"):
                with gr.Row():
                    with gr.Column(scale=0.8):
                        id_input = gr.Textbox(
                            label="MTurk Worker ID",
                            show_label=True
                        )
                    with gr.Column(scale=0.2, min_width=0):
                        save_id_button = gr.Button("Save MTurk Worker ID")
                skip_button = gr.Button("Skip situations",
                                        visible=False)
                with gr.Column():
                    situation_title = gr.Markdown(
                        "## System Usage Instruction",
                        visible=False
                    )
                    situation_description = gr.Markdown(
                        self.__get_current_scenario(None),
                        visible=False
                    )
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    visible=False
                )

                with gr.Row(visible=False) as ques_row1:
                    question1 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 1] Is the response make sence?",
                        show_label=True,
                    )
                    question4 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 2] Is the response make sense?",
                        show_label=True,
                    )
                with gr.Row(visible=False) as ques_row2:
                    question2 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 1] Is the context of the response consistent?",
                        show_label=True,
                    )
                    question5 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 2] Is the context of the response consistent?",
                        show_label=True,
                    )
                with gr.Row(visible=False) as ques_row3:
                    question3 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 1] Are you interested in the response? Would you like to continue the conversation?",
                        show_label=True,
                    )
                    question6 = gr.Radio(
                        scale=0.5,
                        choices=self.scales["likert"],
                        label="[Model 2] Are you interested in the response? Would you like to continue the conversation?",
                        show_label=True,
                    )
                with gr.Row(visible=False) as pair_row:
                    pairwise_question1 = gr.Radio(
                        choices=self.scales["comparison"],
                        label="Which response makes more sense?",
                        show_label=True,
                    )
                    pairwise_question2 = gr.Radio(
                        choices=self.scales["comparison"],
                        label="Which response is more consistent?",
                        show_label=True,
                    )
                    pairwise_question3 = gr.Radio(
                        choices=self.scales["comparison"],
                        label="Which response feels more personalized?",
                        show_label=True,
                    )
                    pairwise_question4 = gr.Radio(
                        choices=self.scales["comparison"],
                        label="Who do you prefer to talk to more?",
                        show_label=True,
                    )
                with gr.Column(visible=False) as input_column:
                    audio_record = gr.Audio(
                        show_label=False,
                        source="microphone",
                        type="filepath",
                    )
                    text_input = gr.Textbox(
                        show_label=False,
                        placeholder=self.text_input_hint
                    )
                    text_input.style(container=False)
                with gr.Row(visible=False) as btn_row:
                    reset_button = gr.Button(
                        "Reset this conversation",
                    )
                    continue_button = gr.Button(
                        "Continue this conversation",
                    )
                finish_message = gr.Markdown(
                        "### Navigate to the Usability Evaluation tab.",
                        visible=False,
                        show_label=False,
                    )
                finish_button = gr.Button(
                    "Finish this conversation",
                    visible=False
                )
                with gr.Row(visible=False) as last_row:
                    with gr.Column(scale=0.7):
                        last_question = gr.Radio(
                            choices=self.scales["likert"],
                            label="How was the conversation?",
                            show_label=True,
                        )
                    with gr.Column(scale=0.3, min_width=0):
                        submit_button = gr.Button(
                            "Submit",
                        )
            with gr.Tab("Usability Evaluation"):
                with gr.Column(visible=True) as usability_message_ui:
                    usability_message = gr.Markdown("### Please use the \"System\" first.")
                with gr.Column(visible=False) as usability_row:
                    gr.Markdown("## Please evaluate the usability of the free talk system you used last.")
                    usability_question1 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I think that I would like to use this system frequently.",
                        show_label=True
                    )
                    usability_question2 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I found the system unnecessarily complex.",
                        show_label=True,
                    )
                    usability_question3 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I thought the system was easy to use.",
                        show_label=True
                    )
                    usability_question4 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I think that I would need the support of a technical person to be able to use this system.",
                        show_label=True
                    )
                    usability_question5 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I found the various functions in this system were well integrated.",
                        show_label=True
                    )
                    usability_question6 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I thought there was too much inconsistency in this system.",
                        show_label=True
                    )
                    usability_question7 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I would imagine that most people would learn to use this system very quickly.",
                        show_label=True
                    )
                    usability_question8 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I found the system very cumbersome to use.",
                        show_label=True
                    )
                    usability_question9 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I felt very confident using the system.",
                        show_label=True
                    )
                    usability_question10 = gr.Radio(
                        choices=self.scales["likert"],
                        label="I needed to learn a lot of things before I could get going with this system.",
                        show_label=True
                    )
                    usability_button = gr.Button("Submit")
            
            # IRB Agreement save button
            irb_button.click(
                self.__save_irb_agreement,
                inputs=[informed_consent, irb_button, irb_msg],
                outputs=[informed_consent, irb_button, irb_msg],
                queue=False,
            )
            
            # id save button
            save_id_button.click(
                self.__save_id,
                inputs=[id_input, save_id_button, skip_button,
                        situation_title, situation_description, chatbot],
                outputs=[id_input, save_id_button, skip_button,
                         situation_title, situation_description, chatbot, input_column],
                queue=False,
            )
            
            # skip button
            skip_button.click(
                self.__skip_situations,
                inputs=[id_input],
                outputs=[situation_description,
                         chatbot, ques_row1, ques_row2, ques_row3,
                         pair_row, btn_row, finish_button, input_column,
                         audio_record, text_input, last_row, skip_button,
                         question1, question2, question3,
                         question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True
            )
            
            # audio recording component
            audio_record.stop_recording(
                self.__add_record,
                inputs=[chatbot, audio_record],
                outputs=[chatbot, audio_record],
                queue=False,
            ).then(
                lambda: gr.update(visible=False),
                outputs=[input_column],
                queue=False
            ).then(
                self.__process,
                inputs=[chatbot, id_input],
                outputs=chatbot,
                queue=True
            ).then(
                lambda: (gr.update(value=None),) * 10,
                inputs=None,
                outputs=[question1, question2, question3, question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True,
            ).then(
                self.__set_eval_visible,
                inputs=id_input,
                outputs=[ques_row1, ques_row2, ques_row3, pair_row,
                         question1, question2, question3, question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True,
            ).then(
                self.__select_btn,
                inputs=[id_input],
                outputs=[btn_row, finish_button],
                queue=False
            )
            
            # text input component
            text_input.submit(
                self.__add_text,
                inputs=[chatbot, text_input],
                outputs=[chatbot, text_input],
                queue=False,
            ).then(
                lambda: gr.update(visible=False),
                inputs=None,
                outputs=[input_column],
                queue=False
            ).then(
                self.__process,
                inputs=[chatbot, id_input],
                outputs=chatbot,
                queue=True
            ).then(
                self.__set_eval_visible,
                inputs=id_input,
                outputs=[ques_row1, ques_row2, ques_row3, pair_row,
                         question1, question2, question3, question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True,
            ).then(
                self.__select_btn,
                inputs=[id_input],
                outputs=[btn_row, finish_button],
                queue=False
            )
            
            # reset button
            reset_button.click(
                lambda: (gr.update(interactive=False),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=False,
            ).then(
                self.__reset,
                inputs=id_input,
                outputs=[question1, question2, question3,
                         question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4,
                         ques_row1, ques_row2, ques_row3, pair_row],
                queue=True
            ).then(
                lambda: (gr.update(interactive=True),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=True
            ).then(
                lambda: (gr.update(value=None),) * 2,
                inputs=None,
                outputs=[chatbot, audio_record],
                queue=True
            ).then(
                lambda: gr.update(value=""),
                inputs=None,
                outputs=[text_input],
                queue=True
            ).then(
                lambda: gr.update(visible=True),
                inputs=None,
                outputs=[input_column],
                queue=True
            ).then(
                lambda: (gr.update(visible=False),) * 2,
                outputs=[btn_row, finish_button],
                queue=False
            ).then(
                lambda worker_id: gr.update(value=self.__get_current_scenario(worker_id)),
                inputs=[id_input],
                outputs=[situation_description],
                queue=True
            )
            
            # continue button
            continue_button.click(
                lambda: (gr.update(interactive=False),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=False
            ).then(
                self.__clear_audio,
                inputs=audio_record,
                outputs=audio_record,
                queue=True,
            ).then(
                self.__save_survey,
                inputs=[id_input, chatbot, finish_button, situation_description,
                        question1, question2, question3,
                        question4, question5, question6,
                        pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                outputs=[ques_row1, ques_row2, ques_row3, pair_row,
                         btn_row, finish_button, situation_description, input_column,
                         question1, question2, question3,
                         question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True,
            ).then(
                lambda: (gr.update(interactive=True),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=False
            )
            
            # finish button
            finish_button.click(
                lambda: (gr.update(interactive=False),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=False
            ).then(
                self.__finish_conversation,
                inputs=[id_input, chatbot, situation_title,
                        question1, question2, question3,
                        question4, question5, question6,
                        pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                outputs=[situation_description, situation_title,
                         chatbot, ques_row1, ques_row2, ques_row3,
                         pair_row, btn_row, finish_button, input_column,
                         audio_record, text_input, last_row,
                         question1, question2, question3,
                         question4, question5, question6,
                         pairwise_question1, pairwise_question2, pairwise_question3, pairwise_question4],
                queue=True,
            ).then(
                lambda: (gr.update(interactive=True),) * 3,
                outputs=[reset_button, continue_button, finish_button],
                queue=True
            )
            
            # last question submit button
            submit_button.click(
                self.__submit,
                inputs=[id_input, submit_button, last_question],
                outputs=[last_question, submit_button, usability_message_ui,
                         last_row, finish_message, usability_row],
                queue=True,
            )
            
            # usability evaluation submit button
            usability_button.click(
                self.__submit,
                inputs=[id_input,
                        usability_button,
                        usability_question1,
                        usability_question2,
                        usability_question3,
                        usability_question4,
                        usability_question5,
                        usability_question6,
                        usability_question7,
                        usability_question8,
                        usability_question9,
                        usability_question10,],
                outputs=[usability_row, usability_message, usability_message_ui],
                queue=True
            )
        
        return form

    def __set_eval_visible(self, id_input):
        if self.user_temp[id_input]['situation_idx'] <= 2:
            return (gr.update(visible=True),) * 14
        else:
            return (gr.update(visible=True),) * 3 \
                    + (gr.update(visible=False),) * 4 \
                    + (gr.update(visible=True, scale=1, label='Is the response make sense?'), \
                        gr.update(visible=True, scale=1, label='Is the context of the response consistent?'), \
                        gr.update(visible=True, scale=1, label='Are you interested in the response? Would you like to continue the conversation?'),) \
                    + (gr.update(visible=False),) * 4

    def __get_current_scenario(self, mturk_worker_id):
        if mturk_worker_id:
            scenario_count = self.user_temp[mturk_worker_id]['scenario_count']
            situation_idx = self.user_temp[mturk_worker_id]['situation_idx']
        else:
            scenario_count = 0
            situation_idx = 0
        
        scenario_idx = 0
        if scenario_count == 0:
            scenario_idx = 0
        elif scenario_count >= 1 and scenario_count <= self.turns:
            scenario_idx = 1
        else:
            scenario_idx = 2
        
        if situation_idx <= 2:
            description = self.guidance["system_usage_instruction"].format(
                "\n".join([f"Step {i+1}) {situation}"
                           if i != scenario_idx
                           else f"<span style=\"color: red; background-color: white;\">Step {i+1}) {situation}</span>"
                           for i, situation in enumerate(self.situations[situation_idx])]))
            description = description.format(self.turns)
        else:
            description = "<span style=\"color: red; background-color: white;\">Engage in a free talk with the IA for at least 3 turns.</span>"
        return description

    def __save_irb_agreement(self, irb_agreement, irb_button, irb_msg):
        if not irb_agreement:
            return irb_agreement, irb_button, irb_msg
        return (gr.update(visible=False), ) * 2 + (gr.update(value="### Navigate to the 'System Guide' tab."), )
    
    def __add_text(self, history, text_input):
        history = history + [(text_input, None)]
        return history, gr.update(value="", placeholder="")
    
    def __add_record(self, history, audio_record):
        if not audio_record:
            return history
        history = history + [((audio_record,), None)]
        return history, gr.update(value=None)

    def __add_audio(self, history, audio_file):
        history = history + [((audio_file.name,), None)]
        return history, gr.update(value="", interactive=False)
    
    def __save_id(self, id_input, save_id_button, skip_button, *args):
        if not id_input:
            return (id_input, save_id_button, skip_button,) + tuple(args) + (gr.update(visible=False), )
        # self.data[id_input]["mturk_worker_id"] = id_input
        self.__init_user_data(id_input)
        return (gr.update(interactive=False), ) * 2 \
                + (gr.update(visible=self.skip_btn_visible),) \
                + (gr.update(visible=True),) * (len(args) + 1)
    
    def __clear_audio(self, audio):
        if not audio:
            return audio
        return gr.update(value=None)
    
    def __save_survey(self, id_input, chatbot, finish_button, situation_description, *args):
        all_questions = list(args) if self.user_temp[id_input]['situation_idx'] <= 2 else list(args[3:6])
        if None in all_questions:
            gr.Warning(self.evaluation_check_msg)
            return (gr.update(visible=True),) * 5 + (finish_button, situation_description, gr.update(visible=False),) + args
        
        assistant_message = dict()
        
        if self.user_temp[id_input]['situation_idx'] <= 2:
            # if self.random_num == 0 1=base, 2=augmented
            # else 1=augmented, 2=base
            if self.random_num == 0:
                for i in range(0, 3):
                    all_questions[i] = f"base={all_questions[i]}"
                for i in range(3, 6):
                    all_questions[i] = f"augmented={all_questions[i]}"
                
                for i in range(-1, -5, -1):
                    all_questions[i] = "base" if all_questions[i] == 1 else "augmented"
            else:
                for i in range(0, 3):
                    all_questions[i] = f"augmented={all_questions[i]}"
                for i in range(3, 6):
                    all_questions[i] = f"base={all_questions[i]}"

                for i in range(-1, -5, -1):
                    all_questions[i] = "augmented" if all_questions[i] == 1 else "base"

            message1, message2 = chatbot[-1][1].split('[Model 2]')
            message1 = message1.replace("[Model 1]","").strip()
            message2 = message2.strip()

            assistant_message['base_model'] = message1 if self.random_num == 0 else message2
            assistant_message['augmented_model'] = message2 if self.random_num == 0 else message1
        else:
            message = chatbot[-1][1]
            assistant_message['augmented_model'] = message
            for i in range(len(all_questions)):
                all_questions[i] = f"augmented={all_questions[i]}"
        
        self.user_temp[id_input]['scenario_count'] += 1
        description = self.__get_current_scenario(id_input)
        
        content = dict()
        content["mturk_worker_id"] = id_input
        content["user_speech"] = chatbot[-1][0]
        content["assistant_message"] = assistant_message
        content["answer"] = all_questions
        content["situation_index"] = self.user_temp[id_input]['situation_idx']
        self.logger.info(f"HIT: {str(content)}")
        
        del content["mturk_worker_id"]
        del content["situation_index"]
        if self.user_temp[id_input]['situation_idx'] <= 2:
            self.user_data[id_input]["result"][f"situation{self.user_temp[id_input]['situation_idx'] + 1}"].append(content)
        else:
            self.user_data[id_input]["result"]["freetalk"].append(content)
            
        return (gr.update(visible=False),) * 6 \
                + (gr.update(value=description), gr.update(visible=True),) \
                + (gr.update(value=None),) * len(args)
    
    def __process(self, history, id_input):
        input = history[-1][0]

        if isinstance(input, str):
            response = self.user_model[id_input].fn(text_input=input)
        elif isinstance(input, tuple):
            response = self.user_model[id_input].fn(audio_file=input[0])
        else:
            response = self.user_model[id_input].fn(
                text_input=input,
                forced_response="This type of input is not supported.",
            )
        transcript, message, message2 = response
        print(f"- User: {transcript}")

        message = message.strip()
        message2 = message2.strip()
        self.random_num = random.randrange(2)
        #speech_data = f"data:audio/wav;base64,{speech if self.random_num == 0 else speech2}"
        #output = f"[Model 1]<br/><audio controls autoplay src=\"{speech_data}\" type=\"audio/wav\"></audio>"
        if self.user_temp[id_input]['situation_idx'] <= 2:
            output = "[Model 1]\n"
            output += message if self.random_num == 0 else message2
        
            #speech_data = f"data:audio/wav;base64,{speech2 if self.random_num == 0 else speech}"
            output += f"\n\n[Model 2]\n" #<br/><audio controls src=\"{speech_data}\" type=\"audio/wav\"></audio>"
            output += message2 if self.random_num == 0 else message

            print(f"- Assistant 1: {message if self.random_num == 0 else message2}")
            print(f"- Assistant 2: {message2 if self.random_num == 0 else message}")
        else:
            output = message2
            print(f"- Our Assistant: {message2}")

        history[-1] = (transcript, output)
        return history
    
    def __reset(self, id_input):
        content = dict()
        content['mturk_worker_id'] = id_input
        if self.user_temp[id_input]['situation_idx'] <= 2:
            content["situation_index"] = self.user_temp[id_input]['situation_idx']
        content['message'] = "The conversation history has been reset."
        self.user_temp[id_input]['scenario_count'] = 0
        self.logger.info(f"HIT: {str(content)}")
        if self.user_temp[id_input]['situation_idx'] <= 2:
            self.user_data[id_input]["result"][f"situation{self.user_temp[id_input]['situation_idx'] + 1}"].clear()
        else:
            self.user_data[id_input]["result"][f"freetalk"].clear()

        self.user_model[id_input].generate_model_1.reset()
        self.user_model[id_input].generate_model_2.reset()

        return (gr.update(value=None),) * 10 + (gr.update(visible=False),) * 4
    
    def __select_btn(self, id_input):
        if self.user_temp[id_input]['scenario_count'] >= (self.turns + 1):
            return gr.update(visible=True), gr.update(visible=True)
        elif self.user_temp[id_input]['situation_idx'] > 2 and self.user_temp[id_input]['scenario_count'] >= (self.turns - 1):
            return gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=True), gr.update(visible=False)
    
    def __skip_situations(self, id_input):
        self.user_temp[id_input]['situation_idx'] = 3
        self.user_temp[id_input]['scenario_count'] = 0
        situation_msg = self.__get_current_scenario(id_input)
        self.user_model[id_input].generate_model_1.reset()
        self.user_model[id_input].generate_model_2.reset()
        self.logger.info(f"HIT: User {id_input} skipped situations.")
        return (gr.update(value=situation_msg),) \
                + (gr.update(value=None),) \
                + (gr.update(visible=False),) * 6 \
                + (gr.update(visible=True),) \
                + (gr.update(value=None, visible=True),) * 2 \
                + (gr.update(visible=False),) * 2 \
                + (gr.update(value=False),) * 10
    
    def __finish_conversation(self,
                             id_input, chatbot,
                             situation_title, *args):
        all_questions = list(args) if self.user_temp[id_input]['situation_idx'] <= 2 else list(args[3:6])
        if None in all_questions:
            gr.Warning(self.evaluation_check_msg)
            return (gr.update(visible=True),) * 9 \
                    + (gr.update(visible=False),) \
                    + (gr.update(visible=True),) * 2 \
                    + (gr.update(visible=False),) * 1 \
                    + args
    
        assistant_message = dict()

        if self.user_temp[id_input]['situation_idx'] <= 2:
            # if self.random_num == 0 1=base, 2=augmented
            # else 1=augmented, 2=base
            if self.random_num == 0:
                for i in range(0, 3):
                    all_questions[i] = f"base={all_questions[i]}"
                for i in range(3, 6):
                    all_questions[i] = f"augmented={all_questions[i]}"

                for i in range(-1, -5, -1):
                    all_questions[i] = "base" if all_questions[i] == 1 else "augmented"
            else:
                for i in range(0, 3):
                    all_questions[i] = f"augmented={all_questions[i]}"
                for i in range(3, 6):
                    all_questions[i] = f"base={all_questions[i]}"

                for i in range(-1, -5, -1):
                    all_questions[i] = "augmented" if all_questions[i] == 1 else "base"
        
            message1, message2 = chatbot[-1][1].split('[Model 2]')
            message1 = message1.replace("[Model 1]","").strip()
            message2 = message2.strip()

            assistant_message['base_model'] = message1 if self.random_num == 0 else message2
            assistant_message['augmented_model'] = message2 if self.random_num == 0 else message1
        else:
            message = chatbot[-1][1]
            assistant_message['augmented_model'] = message
            for i in range(len(all_questions)):
                all_questions[i] = f"augmented={all_questions[i]}"
        
        content = dict()
        content["mturk_worker_id"] = id_input
        content["user_speech"] = chatbot[-1][0]
        content["assistant_message"] = assistant_message
        content["answer"] = all_questions
        content["situation_index"] = self.user_temp[id_input]['situation_idx']
        self.logger.info(f"HIT: {str(content)}")
        
        del content["mturk_worker_id"]
        del content["situation_index"]
        if self.user_temp[id_input]['situation_idx'] <= 2:
            self.user_data[id_input]["result"][f"situation{self.user_temp[id_input]['situation_idx'] + 1}"].append(content)
        else:
            self.user_data[id_input]["result"]["freetalk"].append(content)
        
        self.user_temp[id_input]['scenario_count'] = 0
        self.user_temp[id_input]['situation_idx'] += 1
        if self.user_temp[id_input]['situation_idx'] > 3:
            return (gr.update(visible=False), ) * 12 \
                    + (gr.update(visible=True),) \
                    + (gr.update(visible=False),) * len(args)
        else:
            self.user_model[id_input].generate_model_1.reset()
            self.user_model[id_input].generate_model_2.reset()
            
            situation_msg = self.__get_current_scenario(id_input)
            return (gr.update(value=situation_msg), situation_title, gr.update(value=None),) \
                    + (gr.update(visible=False),) * 6 \
                    + (gr.update(visible=True),) \
                    + (gr.update(value=None, visible=True),) * 2 \
                    + (gr.update(visible=False),) \
                    + (gr.update(value=False),) * 10
    
    def __submit(self, id_input, submit_button, *args):
        if None in args:
            if len(args) == 1:
                return args + (submit_button,) \
                        + (gr.update(visible=False),) * 2 \
                        + (gr.update(visible=True),) * 2
            else:
                gr.Warning(self.evaluation_check_msg)
                return gr.update(visible=True), \
                    gr.update(visible=False), gr.update(visible=False)
        content = dict()
        content["mturk_worker_id"] = id_input
        if len(args) == 1:
            content["last_answer"] = args[0]
            self.user_data[id_input]["result"]["last_answer"] = args[0]
        else:
            content["usability_answer"] = list(args)
            self.user_data[id_input]["result"]["usability_answer"] = list(args)
        self.logger.info(f"HIT: {str(content)}")
        
        os.makedirs(f"{self.data_path}", exist_ok=True)
        userid = id_input.replace("/","-")
        with open(f"{self.data_path}/user_{userid}_data.json", "w", encoding="utf-8") as file:
            json.dump(self.user_data[id_input], file)
        
        self.user_temp[id_input]['situation_idx'] = 0
        self.user_temp[id_input]['scenario_count'] = 0
        
        if len(args) == 1:
            return (gr.update(visible=False),) * 2 \
                    + (gr.update(visible=False),) * 2 \
                    + (gr.update(visible=True),) * 2
        else:
            return gr.update(visible=False), \
                    gr.update(visible=True, value=self.guidance["end_of_tasks"]), \
                    gr.update(visible=True)
    
    def __load_scenario(self, prefix: str) -> list[str]:
        index = 1
        scenarios = []

        while True:
            filepath = osp.join("conversation", "scenarios", f"{prefix}{index}.txt")
            if osp.exists(filepath):
                with open(filepath, "r") as file:
                    scenario = [line.strip() for line in file.readlines()]
                    scenarios.append(scenario)
                index += 1
            else:
                break
        
        return scenarios
    
    def __load_guidance(self, name: str) -> str:
        with open(osp.join("conversation", "guidances", f"{name}.txt"), "r", encoding="utf-8") as file:
            return file.read()
    
    def __load_experiment_description(self, name: str, situations: list[str]) -> str:
        with open(osp.join("conversation", "guidances", f"{name}.txt"), "r", encoding="utf-8") as file:
            description = file.read().format(
                situations[0][0], situations[0][1], situations[0][2],
                situations[1][0], situations[1][1], situations[1][2],
                situations[2][0], situations[2][1], situations[2][2],
            )
            description = description.format(self.turns, self.turns, self.turns)
            return description
