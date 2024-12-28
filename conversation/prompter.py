import os
import os.path as osp
import time
import json
import typing
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, DPRConfig, DPRQuestionEncoder, DPRContextEncoder
from dataclasses import dataclass, field
from utils.model import Model
from models.palm.core import PaLM
from conversation.retriever import BiEncoderRetriever

@dataclass
class MemoryConfig:
    stm_capacity: int = 10
    ltm_capacity: int = 5000
    similarity_threshold: float = 0.75
    time_window: float = 3600

@dataclass
class ReasoningConfig:
    min_steps: int = 3
    max_steps: int = 5
    temperature: float = 0.7
    top_k: int = 5

@dataclass
class PromptConfig:
    remove_cot: bool = False
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    beta: float = 0.6  # Weight for balancing semantic similarity vs recency

@dataclass
class MemoryEntry:
    query: str  # qi: user query
    response: str  # ri: system response
    embedding: np.ndarray  # ei: context embedding
    timestamp: float

    @property
    def content(self) -> str:
        return f"Q: {self.query}\nA: {self.response}"

class ConversationState:
    """Maintains conversation state and history"""
    def __init__(self):
        self.current_query: typing.Optional[str] = None  # qt
        self.current_embedding: typing.Optional[np.ndarray] = None  # et 
        self.conversation_state: typing.Dict[str, typing.Any] = {}  # St
        self.history: typing.List[MemoryEntry] = []  # Ht

    def update(self, query: str, response: str, embedding: np.ndarray) -> None:
        """Update conversation state with new interaction"""
        self.current_query = query
        self.current_embedding = embedding
        
        entry = MemoryEntry(
            query=query,
            response=response,
            embedding=embedding,
            timestamp=time.time()
        )
        self.history.append(entry)
        
        # Update conversation state (St)
        self.conversation_state.update({
            "last_interaction": entry,
            "total_turns": len(self.history)
        })

class BasePrompter(Model):
    def __init__(
        self,
        model_class: typing.Type[Model],
    ) -> None:
        super().__init__()

        self.model_class = model_class
        self.model_loaded = False
        self.fn = self.prompt
    
    def reset(self) -> None:
        self.__instantiate_model()
    
    def __instantiate_model(self) -> Model:
        self.model = self.model_class(context=True)
        self.model_loaded = True
        return self.model
    
    def prompt(
        self,
        input: str,
    ) -> str:
        completion = "".join(self.model.fn(
            input=input,
            temperature=0.7,
            stop=["\n"],
        ))
        return completion

class AugmentedPrompter(Model):
    def __init__(
        self,
        model_class: typing.Type[Model],
    ) -> None:
        super().__init__()

        self.model_class = model_class
        self.model_loaded = False
        self.fn = self.prompt
    
    def reset(self) -> None:
        self.__instantiate_model()
        self.config = PromptConfig()
        self.templates = self._load_prompts("conversation/prompts/")
        self.session = {
            "history": [],
            "history_summaries": [],
            "prefix": None,
            "suffix": None,
        }
        self.retriever = BiEncoderRetriever()
    
    def __instantiate_model(self) -> Model:
        self.model = self.model_class(context=False)
        self.role_key = "role"

        if isinstance(self.model, PaLM):
            self.model = self.model_class(
                model="models/text-bison-001",
                context=False,
            )
            self.role_key = "author"

        self.model_loaded = True
        return self.model
    
    def prompt(
        self,
        input: str,
    ) -> str:
        attempts = 0
        while attempts < 3:
            try:
                # Conversation Layer
                print("\n ** Extractor **")
                knowledge, query = self.extract(input)
                print("* Knowledge, Question:", (knowledge, query))

                print("\n ** Retriever **")
                retrieval = self.retrieve(query)
                print("* Retrieval:", retrieval)

                print("\n ** Reasoning **")
                conclusion = self.reasoning(knowledge + retrieval, query)
                print("* Conclusion:", conclusion)

                print("\n ** Generator **")
                response = self.generate(conclusion, query)
                print("* Generation:", response)

                extended_history = [
                    {self.role_key: "user", "content": input},
                    {self.role_key: "assistant", "content": response},
                ]
                self.session["history"].extend(extended_history)

                # Memorization Layer
                ## TODO: Conversation 결과를 제공한 후 Memorize하여 응답 시간 단축
                summaries = self.summarize(self.session["history"])
                print(" ** Summarization **\n", summaries)

                self.session["history_summaries"].extend(summaries)

                return response
            except Exception as e:
                print(f"Attempt {attempts+1} failed with error: {e}")
                if attempts < 3:
                    time.sleep(1)
                attempts += 1
        return "Sorry, there was an error processing your request. Please try again, and if the error persists, reset the conversation and start over."

    def clarify(
        self,
        input: str,
        num_examples: int = None,
    ) -> str:
        if num_examples is None:
            num_examples = len(self.examples["clarifier"])
        input = self.templates["clarifier"].format(
            examples="\n".join(self.examples["clarifier"][:num_examples]),
            input=input,
        )

        clarified_question = "".join(self.model.fn(
            input,
            temperature=0.3,
        ))
        return clarified_question

    def extract(
        self,
        input: str,
    ) -> tuple[list[str], str]:
        prompt = self.templates["extractor"].format(
            input=input,
        )
        completion = "".join(self.model.fn(
            input=prompt,
            temperature=0,
        ))
        print("* Completion:", completion)

        knowledge = self._parse_completion(completion, "Knowledge")
        query = self._parse_completion(completion, "Query")[0]
    
        return knowledge, query
    
    def retrieve(
        self,
        query: str,
    ) -> list[str]:
        prompt = self.templates["retriever"].format(
            question=query,
        )
        print("* Prompt:", prompt)
        completion = "".join(self.model.fn(
            input=prompt,
            temperature=0,
            history=self.session["history"],
        ))
        print("* History:", self.session["history"])
        print("* Completion:", completion)

        hsitory_content = [f"{x[self.role_key]}: {x['content']}" for x in self.session["history"]]

        if len(self.session["history_summaries"]) == 0:
            if "i can't answer" in completion.strip().lower() or "i cannot answer" in completion.strip().lower():
                return hsitory_content
            return [completion]

        retrieval = self.retriever.retrieve_top_summaries(
            query, self.session["history_summaries"],
        )

        if "i can't answer" in completion.strip().lower() or "i cannot answer" in completion.strip().lower():
            if retrieval:
                return retrieval
            return self.session["history_summaries"]
        return [completion] + retrieval
    
    def reasoning(
        self,
        knowledge: list[str],
        query: str,
    ) -> str:
        prompt = self.templates["reasoner"].format(
            knowledge="\n".join(f"({i+1}) {item}" for i, item in enumerate(knowledge)),
            query=query,
        )
        print("* Prompt:", prompt)
        completion = "".join(self.model.fn(
            input=prompt,
            temperature=0,
        ))
        print("* Completion:", completion)

        #conclusion = self._parse_completion(completion, "Conclusion")[0]
        conclusion = completion

        if len(conclusion) == 0:
            conclusion = ""
            print("** No conclusions were reached. **")
            print(completion)
            return completion
        
        return conclusion

    def generate(
        self,
        information: str,
        query: str,
    ) -> str:
        prompt = self.templates["generator"].format(
            query=query,
            information=information,
        )
        print("* Prompt:", prompt)
        completion = "".join(self.model.fn(
            input=prompt,
            temperature=0.7,
        ))
        print("* Completion:", completion)
        
        return completion
    
    def summarize(
        self,
        history: list[dict[str, str]],
    ) -> list[str]:
        dialogue = "\n".join(f"{item[self.role_key]}: {item['content']}" for item in history)

        prompt = self.templates["summarizer"].format(
            dialogue=dialogue,
        )
        print("* Prompt:", prompt)
        completion = "".join(self.model.fn(
            input=prompt,
            temperature=0,
        ))
        print("* Completion:", completion)

        summary = self._parse_completion(completion, "Summary")
        
        return summary

    def _load_templates(
        self,
        filename: str,
    ) -> dict:
        try:
            with open(osp.join("conversation", "configs", filename), "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise Exception(f"Error: 'conversation/configs/{filename}' file not found!")
        except json.JSONDecodeError:
            raise Exception(f"Error: JSON decoding failed for 'conversation/configs/{filename}'!")
        
    def _load_prompts(
        self,
        directory: str,
    ) -> dict[str, str]:
        prompts = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                name_without_extension = os.path.splitext(filename)[0]
                prompts[name_without_extension] = content    
        return prompts
    
    def _parse_completion(
        self,
        completion: str,
        title: str
    ) -> typing.Union[list[str], str]:
        start_tags = [f"#{title}\n", f"#{title}:", f"{title}:", f"{title}\n"]
        end_tag = "#"
        
        start_tag = next((tag for tag in start_tags if tag in completion), None)
        
        if start_tag:
            content = completion.split(start_tag)[1].split(end_tag)[0].strip()
            
            # Determine if the content starts with '- ' (to decide if returning a list)
            if content.startswith("- "):
                return [item.strip().replace("- ", "").replace("* ", "") for item in content.split("\n") if item.strip()]
            else:
                return [content]
        else:
            return []

    def _combine_knowledge(
        self, 
        *args,
    ) -> str:
        combined = [f"({idx + 1}) {item}" for idx, item in enumerate(item for arg in args for item in arg)]
        return " ".join(combined)

class MAIAPrompter(Model):
    def __init__(
        self,
        model_class: typing.Type[Model],
        config: typing.Optional[PromptConfig] = None
    ) -> None:
        super().__init__()
        self.model_class = model_class
        self.config = config or PromptConfig()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.conversation_state = ConversationState()
        
        # Initialize encoders
        self.query_encoder = None  # E_Q
        self.passage_encoder = None  # E_P
        
        # Memory storages
        self.stm = []  # Short-term memory
        self.ltm = []  # Long-term memory
        
    def reset(self) -> None:
        """Initialize models, matrices and conversation state"""
        self.__instantiate_model()
        self.templates = self._load_prompts("conversation/prompts/")
        self.__initialize_encoders()
        self.conversation_state = ConversationState()
    
    def _load_prompts(
        self,
        directory: str,
    ) -> dict[str, str]:
        prompts = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                name_without_extension = os.path.splitext(filename)[0]
                prompts[name_without_extension] = content    
        return prompts
        
    def __instantiate_model(self) -> None:
        """Initialize main language model"""
        self.model = self.model_class(context=True)
        self.model_loaded = True
        
    def __initialize_encoders(self) -> None:
        """Initialize DPR encoders for query and passage"""
        config = DPRConfig(output_attentions=True, output_hidden_states=True)
        self.dpr_tokenizer = AutoTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
        self.query_encoder = DPRQuestionEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", config=config).to(self.device)
        self.passage_encoder = DPRContextEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder").to(self.device)

    def prompt(self, input: str) -> str:
        """Prompt chaining process"""
        try:
            # Context Extraction
            # self.state["qt"] = input
            # et = self._extract_context(input)
            # self.state["et"] = et
            et = self._extract_context(input)
            self.conversation_state.current_embedding = et
            self.conversation_state.current_query = input
            
            # Memory Retrieval
            R_qt = self._retrieve_memories(input)
            
            # Prompt Generation
            P = self._generate_prompts(input, R_qt)
            
            # Multi-step Reasoning and Response Generation
            response = self._execute_reasoning_chain(P)
            
            # Update memories
            et_ = self._encode_passage(input, response)['pooler_output']
            self._update_memories(input, response, et_)
            
            return response
            
        except Exception as e:
            print(f"Error in prompt chain: {e}")
            return "I apologize, I encountered an error. Please try again."

    def _extract_context(self, qt: str) -> np.ndarray:
        """Context Extractor"""
        # Get hidden states directly from encoder
        encoded = self._encode_query(qt)
        return encoded['attentions']

        # h = encoded['hidden_states'][-1]

        # attention_layer = self.query_encoder.base_model.base_model.encoder.layer[0].attention.self
        
        # # Compute attention weights
        # d_k = h.shape[-1]
        # W_q = attention_layer.query.weight
        # W_k = attention_layer.key.weight
        # attention_scores = np.dot(W_q @ h, (W_k @ h).T) / np.sqrt(d_k)
        # alpha = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
        
        # # Compute context vector
        # return np.sum(alpha[:, np.newaxis] * h, axis=0)

    def _retrieve_memories(self, qt: str) -> typing.List[MemoryEntry]:
        """Memory Module returning MemoryEntry list"""
        query_vector = self._encode_query(qt)['pooler_output']
        retrieved = []
        for memory in self.stm + self.ltm:
            sim = np.dot(query_vector.T, memory.embedding)
            if sim > self.config.memory.similarity_threshold:
                retrieved.append((memory, sim))

        stored = sorted(retrieved, key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in stored[:self.config.reasoning.top_k]]

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode query using query encoder"""
        input_dict = self.dpr_tokenizer(text, padding='max_length', max_length=32, truncation=True, return_tensors="pt").to(self.device)
        del input_dict["token_type_ids"]
        return self.query_encoder(**input_dict)
        
    def _encode_passage(self, q: str, a: str) -> np.ndarray:
        """Encode passage using passage encoder"""
        input_dict = self.dpr_tokenizer([q, a], padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(self.device)
        del input_dict["token_type_ids"]
        return self.passage_encoder(**input_dict)

    def _generate_prompts(self, qt: str, R_qt: typing.List[MemoryEntry]) -> typing.List[str]:
        """Prompt Generator"""
        # Calculate number of reasoning steps
        n = min(
            max(
                self.config.reasoning.min_steps,
                int(np.log2(len(R_qt) + 1))
            ),
            self.config.reasoning.max_steps
        )
        
        # Sort context by priority
        R_sorted = self._prioritize_context(R_qt)
        
        prompts = []
        outputs = []  # [o1, ..., oi-1]
        
        for i in range(n):
            # Select context subset for current step
            step_context = R_sorted[:i+1]  # {c1, ..., ci}
            
            # Generate prompt using template function
            prompt = self._f_template(
                query=qt,
                context=[entry.query + ": " + entry.response for entry in step_context],
                previous_outputs=outputs
            )
            
            # Execute current step
            output = self.model.fn(prompt, temperature=self.config.reasoning.temperature)
            
            prompts.append(prompt)
            outputs.append(output)
                
        return prompts

    def _prioritize_context(self, contexts: typing.List[MemoryEntry]) -> typing.List[MemoryEntry]:
        """Context prioritization"""
        priorities = []
        
        for memory in contexts:
            # Use stored embeddings
            sim = np.dot(self.conversation_state.current_embedding.T, 
                        memory.embedding)
            
            # Compute recency score directly from memory timestamp
            recency = self._compute_recency(memory)
            
            # Calculate priority
            priority = (
                self.config.beta * sim + 
                (1 - self.config.beta) * recency
            )
            priorities.append((memory, priority))
        
        return [memory for memory, _ in sorted(
            priorities, 
            key=lambda x: x[1], 
            reverse=True
        )]

    def _execute_reasoning_chain(self, prompts: typing.List[str]) -> str:
        """Multi-step reasoning"""
        intermediate_outputs = []
        
        for p_i in prompts:
            # Get LLM response for each prompt
            output = self.model.fn(p_i, temperature=self.config.reasoning.temperature)
            intermediate_outputs.append(output)
        
        # Generate final response based on reasoning chain
        final_prompt = self._f_template(
            self.conversation_state.current_query,
            intermediate_outputs,
            []
        )
        
        return self.model.fn(final_prompt, temperature=self.config.reasoning.temperature)

    def _update_memories(self, qt: str, rt: str, et: np.ndarray) -> None:
        """Memory update"""
        current_time = time.time()
        
        # Update conversation state
        self.conversation_state.update(qt, rt, et)
        
        # Update STM: maintain last k turns within time window
        time_window = current_time - (self.config.memory.time_window)
        self.stm = [
            entry for entry in self.conversation_state.history
            if entry.timestamp >= time_window
        ][-self.config.memory.stm_capacity:]
        
        # Update LTM with entries outside the time window
        older_entries = [
            entry for entry in self.conversation_state.history
            if entry.timestamp < time_window
        ]
        self.ltm.extend(older_entries)
        
        # Maintain LTM capacity
        if len(self.ltm) > self.config.memory.ltm_capacity:
            self.ltm = self.ltm[-self.config.memory.ltm_capacity:]

    def _compute_recency(self, memory: MemoryEntry) -> float:
        """Compute recency score for memory entry"""
        current_time = time.time()
        time_diff = current_time - memory.timestamp
        return 1 / (1 + time_diff)

    def _f_template(
        self, 
        query: str, 
        context: typing.List[str], 
        previous_outputs: typing.List[str]
    ) -> str:
        """Template function implementation"""
        template = self.templates["reasoning_step"]
        return template.format(
            query=query,
            context="\n".join(context),
            previous_steps="\n".join(previous_outputs)
        )
