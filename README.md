# MAIA

MAIA (Memory-Augmented Intelligent Assistant) implements prompt chaining with enhanced long-term recall capabilities for LLM-powered intelligent assistants.

This work has been accepted to IUI 2025 (ACM Conference on Intelligent User Interfaces). [📄 Read the paper](https://doi.org/10.1145/3708359.3712117). Citation details are available in the [Citation](#citation) section.

## Requirements

- Python 3.9
- CUDA 11.7
- PyTorch
- Transformers
- Additional dependencies listed in `environment.yml` or `requirements.txt`

## Installation

Install dependencies using either conda:
```bash
conda env create --file environment.yml
```
Or pip:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following environment variables:
```
SSL_CERT_PATH={Path to SSL fullchain}
SSL_KEY_PATH={Path to SSL privkey}
OPENAI_API_KEY={OpenAI API Key}
PALM_API_KEY={PaLM API Key}
GOOGLE_TTS_API_KEY={Google Cloud Text-to-Speech API Key}
```

## Implementation

MAIA's core framework is implemented in the `MAIAPrompter` class ([`conversation/prompter.py`](conversation/prompter.py)). Key features include:
- Context extraction using pretrained transformer attention
- Short-term and long-term memory management
- Multi-step reasoning with dynamic prompt generation
- DPR-based memory retrieval

## Usage

### Web Interface

Launch the Gradio-based web interface for voice and text interactions:
```bash
python run_gradio.py [--server_name SERVER_NAME] [--server_port PORT] [--share]
```
Optional arguments:
- `--server_name`: Specify server name (default: 0.0.0.0)
- `--server_port`: Specify port number
- `--share`: Enable Gradio's share feature

### Console Interface

Start text-based interactions in console:
```bash
python app.py
```

## Architecture

![system_architecture](https://github.com/user-attachments/assets/0f4fdf3f-82f6-4efb-b132-ea9ac8744efa)

MAIA consists of four main components:
1. **Context Extraction**: Pretrained transformer attention mechanism
2. **Memory Module**: DPR-based retrieval with STM/LTM
3. **Prompt Generation**: Dynamic multi-step reasoning chain
4. **Response Generation**: Template-based response synthesis

## Citation

```bibtex
@inproceedings{seo2025prompt,
  author = {Seo, Seongbum and Yoo, Sangbong and Jang, Yun},
  title = {A Prompt Chaining Framework for Long-Term Recall in LLM-Powered Intelligent Assistant},
  year = {2025},
  month = mar,
  isbn = {9798400713064},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3708359.3712117},
  doi = {10.1145/3708359.3712117},
  booktitle = {Proceedings of the 30th International Conference on Intelligent User Interfaces},
  pages = {89–105},
  numpages = {17},
  keywords = {Intelligent Assistants, Large Language Models, Prompt Chaining},
  location = {Cagliari, Italy},
  series = {IUI '25}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details
