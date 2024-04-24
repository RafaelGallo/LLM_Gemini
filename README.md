# LLM Gemini

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/transformers-white.svg)](https://huggingface.co/docs/transformers)
[![](https://img.shields.io/badge/Google_Cloud-white.svg)](https://huggingface.co/docs/transformers)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/) 

![Logo](https://logowik.com/content/uploads/images/google-ai-gemini91216.logowik.com.webp)

# Descrição projeto

Este projeto LLM com o modelo Gemini representa um esforço significativo para explorar a riqueza dos dados textuais em termos de sentimentos e emoções. Ao incorporar análise de sentimento e classificação de sentimentos, estamos não só identificando as emoções expressas, mas também categorizando-as de forma eficaz para insights mais precisos. Além disso, a etapa de clusterização acrescenta uma camada adicional de sofisticação, permitindo-nos agrupar os sentimentos de maneira mais granular, revelando padrões sutis e tendências subjacentes nos textos analisados. Essa abordagem multidimensional não apenas amplia nossa compreensão dos dados, mas também nos capacita a extrair insights mais valiosos e acionáveis para informar tomadas de decisão estratégicas.

## Stack utilizada

**Ciência de dados:** Python, Sklearn, Transforns, Google Cloud

## Variáveis de Ambiente

Para preparar o ambiente Python deste projeto, instale as seguintes bibliotecas utilizando o gerenciador de pacotes conda:


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib
  conda install transformers
  conda install torch

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
  transformers==1.1.0
  torch==2.3.0
```
A criação de um ambiente virtual para este projeto, o que facilitará a administração e evitará conflitos de dependências. Se precisar de orientações sobre como criar ambientes virtuais em Python, consulte este guia.

Para criar um ambiente virtual no diretório do seu projeto, execute o seguinte comando:

```bash
python -m venv .env
```
Para ativar o ambiente virtual, utilize:
```bash
source .env/bin/activate
```
Agora, você pode instalar o Transformers com o seguinte comando:

```bash
pip install transformers
```
Se estiver trabalhando apenas com CPU, você pode instalar o Transformers e a biblioteca de deep learning correspondente em uma única linha. Por exemplo, para instalar o Transformers e o PyTorch:

```bash
pip install transformers[torch]
```
Para utilizar o Transformers com TensorFlow 2.0, utilize o seguinte comando:

```bash
pip install transformers[tf-cpu]
```
## Demo modelo LLM Gemini

Neste trecho de código, estamos importando e configurando as bibliotecas necessárias para utilizar o modelo Gemini com a biblioteca Transformers.

```
# Importação das bibliotecas

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


# Utiliza o pipeline para facilitar a geração de texto com o modelo Gemini
pipe = pipeline("text2text-generation", model="describeai/gemini")

# Importa o tokenizer para preparar os textos antes de passá-los para o modelo
tokenizer = AutoTokenizer.from_pretrained("describeai/gemini")

# Importa o modelo Gemini pré-treinado
model = AutoModelForSeq2SeqLM.from_pretrained("describeai/gemini")

# Configura o pipeline para geração de texto com o modelo Gemini
summarizer = pipeline('text2text-generation', model='describeai/gemini')

# Código a ser resumido
code = "print('hello world!')"

# Gera um resumo do código
response = summarizer(code, max_length=100, num_beams=3)

# Imprime o resumo gerado
print("Código resumido: " + response[0]['generated_text'])




```

## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com
## Referência

 - [Modelos de linguagem grandes com a tecnologia de ponta da IA do Google](https://cloud.google.com/ai/llms?hl=pt-BR)
 - [Hugging Face](https://huggingface.co/describeai/gemini)
 - [Hugging Face gemma-7b ](https://huggingface.co/google/gemma-7b)

