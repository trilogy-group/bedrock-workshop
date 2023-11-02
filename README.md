
# AWS Bedrock and LangChain Workshop

This repository contains code to create a question answering bot that can answer questions about AWS based on the [AWS Made Easy Livestream](https://www.youtube.com/channel/UC-iqkyAw-jFd0RtdRdqzvRQ) livestream. The bot is built using [AWS Bedrock](https://aws.amazon.com/bedrock/) for llms and [LangChain](https://www.langchain.com/).

## Workshop Agenda

In this workshop we will:

- Set up an AWS Bedrock environment 
- Scrape transcripts from the AWS Made Easy podcast
- Preprocess and format the text data
- Store the transcripts in vector store usin Titan Embeddings 
- Implement RAG to retrieve relevant documents when asked a question

## Prerequisites

- An AWS account
- Python 3.7+
- Git and a GitHub account

## Getting Started

1. Clone this repository 
2. Download `ame-data-oct5.csv` to get all podcast transcripts
3. Run `create_scripts.py` to get txt transcrits
4. Run `streamlit run app.py` to start the streamlit server


## Resources

- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [LangChain](https://www.langchain.com/)
- [AWS Made Easy Livestream](https://www.youtube.com/channel/UC-iqkyAw-jFd0RtdRdqzvRQ)

## Credits

This workshop was created by [Harsh Arya](https://github.com/harsharyacnu).