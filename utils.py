from collections import defaultdict
from functools import lru_cache

import bs4
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from gtts import gTTS
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import streamlit as st


def get_article_links(company_name: str):
    url = f"https://timesofindia.indiatimes.com/topic/{company_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    articles_list = soup.find_all(class_="uwU81")
    articles_link = [tag.a["href"] for tag in articles_list][:10]
    return articles_link


def get_article_text(links):
    loader_multiple_pages = WebBaseLoader(
        links,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("_s30J clearfix  ", "HNMDR"))
        ),
    )

    data = loader_multiple_pages.load()
    articles = [content.page_content for content in data]

    return articles


def load_hf_model(repo_id="google/gemma-2-2b-it"):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
    )

    return ChatHuggingFace(llm=llm)


def cal_sentiment_dist(article_result):
    sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
    for result in article_result:
        sentiment_distribution[result["sentiment"]] += 1

    return sentiment_distribution


def final_dict_result(
    company_name: str,
    article_result: list,
    sentiment_dist: dict,
    cov_diff: list,
    final_sentiment: str,
):
    return {
        "Company": company_name,
        "Articles": article_result,
        "Comparative Study": {
            "Sentiment Distributution": sentiment_dist,
            "Coverage Differences": cov_diff,
        },
        "Final Sentiment Analysis": final_sentiment,
    }


def generate_direct_hindi_tts(
    hindi_text: str, output_file: str = "final_sentiment_hindi.mp3"
):
    """
    Directly use English sentiment analysis text and generate TTS with Hindi language settings.
    Note: This may not be as natural since the text remains in English.

    Parameters:
        english_text (str): The final sentiment analysis text in English.
        output_file (str): The name for the output audio file.

    Returns:
        str: The path to the generated audio file.
    """
    tts = gTTS(text=hindi_text, lang="hi")
    tts.save(output_file)
    return output_file


from transformers import pipeline


def translate_to_hindi(text: str) -> str:
    """
    Translate English text to Hindi using Hugging Face API.

    Parameters:
        text (str): The English text.

    Returns:
        str: Translated Hindi text.
    """
    # Load translation pipeline from Hugging Face
    translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")

    # Translate text
    translation = translator(text)

    return translation[0]["translation_text"]  # Extract translated text
