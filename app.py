import json
import time

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from schema import Article, ComparativeSentimentScore
from templates import article_template, coverage_diff_template, final_sentiment_prompt
from utils import (
    cal_sentiment_dist,
    final_dict_result,
    generate_direct_hindi_tts,
    get_article_links,
    get_article_text,
    load_hf_model,
    translate_to_hindi,
)

import streamlit as st

# Streamlit UI Setup
st.title("News Sentiment Analyzer")
st.sidebar.header("Configuration")

company = st.sidebar.text_input("Enter Company Name", "apple")
api_token = st.sidebar.text_input("Enter HuggingFace API TOKEN", "...")
process_button = st.sidebar.button("Run Analysis")

if process_button:
    st.write("## Processing News Articles...")
    progress_bar = st.progress(0)

    # Step 1: Fetch Articles
    progress_bar.progress(10)
    st.write("### Fetching Articles...")
    article_links = get_article_links(company)

    # Step 2: Retrieve Article Text
    progress_bar.progress(30)
    st.write("### Extracting Article Content...")
    articles = get_article_text(article_links)

    # Step 3: Load Language Model
    progress_bar.progress(50)
    st.write("### Loading Sentiment Analysis Model...")
    model = load_hf_model(api_token=api_token)

    article_parser = PydanticOutputParser(pydantic_object=Article)
    comp_diff_parser = PydanticOutputParser(pydantic_object=ComparativeSentimentScore)

    # Step 4: Process Articles
    progress_bar.progress(70)
    st.write("### Analyzing Articles...")
    article_chain = article_template | model | article_parser
    article_result = []
    for article in articles:
        try:
            article_result.append(dict(article_chain.invoke({"content": article})))
        except:
            st.write("Some article cannot be analyze")

    # Step 5: Sentiment Comparison
    progress_bar.progress(80)
    st.write("### Performing Comparative Analysis...")
    comp_diff_chain = coverage_diff_template | model | comp_diff_parser
    comp_diff_result = comp_diff_chain.invoke(
        {"company": company, "articles": article_result}
    )

    comp_diff_list = []
    for key, value in dict(comp_diff_result).items():
        # print(f"{key}: {value}")
        for v in value:
            comp_diff_list.append(dict(v))

    sentiment_dist = cal_sentiment_dist(article_result)
    final_sentiment_chain = final_sentiment_prompt | model | StrOutputParser()
    final_sentiment_result = final_sentiment_chain.invoke(
        {
            "company": company,
            "articles": article_result,
            "comparative_analysis": {
                "Coverage Differences": comp_diff_list,
                "sentiment_distribution": sentiment_dist,
            },
        }
    )

    # Display Results
    st.write("## Final Sentiment Analysis")
    st.json(
        final_dict_result(
            company,
            article_result,
            sentiment_dist,
            comp_diff_list,
            final_sentiment_result,
        )
    )
    # Step 6: Translate to Hindi
    progress_bar.progress(90)
    st.write("### Translating to Hindi...")
    hindi_translation = translate_to_hindi(final_sentiment_result)

    # Step 7: Generate Audio
    progress_bar.progress(100)
    st.write("### Generating Hindi Audio...")
    audio_file = generate_direct_hindi_tts(hindi_translation)

    st.success("Analysis Complete!")

    st.write("### Hindi Translation")
    st.write(hindi_translation)

    # Provide Audio Download Link
    with open(audio_file, "rb") as file:
        st.download_button(
            "Download Hindi Audio",
            file,
            file_name="final_sentiment_hindi.mp3",
            mime="audio/mpeg",
        )
