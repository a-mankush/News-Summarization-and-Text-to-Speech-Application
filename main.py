import pickle
from pprint import pprint

from dotenv import load_dotenv
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

load_dotenv()

COMPANY = "apple"

if __name__ == "__main__":
    print(" GETTING ARTICLE  ")
    article_links = get_article_links(COMPANY)
    # print(article_links)

    article = get_article_text(article_links)
    # print(article)

    print("  LOADING MODEL  ")
    model = load_hf_model()

    article_parser = PydanticOutputParser(pydantic_object=Article)
    comp_diff_parser = PydanticOutputParser(pydantic_object=ComparativeSentimentScore)

    article_chain = article_template | model | article_parser

    print("  GETTING ARTICLE RESULT")
    article_result = []
    for article in article:
        try:
            article_result.append(dict(article_chain.invoke({"content": article})))
        except:
            print(f"unable to analyse the text.\ntext:{article[:100]}")

    # print(article_result)

    print("COMP DIFF CHAIN")
    comp_diff_chain = coverage_diff_template | model | comp_diff_parser
    comp_diff_result = comp_diff_chain.invoke(
        {"company": COMPANY, "articles": article_result}
    )

    comp_diff_list = []
    print("GETTING COMP DIF LIST")
    for value in dict(comp_diff_result).values():
        comp_diff_list.append(dict(value))

    # print(comp_diff_list)
    print("SENTIMENT DIST")
    sentiment_dist = cal_sentiment_dist(article_result=article_result)

    sentiment_dist = comparative_sentiment_score = {
        "Coverage Differences": comp_diff_list,
        "sentiment_distribution": sentiment_dist,
    }
    # print(f"==>> comparative_sentiment_score: {comparative_sentiment_score}")
    print("FINAL SENTIMENT")
    final_sentiment_chain = final_sentiment_prompt | model | StrOutputParser()
    final_sentiment_result = final_sentiment_chain.invoke(
        {
            "company": COMPANY,
            "articles": article_result,
            "comparative_analysis": comparative_sentiment_score,
        }
    )

    # pprint(final_sentiment_result)
    print("COMBINING REUSULT")
    final_result = final_dict_result(
        company_name=COMPANY,
        article_result=article_result,
        sentiment_dist=sentiment_dist,
        cov_diff=comp_diff_list,
        final_sentiment=final_sentiment_result,
    )

    with open(r"final_result.pickle", "wb") as output_file:
        pickle.dump(final_result, output_file)

    pprint(final_result)

    # english_text = final_result["Final Sentiment Analysis"]

    # hindi_translation = translate_to_hindi(english_text)
    # print("Translated Hindi Text:", hindi_translation)

    # audio_file = generate_direct_hindi_tts(hindi_translation)
    # print(f"Direct Hindi TTS audio generated: {audio_file}")
