from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from schema import Article, ComparativeSentimentScore

article_template = PromptTemplate(
    template="""
    Analyze this news article content:
    {content}

    Generate output in JSON format with:
    - A Title
    - 30-word summary
    - sentiment (positive/negative/neutral)
    - 3 key topics

    {format_instructions}

    \nOnly return JSON object, nothing else.
    """,
    input_variables=["content"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=Article
        ).get_format_instructions()
    },
)


coverage_diff_template = PromptTemplate(
    template="""Analyze these news articles about {company} and identify key differences in coverage:

**Articles:**
{articles}

**Instructions:**
1. Compare articles in pairs (Article 1 vs 2, 1 vs 3, 2 vs 3, etc.)
2. Focus on contrasts in:
   - Sentiment polarity
   - Key topics emphasized
   - Narrative focus
   - Tone implications
3. Highlight 3-5 most significant differences
4. For each difference:
   - Explain the comparison clearly
   - Describe potential business impact
   - Use professional business analysis language

**Article Format (for reference):**
{{
  "Title": "...",
  "Summary": "...",
  "Sentiment": "...",
  "Topics": ["...", "..."]
}}
**Important Rules:**
- Never invent information not present in articles
- Maintain original sentiment labels
- Use exact topic names from articles

**Example Output:**

  "Coverage Differences": [
    {{
      "Comparison": "Article 1 focuses on financial success while Article 2 emphasizes regulatory challenges",
      "Impact": "Investors may see conflicting signals about growth potential vs operational risks"
    }},
    {{
      "Comparison": "Article 1 highlights product innovation whereas Article 3 discusses supply chain issues",
      "Impact": "Shows tension between R&D capabilities and operational execution challenges"
    }}
  ]


**Required Output Format (JSON):**
{format_instructions}

Only return JSON object, nothing else.""",
    input_variables=["company", "articles"],
    partial_variables={
        "format_instructions": PydanticOutputParser(
            pydantic_object=ComparativeSentimentScore
        ).get_format_instructions()
    },
)


final_sentiment_prompt = PromptTemplate(
    template="""Synthesize a comprehensive final sentiment analysis for {company} based on the following data:

    **Articles Analysis:**
    {articles}

    **Comparative Analysis:**
    {comparative_analysis}

    **Instructions:**
    1. Start with overall sentiment assessment using sentiment distribution
    2. Explain key conflicting narratives from coverage differences
    3. Highlight dominant topics and their implications
    4. Address investor/consumer impact potential
    5. Conclude with forward-looking statement
    6. Maintain professional, analytical tone
    7. Keep it concise, within 1 paragraph

    **Required Structure:**
    - Overall Sentiment Balance: [Positive/Negative/Neutral] leaning due to [X positive vs Y negative articles]
    - Key Contradictions: [Highlight 2-3 major conflicting narratives from coverage differences]
    - Dominant Themes: [Most frequent topics and their sentiment correlations]
    - Risk Assessment: [Critical risks from negative coverage]
    - Growth Potential: [Opportunities from positive coverage]
    - Final Outlook: [Synthesized prediction/forecast]

    **Example Output:**
    "Tesla's news coverage shows predominantly negative sentiment (6 negative vs 2 positive articles), though with notable bullish counter-narratives. The primary contradiction emerges between analyst optimism about Tesla's AI/robotics potential and widespread criticism of Elon Musk's leadership controversies..."

    **Your Output (NO MARKDOWN, plain text):**""",
    input_variables=["company", "articles", "comparative_analysis"],
)
