from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time

# Traits and rubric
traits = ["relevance", "organization", "vocabulary", "style", "development", "mechanics", "grammar"]

rubric = {
    "relevance_rubric": {1: "Partially related to the essay_prompt", 2: "Fully related to the essay_prompt"},
    "organization_rubric": {
        1: "The introduction and conclusion may be absent, and the body lacks organization and coherence between paragraphs",
        2: "The introduction or conclusion may be absent, and the body lacks organization and coherence between paragraphs",
        3: "Acceptably organized text with an introduction and conclusion, and a body of one or two paragraphs lacking strong coherence",
        4: "Well-organized text with an introduction that sets up the topic and an appropriate conclusion, and a body of two to three paragraphs with sequential and coherent flow",
        5: "Well-organized text with an introduction that sets up the topic and a conclusion that leads to effective insights, and a body of two to three paragraphs with good coherence and flow"
    },
    "vocabulary_rubric": {
        1: "Limited use of vocabulary and phrases that do not form a clear meaning, with frequent repetition and lexical errors, and generally inappropriate word choices that obscure meaning",
        2: "Basic vocabulary use with repetition and lexical errors, and many inappropriate word choices that may obscure meaning",
        3: "Adequate vocabulary use with some repetition and lexical errors, and a few inappropriate word choices that may obscure meaning",
        4: "Good and appropriate vocabulary use with few lexical errors, some inappropriate choices that do not affect meaning, and occasional use of idiomatic expressions",
        5: "Extensive, accurate, and appropriate vocabulary use with few minor errors, good command of idioms, and understanding of implied meanings"
    },
    "style_rubric": {
        1: "Only very basic linear connectors used such as 'and', 'or', 'then', etc.",
        2: "Discourse developed as a simple list of points using only common connectors",
        3: "Discourse developed as a straightforward linear sequence using common cohesive devices",
        4: "Clear discourse development with main points supported by relevant details, consistently appropriate use of organizational patterns and cohesive devices, with occasional jumps in long sentences",
        5: "Well-crafted discourse development with clear subtopics, detailed explanations, sound conclusions, consistently appropriate use of various organizational patterns and a wide range of cohesive devices"
    },
    "development_rubric": {
        1: "Content is unrelated to the question topic, ideas are random and disconnected, main idea is completely absent, and writing is overly general with no relevance to expository writing",
        2: "Content is somewhat related to the question topic, ideas are somewhat connected, partial organization and sequencing, main idea appears in the text, but lacks support through examples or explanations to enhance clarity",
        3: "Content is fully related to the question topic, main idea is evident in most of the text, some information is organized and connected, with explanations that support and relate to the topic",
        4: "Content is fully related to the topic, ideas are clear, organized, sequential, and connected, main idea is maintained throughout, with effective presentation and connection of information and supporting ideas",
        5: "Content is fully related to the topic, ideas are clear, well-organized, sequential and connected, main idea is maintained, information is accurate and coherent, includes objective analysis, avoids personal opinion, and supports arguments with multiple logical explanations and statistics that enhance clarity"
    },
    "mechanics_rubric": {
        1: "Limited application of spelling rules",
        2: "Frequent spelling and punctuation errors",
        3: "Effective application of standard formatting, paragraphing, spelling, and punctuation most of the time",
        4: "Effective application of standard formatting, paragraphing, spelling, and punctuation with few errors",
        5: "Highly accurate paragraphing, punctuation, and spelling with only occasional slips"
    },
    "grammar_rubric": {
        1: "Limited use of simple grammatical structures and sentence patterns with little flexibility or accuracy",
        2: "Correct use of some simple grammatical structures with frequent systematic errors that may obscure meaning",
        3: "Use of varied grammatical structures with noticeable errors and inaccuracies that may sometimes obscure meaning",
        4: "Good use of a variety of grammatical structures with rare errors and minor sentence structure flaws that do not generally affect meaning",
        5: "Consistent and flexible use of a wide range of grammatical structures with only occasional minor slips"
    }
}

def append_output_to_df(output_text, df):
    try:
        data = dict(line.split(": ") for line in output_text.strip().split("\n"))
        data = {k.strip(): int(v.strip()) for k, v in data.items()}
        return pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    except Exception as e:
        print("Error parsing output_text:", output_text)
        print("Exception:", e)
        return df

#use english essay_prompt
def evaluate_essays(API_key, essays, essay_prompt, essay_type, traits, rubric, examples_count=10, sleep_sec=1, save_path=None):
    import time
    from tqdm import tqdm
    import pandas as pd


    client = OpenAI(api_key=API_key)

    if examples_count != 0:
        examples = essays.sample(frac=1).head(examples_count)
    else:
        examples = ""

    results_df = pd.DataFrame()
    system_message = {
        "role": "system",
        "content": "You are an expert Arabic language teacher responsible for evaluating Arabic essays written by students based on specific traits and rubrics."
    }

    fixed_user_message = {
        "role": "user",
        "content": f"""Think step-by-step about the following criteria then start scoring the provided essay:
-Essays are evaluated on the following traits: {traits}.
-Each trait is described in this rubric (dictionary of trait:score-explanation pairs): {rubric}.
-The essay was written in response to the following prompt: {essay_prompt}
-Essay type: {essay_type}

Scoring-steps:
1. check the first trait and its rubric
2. read the essay
3. provide a score
4. repeat step 1 for each trait.
5. After scoring all traits, format the output as follows:
<trait_name>: <score>
Do not provide any additional text or explanation.

{examples}
"""
    }

    for essay in tqdm(essays):
        essay_message = {"role": "user", "content": f"Essay: {essay}"}
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[system_message, fixed_user_message, essay_message]
        )
        output_text = response.choices[0].message.content
        results_df = append_output_to_df(output_text, results_df)
        time.sleep(sleep_sec)

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    return results_df

# Example usage:
# explanatory_results_df = evaluate_essays(client, essays_explanatory, essay_prompt_explanatory, essay_type_explanatory, traits, rubric, save_path="explanatory_results.csv")
# persuasive_results_df = evaluate_essays(client, essays_persuasive, essay_prompt_persuasive, essay_type_persuasive, traits, rubric, save_path="persuasive_results.csv")
