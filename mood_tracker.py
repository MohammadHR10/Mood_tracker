# Import necessary libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize  # For sentence tokenization

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_text_sentiment(text):
    # Step 1: Tokenize text into sentences
    sentences = sent_tokenize(text)
    overall_score = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    sentence_analysis = []

    # Step 2: Analyze each sentence
    for sentence in sentences:
        sentiment = sia.polarity_scores(sentence)
        sentence_analysis.append((sentence, sentiment))
        
        # Aggregate overall scores
        for key in overall_score:
            overall_score[key] += sentiment[key]

    # Step 3: Average the overall sentiment scores
    num_sentences = len(sentences)
    for key in overall_score:
        overall_score[key] /= num_sentences

    # Determine overall mood based on compound score
    compound = overall_score['compound']
    if compound >= 0.05:
        overall_mood = 'positive'
    elif compound <= -0.05:
        overall_mood = 'negative'
    else:
        overall_mood = 'neutral'

    return overall_mood, overall_score, sentence_analysis

if __name__ == "__main__":
    # Take user input
    text = input("Enter your text: ")

    # Analyze the text
    overall_mood, overall_score, sentence_analysis = analyze_text_sentiment(text)
    
    # Output the results
    print(f"\nOverall Mood: {overall_mood}")
    print(f"Overall Scores: {overall_score}\n")
    print("Sentence-level Analysis:")
    for sentence, sentiment in sentence_analysis:
        print(f"Sentence: \"{sentence}\"")
        print(f"Sentiment: {sentiment}\n")
