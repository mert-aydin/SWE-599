import os

import pandas as pd

from bert_processor import BERTProcessor
from data_collector import DataCollector
from data_preprocessor import preprocess_commits, preprocess_issues


def main():
    # Set up argument parser
    # parser = argparse.ArgumentParser(description="Software Repository Analyzer")
    # parser.add_argument('repo_url', help="URL of the GitHub repository")
    # args = parser.parse_args()

    # Initialize modules
    # data_collector = DataCollector(os.getenv("GITHUB_TOKEN"), args.repo_url)
    data_collector = DataCollector(os.getenv("GITHUB_TOKEN"), "mert-aydin/SWE-573")
    bert_processor = BERTProcessor(model_name='microsoft/codebert-base')

    # 1. Collect data
    commits = data_collector.get_commits()
    issues = data_collector.get_issues()

    # 2. Preprocess data
    preprocessed_commits = preprocess_commits(commits)
    preprocessed_issues = preprocess_issues(issues)

    # 3. Perform BERT analysis
    commit_embeddings = bert_processor.encode_texts(preprocessed_commits['message'].tolist())
    issue_embeddings = bert_processor.encode_texts(preprocessed_issues['title_body'].tolist())
    similarity_matrix = bert_processor.calculate_similarity(commit_embeddings, issue_embeddings)

    # 4. Generate visualizations
    # plot_heatmap(similarity_matrix, "Commit-Issue Similarity", "Issues", "Commits")

    # BertScore
    import numpy as np

    # Initialize matrix to store scores
    bert_scores = bert_processor.compute_pairwise_bertscore(preprocessed_commits['message'],
                                                            preprocessed_issues['title_body'])

    print(bert_scores)
    return

    commit_to_issue_matches = []
    for commit_index, scores in enumerate(similarity_matrix):
        highest_similarity_index = scores.argmax()
        highest_similarity_score = scores[highest_similarity_index]
        commit_text = preprocessed_commits['message'][commit_index]
        issue_text = preprocessed_issues['title'][highest_similarity_index]
        commit_to_issue_matches.append({
            'Commit': commit_text,
            'Matched Issue': issue_text,
            'Similarity Score': highest_similarity_score
        })

    pd.set_option('display.max_colwidth', None)
    matches_df = pd.DataFrame(commit_to_issue_matches)

    # Sort the DataFrame by the 'Similarity Score' column in descending order
    matches_df = matches_df.sort_values(by='Similarity Score', ascending=False)

    # Reset the index if you want to reindex the DataFrame
    matches_df = matches_df.reset_index(drop=True)

    matches_df.to_csv('matches.csv', index=False)
    print(matches_df)

    return

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plotting similarity scores of matched issues for the first few commits for visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=matches_df.head(10), x='Similarity Score', y='Commit')
    plt.title('Top Commit-Issue Matches')
    plt.show()

    print("Analysis completed.")


if __name__ == "__main__":
    main()
