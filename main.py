import os
import pandas as pd
from bert_processor import BERTProcessor  # The BERT integration module
from data_collector import DataCollector  # Assuming this is your data collection module
from data_preprocessor import DataPreprocessor  # Your data preprocessing module
from visualizer import Visualizer  # The visualization module
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Set up argument parser
    # parser = argparse.ArgumentParser(description="Software Repository Analyzer")
    # parser.add_argument('repo_url', help="URL of the GitHub repository")
    # args = parser.parse_args()

    # Initialize modules
    # data_collector = DataCollector(os.getenv("GITHUB_TOKEN"), args.repo_url)
    data_collector = DataCollector(os.getenv("GITHUB_TOKEN"), "mert-aydin/SWE-573")
    data_preprocessor = DataPreprocessor()
    bert_processor = BERTProcessor()
    visualizer = Visualizer()

    # Example workflow
    # 1. Collect data
    commits = data_collector.get_commits()
    issues = data_collector.get_issues()

    # 2. Preprocess data
    preprocessed_commits = data_preprocessor.preprocess_commits(commits)
    preprocessed_issues = data_preprocessor.preprocess_issues(issues)

    # 3. Perform BERT analysis
    commit_embeddings = bert_processor.encode_texts(preprocessed_commits['message'].tolist())
    issue_embeddings = bert_processor.encode_texts(
        (preprocessed_issues['title'] + " " + preprocessed_issues['body']).tolist())
    similarity_matrix = bert_processor.calculate_similarity(commit_embeddings[0].reshape(1, -1), issue_embeddings)

    # 4. Generate visualizations
    visualizer.plot_heatmap(similarity_matrix, "Commit-Issue Similarity", "Issues", "Commits")

    # Assuming 'commit_texts' and 'issue_texts' contain the corresponding texts
    commit_to_issue_matches = []
    for commit_index, scores in enumerate(similarity_matrix):
        highest_similarity_index = scores.argmax()
        highest_similarity_score = scores[highest_similarity_index]
        commit_text = preprocessed_commits['message'][commit_index]
        issue_text = (preprocessed_issues['title'] + " " + preprocessed_issues['body'])[highest_similarity_index]
        commit_to_issue_matches.append({
            'Commit': commit_text,
            'Matched Issue': issue_text,
            'Similarity Score': highest_similarity_score
        })

    matches_df = pd.DataFrame(commit_to_issue_matches)
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
