import os

from bert_processor import BERTProcessor  # The BERT integration module
from data_collector import DataCollector  # Assuming this is your data collection module
from data_preprocessor import DataPreprocessor  # Your data preprocessing module
from visualizer import Visualizer  # The visualization module


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

    # 3. Perform BERT analysis (as an example)
    preprocessed_commits['commits_bert_results'] = preprocessed_commits['message'].apply(bert_processor.process_message)
    preprocessed_issues['issue_titles_bert_results'] = preprocessed_issues['title'].apply(
        bert_processor.process_message)
    preprocessed_issues['issue_bodies_bert_results'] = preprocessed_issues['body'].apply(bert_processor.process_message)

    # 4. Generate visualizations
    # This is an example, replace 'data', 'x', 'y' with actual data fields
    visualizer.plot_line_chart(data=preprocessed_commits, x=preprocessed_commits['commit_date'],
                               y=preprocessed_commits.size, title='Commit Activity Over Time')

    print("Analysis completed.")


if __name__ == "__main__":
    main()
