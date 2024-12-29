import pandas as pd
import re
from utils.x_flow.pandas import PandasPipeline


class DataCleaner(PandasPipeline):
    def __init__(self, get_url=True, drop_copies=True, drop_null=True, drop_empty_reviews=True, include_raw_text=True):
        super().__init__()
        self.get_url = get_url
        self.drop_copies = drop_copies
        self.drop_null = drop_null
        self.drop_empty_reviews = drop_empty_reviews
        self.include_raw_text = include_raw_text

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ['app_id', 'app_name', 'review_text', 'review_score']
        self.validate_df(df, required_cols=required_columns)
        df = df[required_columns]

        if self.drop_copies:
            df = df.drop_duplicates()
        if self.drop_null:
            df = df.dropna(subset=['review_text'])
        else:
            df = df.fillna(value="<MISSING>")

        target_values = set(df['review_score'].unique())
        if len(target_values) > 2:
            raise KeyError(f"column review_score should contain either [1, 0] or [-1, 1]. got {target_values}")
        if target_values.issubset({-1, 1}):
            df['review_score'] = df['review_score'].apply(lambda x: 0 if x == -1 else x)
        elif target_values.issubset({0, 1}):
            pass
        else:
            raise KeyError(f"column review_score should contain either [1, 0] or [-1, 1]. got {target_values}")

        if self.get_url:
            df['urls'] = df['review_text'].apply(self.extract_url)
            df['contains_url'] = df['urls'].apply(lambda x: len(x) > 0)

        if self.include_raw_text:
            df['raw_text'] = df['review_text']
        df['review_text'] = df['review_text'].apply(self.remove_url)
        df['review_text'] = df['review_text'].apply(self.basic_preprocess)
        df['review_length'] = df['review_text'].apply(len)
        df['word_counts'] = df['review_text'].apply(lambda x: len(x.split()))
        if self.drop_empty_reviews:
            df = df[df['word_counts'] > 0]
        return df

    def process_one_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        return self.process_df(chunk)

    @staticmethod
    def basic_preprocess(string):
        string = re.sub(r"\.(?!\d)", "", string)  # Remove single dots that are not part of numbers
        string = re.sub(r"[^a-zA-Z\s\u263a-\U0001F9FF0-9/\.,:)=()]", "", string.lower())  # Retain specific characters
        return string

    @staticmethod
    def extract_url(string):
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex, string)
        return [x[0] for x in url]

    @staticmethod
    def remove_url(string):
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        return re.sub(regex, "", string).strip()
