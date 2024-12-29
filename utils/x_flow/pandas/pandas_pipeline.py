import pandas as pd
from tqdm import tqdm
import os
from typing import Optional, Iterable, Union, Callable
from utils.x_flow.base import BasePipeline


class PandasPipeline(BasePipeline):
    """Manages flow of transformations on pandas dataframe"""

    def __init__(self, event_name: Optional[str] = None, logging_func: Callable = print,
                 use_cache: bool = True, cache_folder: str = "Data/temp"):
        super().__init__(event_name, logging_func, cache_folder)
        self.use_cache = use_cache

    @staticmethod
    def validate_df(df: pd.DataFrame, required_cols: Optional[Iterable[str]] = None) -> None:
        """Checks if the required columns are present in the DataFrame."""
        if required_cols is not None:
            missing_columns = set(required_cols) - set(df.columns)
            if missing_columns:
                raise ValueError(f"The DataFrame is missing required columns: {missing_columns}")

    def step(self, x: Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]) -> pd.DataFrame:
        if isinstance(x, pd.io.parsers.readers.TextFileReader):
            return self.process_in_chunks(x)
        elif isinstance(x, pd.DataFrame):
            return self.process_df(x)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def process_df(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("The method 'process_df' is not implemented in the derived class!")

    def process_one_chunk(self, x: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("The method 'process_one_chunk' is not implemented in the derived class!")

    def process_in_chunks(self, chunks: pd.io.parsers.readers.TextFileReader) -> pd.DataFrame:
        """Process DataFrame in chunks."""
        temp_file = os.path.join(self.cache_folder, f"{self.name}_temp.csv")
        try:
            if self.use_cache:
                with open(temp_file, mode="w", newline='', encoding="utf-8") as f_out:
                    for i, chunk in enumerate(tqdm(chunks, desc=f"{self.name}: Processing chunks")):
                        chunk = self.process_one_chunk(chunk)
                        chunk.to_csv(f_out, index=False, encoding= "utf-8", mode='a', header=(i == 0))
                self.log("All chunks processed successfully!")
                df = pd.read_csv(temp_file)
            else:
                res = []
                for chunk in tqdm(chunks, desc=f"{self.name}: Processing chunks"):
                    res.append(self.process_one_chunk(chunk))
                df = pd.concat(res, axis=0)
        finally:
            if self.use_cache and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.log(f"Failed to remove temporary file '{temp_file}': {e}")

        return df


class WriteFile(PandasPipeline):
    def __init__(self, save_path: str, write_func: Callable = pd.DataFrame.to_csv, **other_kwargs):
        super().__init__(event_name="FileSaver")

        if not (callable(write_func) and hasattr(pd.DataFrame, write_func.__name__)):
            raise TypeError(
                f"Expected a pandas.DataFrame method for write_func, but got {type(write_func).__name__}."
            )

        if not isinstance(save_path, str):
            raise TypeError(f"Expected save_path to be a string, but got {type(save_path).__name__}.")

        self.save_path = save_path
        self.write_func = write_func
        self.other_kwargs = other_kwargs

    def step(self, x: pd.DataFrame) -> str:
        try:
            self.write_func(x, self.save_path, **self.other_kwargs)
            self.log(f"Successfully written the file to {self.save_path}")
        except Exception as e:
            self.log("Error occurred during saving...\n"
                     f"Check your save_path: {self.save_path} or write_func: {self.write_func.__name__}\n"
                     f"Details: {str(e)}")
            raise e
        return self.save_path


class ReadFile(PandasPipeline):
    def __init__(self, read_func: Callable = pd.read_csv, **other_kwargs):
        event_name = "FileLoader"
        self.read_func = read_func
        self.other_kwargs = other_kwargs

        if "chunksize" in other_kwargs:
            event_name += "(in_chunks)"
        super().__init__(event_name=event_name)

    @staticmethod
    def validate_file(filepath: str) -> str:
        """Ensure the file exists."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file '{filepath}' does not exist.")
        return filepath

    def step(self, x: str) -> Union[pd.DataFrame, pd.io.parsers.TextFileReader]:
        """Read the file using the provided read_func."""
        filepath = self.validate_file(x)  # Validate file existence
        try:
            result = self.read_func(filepath, **self.other_kwargs)  # Call read_func
            self.log(f"Successfully read file: {filepath}")
        except Exception as e:
            self.log(f"Error occurred during reading...\n"
                     f"Check your file_path (x): {x} or read_func: {self.read_func.__name__}\n"
                     f"Details: {str(e)}")
            raise e
        return result
