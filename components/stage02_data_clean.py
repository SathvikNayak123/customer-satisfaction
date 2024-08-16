import logging
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step


@step
def clean_data(data: pd.DataFrame):
    pass