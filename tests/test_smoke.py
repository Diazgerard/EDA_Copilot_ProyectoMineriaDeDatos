from eda_copilot.profile import summarize_dataframe
import pandas as pd

def test_summarize_dataframe_basic():
    df = pd.DataFrame({
        "a": [1,2,3,4,5],
        "b": [1,1,2,2,3],
        "c": ["x","y","x","z","x"]
    })
    s = summarize_dataframe(df)
    assert s["shape"]["rows"] == 5
    assert "a" in s["numeric_cols"]
    assert "c" in s["cat_cols"]
