## Import values from output files ##
import sys
import os
import pandas as pd

# Read csv file
m_df = pd.read_csv("output_multiple_medium.csv")
df = m_df.groupby(['rp','rx']).mean()
df.to_csv("Av_output_multiple_medium.csv", index = True)

# Read csv file
m_df = pd.read_csv("coexistence_multiple_medium.csv")
df = m_df.groupby(['rp','rx']).sum()
df.to_csv("Av_coexistence_multiple_medium.csv", index = True)