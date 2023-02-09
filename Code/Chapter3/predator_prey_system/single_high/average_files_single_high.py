## Import values from output files ##
import sys
import os
import pandas as pd

# Read csv file
m_df = pd.read_csv("output_single_high.csv")
df = m_df.groupby(['rp','rx']).mean()
df.to_csv("Av_output_single_high.csv", index = True)

# Read csv file
m_df = pd.read_csv("coexistence_single_high.csv")
df = m_df.groupby(['rp','rx']).sum()
df.to_csv("Av_coexistence_single_high.csv", index = True)