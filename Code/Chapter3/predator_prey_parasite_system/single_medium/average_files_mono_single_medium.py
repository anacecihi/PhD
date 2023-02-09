## Import values from output files ##
import sys
import os
import pandas as pd

# Read csv file
m_df = pd.read_csv("output_mono_single_medium.csv")
df = m_df.groupby(['rp','rx']).mean()
df.to_csv("Av_output_mono_single_medium.csv", index = True)

# Read csv file
m_df = pd.read_csv("coexistence_mono_single_medium.csv")
df = m_df.groupby(['rp','rx']).sum()
df.to_csv("Av_coexistence_mono_single_medium.csv", index = True)