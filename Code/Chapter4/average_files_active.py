## Import values from output files ##
import sys
import os
import pandas as pd

# Read csv file
m_df = pd.read_csv("output_active.csv")
df = m_df.groupby(['rp1','rp2','rx']).mean()
df.to_csv("Av_output_active.csv", index = True)

# Read csv file
m_df = pd.read_csv("coexistence_active.csv")
df = m_df.groupby(['rp1','rp2','rx']).sum()
df.to_csv("Av_coexistence_active.csv", index = True)