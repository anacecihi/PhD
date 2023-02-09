## Import values from output files ##
import sys
import os
import pandas as pd

# Read csv file
m_df = pd.read_csv("output_Qx_Qy_rp1.csv")
df = m_df.groupby(['Qx','Qy','rp']).sum()
df.to_csv("Average_output_Qx_Qy_rp1.csv", index = True)