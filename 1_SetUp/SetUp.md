# Set Up

### Import Pandas and NumPy
```py
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format="{:,.2f}".format
```

### Import Matplotlib and Seaborn
```py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('viridis')
plt.style.use('fivethirtyeight')
```
```py
# Qualitative palettes, good for representing categorical data
sns.set_palette("hls", 8)
sns.set_palette('bright')
sns.set_palette('tab10')

# Sequential palettes, good for representing numeric data
sns.set_palette('viridis')
sns.set_palette('cubehelix')
sns.set_palette('inferno')
```
