Marketing Analytics

Amalya Tadevosyan

Assignment 1

1. Choose an innovation from the list.
The link to the 2024 innovations list is here. Once you select a product from the list, add its link here.

I chose Wireless Headphone Luxury | Sonos Ace.
Link to the innovation: https://time.com/7094621/sonos-ace/

2. Identify a similar innovation from the past.
Reflect on an innovation that resembles the one you’ve chosen. In 1–2 paragraphs, justify your choice
by comparing both innovations in terms of their functionality, technology, or market impact

The Apple AirPods are a fitting comparison to the Sonos Ace Wireless Headphones, as both revolutionized the wireless audio experience with cutting-edge technology and user-centered design. The AirPods debuted in 2016, quickly gaining traction due to their seamless integration with Apple devices, compact design, and ease of use. They introduced the world to a new level of convenience in wireless audio, eliminating the need for tangled cords, and set a new standard for on-the-go listening.

On the other hand, the Sonos Ace Wireless Headphones take the concept of wireless listening further by focusing on luxury and immersive sound. Unlike the AirPods, which prioritize portability and Apple ecosystem integration, the Sonos Ace places a stronger emphasis on high-end audio performance, noise-cancellation, and premium materials, offering a more audiophile-centric experience. Both innovations have disrupted the market, but while the AirPods popularized true wireless earbuds and became synonymous with casual listening, Sonos Ace pushes the envelope in terms of sound quality and user experience, appealing to those seeking both style and substance in their audio products.

3. Find historical data.
Use Statista (the university provides access through AUA Wi-Fi) or another reliable resource to find
a time series that matches your look-alike innovation. Provide the reference for the data source.


Dataset Link: https://www.kaggle.com/datasets/kushagra1211/usa-sales-product-datasetcleaned


```python
!pip install pandas
```

    Requirement already satisfied: pandas in d:\anaconda\lib\site-packages (2.2.3)
    Requirement already satisfied: numpy>=1.26.0 in d:\anaconda\lib\site-packages (from pandas) (2.1.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in d:\anaconda\lib\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in d:\anaconda\lib\site-packages (from pandas) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in d:\anaconda\lib\site-packages (from pandas) (2025.2)
    Requirement already satisfied: six>=1.5 in d:\anaconda\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    


```python
import pandas as pd

file_path = 'Sales_Product_Combined.csv'
df = pd.read_csv(file_path)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Product</th>
      <th>Quantity Ordered</th>
      <th>Price</th>
      <th>Order Date</th>
      <th>Time</th>
      <th>Purchase Address</th>
      <th>City</th>
      <th>Product Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>176558</td>
      <td>USB-C Charging Cable</td>
      <td>2</td>
      <td>11.95</td>
      <td>19-04-2019</td>
      <td>8:46 AM</td>
      <td>917 1st St, Dallas, TX 75001</td>
      <td>Dallas</td>
      <td>Cable</td>
    </tr>
    <tr>
      <th>1</th>
      <td>176559</td>
      <td>Bose SoundSport Headphones</td>
      <td>1</td>
      <td>99.99</td>
      <td>07-04-2019</td>
      <td>10:30 PM</td>
      <td>682 Chestnut St, Boston, MA 02215</td>
      <td>Boston</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>2</th>
      <td>176560</td>
      <td>Google Phone</td>
      <td>1</td>
      <td>600</td>
      <td>12-04-2019</td>
      <td>2:38 PM</td>
      <td>669 Spruce St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Phone</td>
    </tr>
    <tr>
      <th>3</th>
      <td>176560</td>
      <td>Wired Headphones</td>
      <td>1</td>
      <td>11.99</td>
      <td>12-04-2019</td>
      <td>2:38 PM</td>
      <td>669 Spruce St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>4</th>
      <td>176561</td>
      <td>Wired Headphones</td>
      <td>1</td>
      <td>11.99</td>
      <td>30-04-2019</td>
      <td>9:27 AM</td>
      <td>333 8th St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Headphones</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_airpods = df[df['Product'] == 'Apple Airpods Headphones']

df_airpods.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Product</th>
      <th>Quantity Ordered</th>
      <th>Price</th>
      <th>Order Date</th>
      <th>Time</th>
      <th>Purchase Address</th>
      <th>City</th>
      <th>Product Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>176572</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>8:30 PM</td>
      <td>149 Dogwood St, New York City, NY 10001</td>
      <td>New York City</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>20</th>
      <td>176576</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>28-04-2019</td>
      <td>11:42 AM</td>
      <td>771 Ridge St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>21</th>
      <td>176577</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>7:25 PM</td>
      <td>260 Spruce St, Dallas, TX 75001</td>
      <td>Dallas</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>22</th>
      <td>176578</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>09-04-2019</td>
      <td>11:35 PM</td>
      <td>513 Church St, Boston, MA 02215</td>
      <td>Boston</td>
      <td>Headphones</td>
    </tr>
    <tr>
      <th>37</th>
      <td>176591</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>21-04-2019</td>
      <td>7:21 AM</td>
      <td>600 Maple St, Austin, TX 73301</td>
      <td>Austin</td>
      <td>Headphones</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_airpods['Year'] = df_airpods['Order Date'].str.split('-').str[2]
df_airpods.head()
```

    C:\Users\amaly\AppData\Local\Temp\ipykernel_15800\2218239153.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_airpods['Year'] = df_airpods['Order Date'].str.split('-').str[2]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Product</th>
      <th>Quantity Ordered</th>
      <th>Price</th>
      <th>Order Date</th>
      <th>Time</th>
      <th>Purchase Address</th>
      <th>City</th>
      <th>Product Type</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>176572</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>8:30 PM</td>
      <td>149 Dogwood St, New York City, NY 10001</td>
      <td>New York City</td>
      <td>Headphones</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>20</th>
      <td>176576</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>28-04-2019</td>
      <td>11:42 AM</td>
      <td>771 Ridge St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Headphones</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>21</th>
      <td>176577</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>7:25 PM</td>
      <td>260 Spruce St, Dallas, TX 75001</td>
      <td>Dallas</td>
      <td>Headphones</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>22</th>
      <td>176578</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>09-04-2019</td>
      <td>11:35 PM</td>
      <td>513 Church St, Boston, MA 02215</td>
      <td>Boston</td>
      <td>Headphones</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>37</th>
      <td>176591</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>21-04-2019</td>
      <td>7:21 AM</td>
      <td>600 Maple St, Austin, TX 73301</td>
      <td>Austin</td>
      <td>Headphones</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_airpods['Year-Month'] = df_airpods['Order Date'].str.split('-').apply(lambda x: f"{x[2]}-{x[1]}")

df_airpods.head()


```

    C:\Users\amaly\AppData\Local\Temp\ipykernel_15800\2810189406.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_airpods['Year-Month'] = df_airpods['Order Date'].str.split('-').apply(lambda x: f"{x[2]}-{x[1]}")
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order ID</th>
      <th>Product</th>
      <th>Quantity Ordered</th>
      <th>Price</th>
      <th>Order Date</th>
      <th>Time</th>
      <th>Purchase Address</th>
      <th>City</th>
      <th>Product Type</th>
      <th>Year</th>
      <th>Year-Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>176572</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>8:30 PM</td>
      <td>149 Dogwood St, New York City, NY 10001</td>
      <td>New York City</td>
      <td>Headphones</td>
      <td>2019</td>
      <td>2019-04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>176576</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>28-04-2019</td>
      <td>11:42 AM</td>
      <td>771 Ridge St, Los Angeles, CA 90001</td>
      <td>Los Angeles</td>
      <td>Headphones</td>
      <td>2019</td>
      <td>2019-04</td>
    </tr>
    <tr>
      <th>21</th>
      <td>176577</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>04-04-2019</td>
      <td>7:25 PM</td>
      <td>260 Spruce St, Dallas, TX 75001</td>
      <td>Dallas</td>
      <td>Headphones</td>
      <td>2019</td>
      <td>2019-04</td>
    </tr>
    <tr>
      <th>22</th>
      <td>176578</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>09-04-2019</td>
      <td>11:35 PM</td>
      <td>513 Church St, Boston, MA 02215</td>
      <td>Boston</td>
      <td>Headphones</td>
      <td>2019</td>
      <td>2019-04</td>
    </tr>
    <tr>
      <th>37</th>
      <td>176591</td>
      <td>Apple Airpods Headphones</td>
      <td>1</td>
      <td>150</td>
      <td>21-04-2019</td>
      <td>7:21 AM</td>
      <td>600 Maple St, Austin, TX 73301</td>
      <td>Austin</td>
      <td>Headphones</td>
      <td>2019</td>
      <td>2019-04</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_airpods_monthly = df_airpods.groupby('Year-Month')['Quantity Ordered'].sum().reset_index()

df_airpods_monthly.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year-Month</th>
      <th>Quantity Ordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01</td>
      <td>814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02</td>
      <td>1012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-03</td>
      <td>1322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04</td>
      <td>1519</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05</td>
      <td>1365</td>
    </tr>
  </tbody>
</table>
</div>



4. Estimate Bass Model parameters.
Using the time series data for your look-alike innovation, estimate the Bass diffusion model
parameters (coefficient of innovation p, coefficient of imitation q, and market potential M).



```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from helper_functions import bass_model
time = np.arange(len(df_airpods_monthly)) 
sales = df_airpods_monthly['Quantity Ordered'].values 

p0 = [0.02, 0.2, max(sales)]  

bounds = ([0, 0, 0], [1, 1, max(sales)])

params, covariance = curve_fit(bass_model, time, sales, p0=p0, bounds=bounds, maxfev=10000)

p, q, M = params
print(f"Estimated Parameters: p = {p}, q = {q}, M = {M}")

```

    Estimated Parameters: p = 0.9999998044279225, q = 7.75113111132301e-43, M = 1232.6925432576993
    


```python
fitted_sales = bass_model(time, p, q, M)

plt.figure(figsize=(10, 6))
plt.plot(time, sales, label='Actual Sales', color='blue', marker='o')
plt.plot(time, fitted_sales, label='Fitted Bass Model', color='red', linestyle='--')
plt.xlabel('Time (Months)')
plt.ylabel('Sales (Quantity Ordered)')
plt.title('Bass Diffusion Model Fit')
plt.legend()
plt.show()

plt.savefig('Bass_Diffusion_Model.png', bbox_inches='tight')

from IPython.display import FileLink
FileLink('Bass_Diffusion_Model.png')

```


    
![png](output_17_0.png)
    





<a href='Bass_Diffusion_Model.png' target='_blank'>Bass_Diffusion_Model.png</a><br>




    <Figure size 640x480 with 0 Axes>



```python
time_airpods = np.arange(len(df_airpods_monthly)) 
sales_airpods = df_airpods_monthly['Quantity Ordered'].values

p0_airpods = [0.02, 0.2, max(sales_airpods)]

params_airpods, covariance_airpods = curve_fit(bass_model, time_airpods, sales_airpods, p0=p0_airpods, bounds=([0, 0, 0], [1, 1, max(sales_airpods)]), maxfev=10000)

p_airpods, q_airpods, M_airpods = params_airpods
print(f"Estimated Parameters for Apple Airpods: p = {p_airpods}, q = {q_airpods}, M = {M_airpods}")

```

    Estimated Parameters for Apple Airpods: p = 0.9999998044279225, q = 7.75113111132301e-43, M = 1232.6925432576993
    


```python


# Assuming you already have df_new which contains all your product data (including Sonos Ace)

# Filter the dataset for "Wireless Headphone Luxury | Sonos Ace"
df_sonos_ace = df[df['Product'] == 'Apple Airpods Headphones']

# Convert 'Order Date' to Year-Month format
df_sonos_ace['Year-Month'] = df_sonos_ace['Order Date'].str.split('-').apply(lambda x: f"{x[2]}-{x[1]}")

# Group by 'Year-Month' and sum the sales (Quantity Ordered)
df_sonos_ace_monthly = df_sonos_ace.groupby('Year-Month')['Quantity Ordered'].sum().reset_index()

# Now we can proceed with the diffusion prediction using the parameters from Apple Airpods
time_sonos_ace = np.arange(len(df_sonos_ace_monthly))  # Time periods for Sonos Ace (months)
sales_sonos_ace = df_sonos_ace_monthly['Quantity Ordered'].values  # Monthly sales data for Sonos Ace

# Using the Apple Airpods' parameters (p, q, M)
cumulative_adoption_sonos_ace = bass_model(time_sonos_ace, p_airpods, q_airpods, M_airpods)

new_adopters_sonos_ace = np.diff(np.concatenate(([0], cumulative_adoption_sonos_ace)))

time_predict_sonos_ace = np.arange(len(df_sonos_ace_monthly) + 12)  # Adding 12 months for future prediction

predicted_cumulative_adoption_sonos_ace = bass_model(time_predict_sonos_ace, p_airpods, q_airpods, M_airpods)

predicted_new_adopters_sonos_ace = np.diff(np.concatenate(([0], predicted_cumulative_adoption_sonos_ace)))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(time_predict_sonos_ace, predicted_cumulative_adoption_sonos_ace, label="Predicted Cumulative Adoption (Sonos Ace)", color='green')
plt.xlabel('Time (Months)')
plt.ylabel('Cumulative Adopters')
plt.title('Predicted Diffusion Path for Sonos Ace (Cumulative Adoption)')
plt.legend()
predicted_new_adopters_sonos_ace_fixed = predicted_new_adopters_sonos_ace[:-1]  # Removing the last value to align with time

plt.subplot(1, 2, 2)
plt.plot(time_predict_sonos_ace[1:], predicted_new_adopters_sonos_ace_fixed, label="Predicted New Adopters (Sonos Ace)", color='orange')
plt.xlabel('Time (Months)')
plt.ylabel('New Adopters (Sales)')
plt.title('Predicted New Adopters (Sales) for Sonos Ace')
plt.legend()

plt.tight_layout()
plt.show()


```

    C:\Users\amaly\AppData\Local\Temp\ipykernel_15800\470564470.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_sonos_ace['Year-Month'] = df_sonos_ace['Order Date'].str.split('-').apply(lambda x: f"{x[2]}-{x[1]}")
    


    
![png](output_19_1.png)
    


Choose a scope (global or country-specific).
Decide whether to analyze the diffusion worldwide or within a specific country. Justify your choice
with references or data.

When deciding whether to analyze the diffusion of **Sonos Ace** globally or within a specific country, we should consider the nature of the product and the data available. Analyzing the diffusion globally could give us a broad perspective, allowing us to understand how the product spreads across different regions and how worldwide factors, like global marketing campaigns or trends, influence adoption. However, this approach can be complex, as various countries have differing economic conditions, cultural preferences, and adoption speeds, which could make global predictions less accurate. Moreover, reliable global sales data can sometimes be hard to obtain.

On the other hand, a **country-specific analysis** is more focused and allows for a deeper understanding of how the product will perform in a particular market. It’s especially useful for products like **Sonos Ace**, which is a premium product that might have different adoption rates in different countries due to factors like income levels, local competition, and consumer behavior. By analyzing the diffusion in a specific country, we can account for local preferences, price sensitivity, and regional marketing strategies. This approach will give us more actionable insights, especially if the product is launching in a new country or region.

Given the characteristics of **Sonos Ace**, it makes sense to perform a **country-specific analysis**, focusing on a key market like the **US**, where the product has already been launched or is expected to launch. This would allow us to make predictions based on the local market conditions and better understand the potential adoption rate within that specific context.



```python
import numpy as np
import matplotlib.pyplot as plt

p = 0.02 
q = 0.2  
M = 1232.6925432576933  

time_periods = np.arange(1, 25)  

cumulative_adopters = np.zeros(len(time_periods))

new_adopters = np.zeros(len(time_periods))

for t in range(1, len(time_periods)):
    cumulative_adopters[t] = cumulative_adopters[t-1] + new_adopters[t-1]
    new_adopters[t] = p * (M - cumulative_adopters[t-1]) + q * cumulative_adopters[t-1] * (cumulative_adopters[t-1] / M)

plt.figure(figsize=(10, 6))

plt.plot(time_periods, new_adopters, label='New Adopters (Sales)', color='blue')

plt.plot(time_periods, cumulative_adopters, label='Cumulative Adopters', color='green')

plt.xlabel('Time (Months)')
plt.ylabel('Number of Adopters')
plt.title('Estimated Number of Adopters Over Time (Bass Model)')
plt.legend()
plt.grid(True)

plt.show()

```


    
![png](output_22_0.png)
    

