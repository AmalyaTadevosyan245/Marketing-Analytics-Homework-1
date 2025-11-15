# Marketing Analytics Homework 3 – Survival Analysis

## Overview
This project applies **Survival Analysis** techniques to predict customer retention and calculate **Customer Lifetime Value (CLV)** using the **Weibull Accelerated Failure Time (AFT)** model. The dataset consists of customer information, including demographics, subscription details, and churn behavior.

The main objective of this project is to understand customer churn patterns, segment customers based on their likelihood of survival, and estimate their CLV.

## Files
- **`HW3.ipynb`**: The Jupyter notebook where the analysis is performed. It covers:
  - Data preprocessing and exploration
  - Fitting various AFT models (e.g., Exponential, Weibull, LogNormal) for survival analysis
  - CLV calculations based on survival probability
  - Visualizations of survival curves and CLV
- **`requirements.txt`**: Lists the required Python libraries to run the project, including **pandas**, **numpy**, **lifelines**, and others.
- **`README.md`**: This file, containing an overview of the project and its purpose.

## Libraries
The project uses the following libraries:
- **pandas** for data manipulation and analysis
- **numpy** for numerical computations
- **lifelines** for survival analysis models (e.g., Weibull, Exponential)
- **matplotlib** and **seaborn** for data visualization

## Installation
To run this project, clone the repository and install the necessary libraries using the following command:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AmalyaTadevosyan245/Marketing-Analytics-Homework-1.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Marketing-Analytics-Homework-1
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Jupyter notebook**:
    ```bash
    jupyter notebook HW3.ipynb
    ```

## Data
The dataset includes the following columns:
- **`ID`**: Customer ID
- **`region`**: Customer's region
- **`tenure`**: Duration of the customer's subscription (in months)
- **`age`**: Customer's age
- **`marital`**: Marital status (e.g., Married, Unmarried)
- **`income`**: Customer's annual income (in thousands)
- **`ed`**: Education level
- **`retire`**: Retirement status (Yes/No)
- **`gender`**: Customer's gender
- **`churn`**: Whether the customer churned (1 if yes, 0 if no)

## Data Preprocessing
The following preprocessing steps were performed:
- **Label Encoding**: Applied to binary categorical variables (e.g., `retire`, `gender`, `voice`, `internet`, `forward`).
- **One-Hot Encoding**: Applied to nominal categorical variables (e.g., `region`, `marital`, `ed`, `custcat`).

## Models Used
- **Exponential AFT Model**: Used for survival analysis of customer retention.
- **Weibull AFT Model**: Chosen as the final model for stability and realistic survival curves.
- **LogNormal AFT Model**: Tested but found to produce unrealistic survival times.
- **LogLogistic, Kaplan-Meier, and other models**: Tested for comparison.

Weibull AFT was the best-performing model, providing stable survival estimates and realistic customer lifetime predictions.

## Results
- **Customer churn**: Out of 1,000 customers, 274 (27.4%) churned, and 726 (72.6%) did not churn.
- **Customer Lifetime Value (CLV)**: CLV was calculated for different segments. Female customers have a slightly higher CLV compared to male customers.

The **Weibull model** was selected as the final model due to its stability and realistic survival curves. The survival analysis revealed significant insights into customer retention.

## Conclusion
This project provides valuable insights into customer churn and the potential lifetime value of each customer. By understanding survival probabilities, businesses can target retention strategies toward high-value customer segments. The Weibull model was found to be the most reliable for this analysis.

### Next Steps and Future Work:
- Fine-tuning the model with additional features.
- Exploring machine learning models for more accurate churn predictions.
- Implementing retention strategies based on model results.

## How to Contribute
Feel free to fork the repository, make changes, and submit pull requests for improvements or extensions to the project.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
